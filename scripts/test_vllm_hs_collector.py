#!/usr/bin/env python3
"""
Test vLLM hidden states extraction with Mooncake KV connector.

Uses vLLM's public APIs — no patching:
  - speculative_config with method=extract_hidden_states
  - kv_transfer_config pointing to our MooncakeHiddenStatesConnector

Tests 3 stages independently:
  Stage 1: Import check (no GPU, no Mooncake)
  Stage 2: Mooncake store put/get/remove cycle
  Stage 3: Full pipeline — vLLM prefill → KV connector → Mooncake → verify

Prerequisites:
  pip install vllm>=0.12.0 mooncake torch transformers

Usage:
  # Stage 1 only (import check):
  python scripts/test_vllm_hs_collector.py --stage 1

  # Stage 2 (Mooncake store, needs mooncake_master running):
  python scripts/test_vllm_hs_collector.py --stage 2

  # Stage 3 (full pipeline, needs GPU + mooncake_master):
  python scripts/test_vllm_hs_collector.py --stage 3 --model-path Qwen/Qwen2.5-0.5B-Instruct

  # All stages:
  python scripts/test_vllm_hs_collector.py --stage all --model-path Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Test vLLM HS collector pipeline")
    parser.add_argument("--stage", default="1", help="1, 2, 3, or all")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--aux-layers", default=None, help="Comma-separated layer IDs (auto-detected if omitted)")
    parser.add_argument("--master-host", default="localhost")
    parser.add_argument("--master-port", type=int, default=50051)
    parser.add_argument("--metadata-port", type=int, default=8090)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--seq-len", type=int, default=128, help="For Mooncake store test (stage 2)")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="For Mooncake store test (stage 2)")
    return parser.parse_args()


def header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ─── Stage 1: Import check ────────────────────────────────────

def stage1_imports():
    header("Stage 1: Import check")
    ok = True

    checks = [
        ("verl.utils.mooncake.config", "MooncakeConfig"),
        ("verl.utils.mooncake.eagle_store", "EagleMooncakeStore"),
        ("verl.utils.mooncake.eagle_store", "Eagle3TargetOutput"),
        ("verl.utils.mooncake.hidden_states_connector", "MooncakeHiddenStatesConnector"),
        ("verl.utils.mooncake.helpers", "calculate_eagle3_buffer_size"),
        ("verl.utils.mooncake.store", "MooncakeHiddenStateStore"),
        ("verl.utils.mooncake.buffers", "HostBufferPool"),
        ("verl.utils.mooncake.deferred_delete", "DeferredDeleteManager"),
        ("verl.models.eagle3.ops.loss", "compiled_forward_kl_loss"),
        ("verl.models.eagle3.draft.base", "Eagle3DraftModel"),
        ("verl.trainer.drafter.controller", "DrafterDataController"),
    ]

    for module, name in checks:
        try:
            mod = __import__(module, fromlist=[name])
            getattr(mod, name)
            print(f"  {module}.{name}: OK")
        except Exception as e:
            print(f"  {module}.{name}: FAIL ({e})")
            ok = False

    # vLLM-specific
    try:
        from vllm import LLM, SamplingParams
        print(f"  vllm.LLM: OK")
    except ImportError as e:
        print(f"  vllm.LLM: FAIL ({e})")
        ok = False

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
        print(f"  vllm KVConnectorBase_V1: OK")
    except ImportError as e:
        print(f"  vllm KVConnectorBase_V1: FAIL ({e})")
        ok = False

    # DrafterDataController quick test
    try:
        from verl.trainer.drafter.controller import DrafterDataController, SampleMeta
        ctrl = DrafterDataController(dp_size=4)
        assert ctrl.sample_pool_size == 0
        assert ctrl.drain_as_dataproto() is None
        ctrl.push_samples([SampleMeta(mooncake_key="test", shapes={}, dtypes={}, seq_len=10)])
        proto = ctrl.drain_as_dataproto()
        assert proto is not None
        assert len(proto) == 1
        print(f"  DrafterDataController: OK (push/drain works)")
    except Exception as e:
        print(f"  DrafterDataController: FAIL ({e})")
        ok = False

    # Buffer size calc
    try:
        from verl.utils.mooncake.helpers import calculate_eagle3_buffer_size
        size = calculate_eagle3_buffer_size(max_seq_len=8192, batch_size=8, hidden_dim=4096)
        print(f"  calculate_eagle3_buffer_size: OK ({size / 1024**2:.0f} MB for 8 samples)")
    except Exception as e:
        print(f"  calculate_eagle3_buffer_size: FAIL ({e})")
        ok = False

    return ok


# ─── Stage 2: Mooncake store ──────────────────────────────────

def stage2_mooncake(args):
    header("Stage 2: Mooncake store put/get/remove")

    import torch
    from verl.utils.mooncake.config import MooncakeConfig
    from verl.utils.mooncake.eagle_store import EagleMooncakeStore

    config = MooncakeConfig.from_master_address(
        master_host=args.master_host,
        master_port=args.master_port,
        metadata_port=args.metadata_port,
        protocol="tcp",
        max_seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
    )
    print(f"  Mooncake master: {config.master_server_address}")
    print(f"  Protocol: {config.protocol}")

    store = EagleMooncakeStore(config)
    print(f"  Connecting...")
    store.setup()
    print(f"  Connected.\n")

    # Create test tensors
    seq_len, hidden_dim, num_aux = args.seq_len, args.hidden_dim, 3
    hidden_states = torch.randn(seq_len, hidden_dim * num_aux, dtype=torch.bfloat16)
    input_ids = torch.randint(0, 32000, (seq_len,), dtype=torch.int64)
    last_hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)

    print(f"  Test tensors:")
    print(f"    hidden_states:      {hidden_states.shape} {hidden_states.dtype}")
    print(f"    input_ids:          {input_ids.shape} {input_ids.dtype}")
    print(f"    last_hidden_states: {last_hidden_states.shape} {last_hidden_states.dtype}")

    # PUT
    key = "test_stage2_001"
    t0 = time.time()
    metadata = store.put(key=key, hidden_states=hidden_states, input_ids=input_ids, last_hidden_states=last_hidden_states)
    store.flush()
    dt = (time.time() - t0) * 1000
    print(f"\n  PUT key={key} ({dt:.1f} ms)")
    print(f"    shapes: {metadata['shapes']}")

    # GET
    t0 = time.time()
    output = store.get(key=key, shapes=metadata["shapes"], dtypes=metadata["dtypes"], device=torch.device("cpu"))
    dt = (time.time() - t0) * 1000
    print(f"  GET key={key} ({dt:.1f} ms)")
    print(f"    hidden_states:      {output.hidden_states.shape}")
    print(f"    input_ids:          {output.input_ids.shape}")
    if output.last_hidden_states is not None:
        print(f"    last_hidden_states: {output.last_hidden_states.shape}")

    # Verify
    hs_ok = torch.allclose(hidden_states.float(), output.hidden_states.float(), atol=1e-2)
    ids_ok = torch.equal(input_ids, output.input_ids)
    print(f"\n  Integrity: hidden_states={'OK' if hs_ok else 'FAIL'}, input_ids={'OK' if ids_ok else 'FAIL'}")

    # REMOVE
    store.remove_eagle3_tensors(key=key, has_last_hidden_states=True)
    print(f"  REMOVE queued (deferred delete).")

    store.close()
    return hs_ok and ids_ok


# ─── Stage 3: Full vLLM pipeline ──────────────────────────────

def stage3_vllm(args):
    header("Stage 3: vLLM prefill → KV connector → Mooncake → verify")

    import torch
    from vllm import LLM, SamplingParams
    from verl.utils.mooncake.config import MooncakeConfig

    # Setup Mooncake env vars (vLLM workers read these)
    mc_config = MooncakeConfig.from_master_address(
        master_host=args.master_host,
        master_port=args.master_port,
        metadata_port=args.metadata_port,
        protocol="tcp",
    )
    mc_config.export_env()
    print(f"  Mooncake env exported: {mc_config.master_server_address}")

    # Resolve aux layer IDs
    if args.aux_layers:
        aux_layer_ids = [int(x) for x in args.aux_layers.split(",")]
    else:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        cfg = getattr(cfg, "text_config", cfg)
        n = cfg.num_hidden_layers
        aux_layer_ids = [1, n // 2 - 1, n - 4]
        if n - 1 not in aux_layer_ids:
            aux_layer_ids.append(n - 1)
    print(f"  aux_layer_ids: {aux_layer_ids}")

    # Create vLLM engine
    print(f"  Creating vLLM LLM (model={args.model_path}, tp={args.tp_size})...")
    t0 = time.time()
    engine = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": aux_layer_ids,
                }
            },
        },
        kv_transfer_config={
            "kv_connector": "MooncakeHiddenStatesConnector",
            "kv_connector_module_path": "verl.utils.mooncake.hidden_states_connector",
            "kv_role": "kv_producer",
        },
    )
    print(f"  Engine created in {time.time()-t0:.1f}s")

    # Prefill request
    print(f"\n  Sending prefill request: {args.prompt!r}")
    t0 = time.time()
    outputs = engine.generate([args.prompt], SamplingParams(max_tokens=1, temperature=0))
    dt = (time.time() - t0) * 1000
    print(f"  generate() completed in {dt:.1f} ms")

    ok = True
    for i, output in enumerate(outputs):
        kv_params = getattr(output, "kv_transfer_params", None)
        n_prompt = len(output.prompt_token_ids)
        generated = output.outputs[0].text if output.outputs else ""

        print(f"\n  Output {i}:")
        print(f"    prompt_tokens: {n_prompt}")
        print(f"    generated: {generated!r} (discarded — prefill only)")

        if kv_params is None:
            print(f"    kv_transfer_params: NONE — connector may not have fired")
            ok = False
            continue

        mooncake_key = kv_params.get("mooncake_key", "?")
        shapes = kv_params.get("tensor_shapes", {})
        dtypes = kv_params.get("tensor_dtypes", {})
        print(f"    mooncake_key: {mooncake_key}")
        print(f"    shapes: {shapes}")
        print(f"    dtypes: {dtypes}")

        # Verify Mooncake read
        if shapes:
            print(f"\n  Verifying Mooncake read...")
            from verl.utils.mooncake.eagle_store import EagleMooncakeStore
            store = EagleMooncakeStore(mc_config)
            store.setup()

            dtype_map = {"bfloat16": torch.bfloat16, "int64": torch.int64, "float32": torch.float32}
            torch_dtypes = {k: dtype_map.get(str(v), torch.bfloat16) for k, v in dtypes.items()}
            torch_shapes = {k: tuple(v) if isinstance(v, list) else v for k, v in shapes.items()}

            result = store.get(key=mooncake_key, shapes=torch_shapes, dtypes=torch_dtypes, device=torch.device("cpu"))
            print(f"    hidden_states:      {result.hidden_states.shape} ({result.hidden_states.dtype})")
            print(f"    input_ids:          {result.input_ids.shape} ({result.input_ids.dtype})")
            if result.last_hidden_states is not None:
                print(f"    last_hidden_states: {result.last_hidden_states.shape} (pre-norm)")
            print(f"    Mooncake read: OK")

            # Cleanup
            store.remove_eagle3_tensors(mooncake_key, has_last_hidden_states="last_hidden_states" in shapes)
            store.close()

    del engine
    return ok


# ─── Main ─────────────────────────────────────────────────────

def main():
    args = parse_args()
    stages = [1, 2, 3] if args.stage == "all" else [int(args.stage)]
    results = {}

    if 1 in stages:
        results[1] = stage1_imports()

    if 2 in stages:
        results[2] = stage2_mooncake(args)

    if 3 in stages:
        results[3] = stage3_vllm(args)

    header("Results")
    for s, ok in results.items():
        print(f"  Stage {s}: {'PASSED' if ok else 'FAILED'}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
