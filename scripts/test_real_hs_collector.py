#!/usr/bin/env python3
"""
Real-world HS collector test with vLLM + Mooncake + DrafterDataController.

Launches a REAL vLLM engine with extract_hidden_states speculative method
and MooncakeHiddenStatesConnector. Sends real prompts through the model,
captures hidden states into Mooncake, then reads them back and verifies.

Exercises the full single-controller data lifecycle:
  1. DrafterDataController receives sequences (Level 1)
  2. vLLM prefill → KV Connector → Mooncake put (real model forward)
  3. Controller collects SampleMeta (Level 2)
  4. Controller drains → DataProto → simulated mesh dispatch (Level 3)
  5. Worker reads from Mooncake → verifies tensor shapes & values

Requires: 1 GPU, mooncake_master binary, a HuggingFace model.

Usage:
    # Default: Qwen2-7B-Instruct on GPU 0
    python scripts/test_real_hs_collector.py

    # Custom model / GPU
    python scripts/test_real_hs_collector.py --model Qwen/Qwen2.5-0.5B-Instruct --gpu 1

    # More prompts, wider DP simulation
    python scripts/test_real_hs_collector.py --num-prompts 16 --dp-size 4

    # Use existing mooncake_master
    python scripts/test_real_hs_collector.py --skip-master
"""

import argparse
import atexit
import logging
import os
import signal
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_real_hs")


# ─── Mooncake master lifecycle ─────────────────────────────

def find_mooncake_master_bin():
    if "MOONCAKE_BUILD_DIR" in os.environ:
        return os.path.join(os.environ["MOONCAKE_BUILD_DIR"], "mooncake-store/src/mooncake_master")
    found = shutil.which("mooncake_master")
    if found:
        return found
    return os.path.expanduser("~/build/mooncake-store/src/mooncake_master")


def launch_master(port=50051, http_port=8090, lease_ttl_ms=5000):
    binary = find_mooncake_master_bin()
    if not os.path.exists(binary):
        print(f"ERROR: mooncake_master not found at {binary}")
        print("  Install mooncake or set MOONCAKE_BUILD_DIR")
        sys.exit(1)

    cmd = [
        binary,
        f"--port={port}",
        f"--http_metadata_server_port={http_port}",
        "--http_metadata_server_host=0.0.0.0",
        "--enable_http_metadata_server=true",
        f"--default_kv_lease_ttl={lease_ttl_ms}",
    ]
    print(f"  Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _kill():
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.kill()

    atexit.register(_kill)
    time.sleep(2)

    if proc.poll() is not None:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        print(f"ERROR: mooncake_master exited with code {proc.returncode}")
        if stderr:
            print(f"  stderr: {stderr[:500]}")
        sys.exit(1)

    print(f"  mooncake_master running (PID {proc.pid})")
    return proc


# ─── Model / config helpers ───────────────────────────────

def get_aux_layer_ids(model_path):
    """Default Eagle3 aux layer selection: [1, mid-1, final-4, final]."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config = getattr(config, "text_config", config)
    n = config.num_hidden_layers
    aux = [1, n // 2 - 1, n - 4]
    # Append final layer for last_hidden_states capture
    final = n - 1
    if final not in aux:
        aux.append(final)
    return aux, config.hidden_size, n


def make_mooncake_config(host, master_port, metadata_port, max_seq_len, hidden_dim, lease_ttl):
    from verl.utils.mooncake.config import MooncakeConfig
    return MooncakeConfig(
        local_hostname=host,
        metadata_server=f"http://{host}:{metadata_port}/metadata",
        master_server_address=f"{host}:{master_port}",
        global_segment_size=2 * 1024 * 1024 * 1024,   # 2 GB
        local_buffer_size=512 * 1024 * 1024,            # 512 MB
        protocol="tcp",
        max_seq_len=max_seq_len,
        hidden_dim=hidden_dim,
        async_put_pool_size=4,
        kv_lease_ttl_s=lease_ttl,
    )


SAMPLE_PROMPTS = [
    "Explain the concept of gradient descent in machine learning.",
    "What is the capital of France? Answer in one word.",
    "Write a Python function that computes the Fibonacci sequence.",
    "Summarize the theory of relativity in three sentences.",
    "How does a transformer neural network process text input?",
    "What are the main differences between TCP and UDP?",
    "Explain quantum entanglement to a five year old.",
    "Write a haiku about artificial intelligence.",
    "What is the time complexity of quicksort?",
    "Describe the water cycle in simple terms.",
    "How do GPUs accelerate deep learning training?",
    "What is RLHF and why is it used for language models?",
    "Explain the difference between L1 and L2 regularization.",
    "What is speculative decoding and how does it work?",
    "Describe the architecture of a typical recommendation system.",
    "What are the benefits of using FSDP for distributed training?",
]


def fmt_bytes(n):
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


# ─── Main test ─────────────────────────────────────────────

def run_test(args):
    from verl.trainer.drafter.controller import DrafterDataController, SequenceMeta, SampleMeta
    from verl.utils.mooncake.eagle_store import EagleMooncakeStore

    # ── Phase 0: Discover model config ──
    print(f"\n{'=' * 70}")
    print(f"[Phase 0] Model discovery: {args.model}")
    print(f"{'=' * 70}")

    aux_layer_ids, hidden_size, num_layers = get_aux_layer_ids(args.model)
    num_training_layers = len(aux_layer_ids) - 1  # last is for last_hidden_states
    training_hidden_size = num_training_layers * hidden_size

    print(f"  Model:         {args.model}")
    print(f"  Layers:        {num_layers}")
    print(f"  Hidden size:   {hidden_size}")
    print(f"  Aux layers:    {aux_layer_ids}")
    print(f"  Training HS:   {num_training_layers} layers × {hidden_size} = {training_hidden_size}")
    print(f"  GPU:           cuda:{args.gpu} (TP={args.tp_size})")

    # ── Phase 1: Setup Mooncake + controller ──
    print(f"\n{'=' * 70}")
    print(f"[Phase 1] Setting up Mooncake stores + DrafterDataController")
    print(f"{'=' * 70}")

    mc_config = make_mooncake_config(
        host=args.master_host,
        master_port=args.master_port,
        metadata_port=args.metadata_port,
        max_seq_len=args.max_seq_len,
        hidden_dim=hidden_size,
        lease_ttl=args.lease_ttl,
    )

    # Export env so the KV connector (inside vLLM worker process) can find Mooncake
    mc_config.export_env()
    print("  Mooncake env vars exported")

    controller = DrafterDataController(dp_size=args.dp_size)
    print(f"  Controller ready (dp_size={args.dp_size})")

    # Reader store for the "training side"
    reader_store = EagleMooncakeStore(mc_config)
    reader_store.setup()
    print("  Reader store (training side): connected")

    # ── Phase 2: Launch vLLM engine with extract_hidden_states ──
    print(f"\n{'=' * 70}")
    print(f"[Phase 2] Launching vLLM with extract_hidden_states + KV connector")
    print(f"{'=' * 70}")

    if args.tp_size > 1:
        # Multi-GPU: expose all needed GPUs
        gpu_ids = ",".join(str(args.gpu + i) for i in range(args.tp_size))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from vllm import LLM, SamplingParams

    engine_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tp_size,
        "gpu_memory_utilization": args.gpu_mem,
        "trust_remote_code": True,
        "distributed_executor_backend": "mp",
        "disable_custom_all_reduce": True,
        "enable_prefix_caching": False,
        "max_model_len": args.max_seq_len,
        "speculative_config": {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": list(aux_layer_ids),
                },
            },
        },
        "kv_transfer_config": {
            "kv_connector": "MooncakeHiddenStatesConnector",
            "kv_connector_module_path": "verl.utils.mooncake.hidden_states_connector",
            "kv_role": "kv_producer",
        },
    }

    print(f"  Engine config:")
    print(f"    model:              {args.model}")
    print(f"    tensor_parallel:    {args.tp_size}")
    print(f"    max_model_len:      {args.max_seq_len}")
    print(f"    gpu_mem_utilization: {args.gpu_mem}")
    print(f"    aux_layer_ids:      {aux_layer_ids}")
    print(f"    kv_connector:       MooncakeHiddenStatesConnector")

    t0 = time.time()
    engine = LLM(**engine_kwargs)
    init_sec = time.time() - t0
    print(f"  vLLM engine initialized in {init_sec:.1f}s")

    # ── Phase 3: Tokenize prompts → push to controller (Level 1) ──
    print(f"\n{'=' * 70}")
    print(f"[Phase 3] Tokenizing {args.num_prompts} prompts → controller Level 1")
    print(f"{'=' * 70}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Cycle prompts if num_prompts exceeds the list
    prompts = [SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)] for i in range(args.num_prompts)]
    tokenized = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]

    sequences = []
    for i, token_ids in enumerate(tokenized):
        seq_len = len(token_ids)
        seq_meta = SequenceMeta(
            input_ids=np.array(token_ids, dtype=np.int64),
            attention_mask=np.ones(seq_len, dtype=np.int64),
            prompt_len=seq_len,
            response_len=0,
        )
        sequences.append(seq_meta)
        if i < 3:
            print(f"  [{i}] {seq_len:4d} tokens: \"{prompts[i][:60]}...\"")
    if args.num_prompts > 3:
        print(f"  ... ({args.num_prompts - 3} more)")

    controller.push_raw_prompts(sequences)
    print(f"  Pushed {len(sequences)} sequences → Level 1")

    # ── Phase 4: HS Collection — vLLM generate → Mooncake ──
    print(f"\n{'=' * 70}")
    print(f"[Phase 4] HS Collection: vLLM generate → KV connector → Mooncake")
    print(f"{'=' * 70}")

    raw_prompts = controller.pull_raw_prompts()

    # Build vLLM prompts from token IDs
    vllm_prompts = []
    data_ids = []
    for i, seq_meta in enumerate(raw_prompts):
        vllm_prompts.append({"prompt_token_ids": seq_meta.input_ids.tolist()})
        data_ids.append(f"rl_step0_seq{i:04d}")

    sampling_params = SamplingParams(max_tokens=1, temperature=0)

    print(f"  Sending {len(vllm_prompts)} prompts to vLLM (prefill-only, max_tokens=1)...")
    t0 = time.time()
    outputs = engine.generate(vllm_prompts, sampling_params, use_tqdm=True)
    gen_sec = time.time() - t0
    print(f"  vLLM generate completed in {gen_sec:.2f}s")

    # Collect metadata from kv_transfer_params
    sample_metas = []
    for i, output in enumerate(outputs):
        kv_params = getattr(output, "kv_transfer_params", None)
        if kv_params is None:
            print(f"  WARNING: output[{i}] has no kv_transfer_params — connector may not have fired")
            continue

        mooncake_key = kv_params.get("mooncake_key", data_ids[i])
        shapes = kv_params.get("tensor_shapes", {})
        dtypes = kv_params.get("tensor_dtypes", {})
        seq_len = len(output.prompt_token_ids)

        sample = SampleMeta(
            mooncake_key=mooncake_key,
            shapes=shapes,
            dtypes=dtypes,
            seq_len=seq_len,
            n_tokens=seq_len,
        )
        sample_metas.append(sample)

        if i < 3 or i == len(outputs) - 1:
            hs_shape = shapes.get("hidden_states", "?")
            lhs_shape = shapes.get("last_hidden_states", "?")
            print(f"  [{i}] key={mooncake_key}  seq_len={seq_len}  "
                  f"hs={hs_shape}  lhs={lhs_shape}")
            if i == 2 and len(outputs) > 4:
                print(f"  ... ({len(outputs) - 4} more)")

    if not sample_metas:
        print("\n  FATAL: No samples collected! KV connector did not store anything.")
        print("  Check that MOONCAKE_MASTER_SERVER env var is set before LLM init.")
        return False

    controller.push_samples(sample_metas)
    status = controller.get_status()
    print(f"\n  Pushed {len(sample_metas)} samples → Level 2")
    print(f"  Controller: {status}")

    # ── Phase 5: Drain → DataProto → mesh dispatch ──
    print(f"\n{'=' * 70}")
    print(f"[Phase 5] Drain → DataProto → mesh dispatch (dp_size={args.dp_size})")
    print(f"{'=' * 70}")

    proto = controller.drain_as_dataproto()
    if proto is None:
        print("  ERROR: drain_as_dataproto returned None!")
        return False

    n_samples = len(proto.non_tensor_batch['mooncake_keys'])
    rank_indices = np.array_split(np.arange(n_samples), args.dp_size)

    for rank in range(args.dp_size):
        idx = rank_indices[rank]
        print(f"  Rank {rank}: {len(idx)} samples (indices {idx[0]}..{idx[-1]})")

    # ── Phase 6: Worker-side Mooncake fetch + verify ──
    print(f"\n{'=' * 70}")
    print(f"[Phase 6] Worker-side: Mooncake get → verify shapes + values")
    print(f"{'=' * 70}")

    all_ok = True
    total_get_ms = 0.0
    total_bytes = 0

    for rank in range(args.dp_size):
        idx = rank_indices[rank]
        rank_keys = proto.non_tensor_batch['mooncake_keys'][idx]
        rank_shapes = proto.non_tensor_batch['shapes'][idx]
        rank_dtypes = proto.non_tensor_batch['dtypes'][idx]

        print(f"\n  ── Rank {rank} ({len(rank_keys)} samples) ──")

        for j, (key, shapes, dtypes) in enumerate(zip(rank_keys, rank_shapes, rank_dtypes)):
            # Convert string dtypes to torch dtypes for get()
            resolved_dtypes = {}
            for k, v in dtypes.items():
                if isinstance(v, str):
                    resolved_dtypes[k] = getattr(torch, v) if hasattr(torch, v) else v
                else:
                    resolved_dtypes[k] = v

            t0 = time.time()
            result = reader_store.get(
                key=key,
                shapes=shapes,
                dtypes=resolved_dtypes,
                device=torch.device("cpu"),
            )
            get_ms = (time.time() - t0) * 1000
            total_get_ms += get_ms

            # Verify shapes match metadata
            hs = result.hidden_states
            ids = result.input_ids
            lhs = result.last_hidden_states

            expected_hs = tuple(shapes["hidden_states"])
            expected_ids = tuple(shapes["input_ids"])

            shape_ok = (hs.shape == expected_hs and ids.shape == expected_ids)

            # Verify lhs if present
            lhs_ok = True
            if "last_hidden_states" in shapes:
                expected_lhs = tuple(shapes["last_hidden_states"])
                lhs_ok = lhs is not None and lhs.shape == expected_lhs

            # Verify hidden states are not all zeros (real model should produce non-trivial output)
            nonzero_ok = hs.abs().sum().item() > 0

            # Verify input_ids match original prompt tokens
            original_tokens = tokenized[idx[j]]
            ids_match = ids.tolist() == original_tokens

            ok = shape_ok and lhs_ok and nonzero_ok and ids_match
            if not ok:
                all_ok = False

            hs_bytes = hs.numel() * hs.element_size()
            total_bytes += hs_bytes
            if lhs is not None:
                total_bytes += lhs.numel() * lhs.element_size()
            total_bytes += ids.numel() * ids.element_size()

            status_str = "OK" if ok else "FAIL"
            details = []
            if not shape_ok:
                details.append(f"shape_mismatch(hs={hs.shape}!={expected_hs})")
            if not lhs_ok:
                details.append("lhs_missing_or_wrong_shape")
            if not nonzero_ok:
                details.append("all_zeros!")
            if not ids_match:
                details.append(f"ids_mismatch(got {ids.shape[0]} vs {len(original_tokens)})")

            if j < 3 or j == len(rank_keys) - 1 or not ok:
                extra = f"  [{', '.join(details)}]" if details else ""
                print(f"    [{j}] key={key}  {fmt_bytes(hs_bytes)} in {get_ms:.1f}ms  "
                      f"hs={list(hs.shape)} lhs={'yes' if lhs is not None else 'NO'}  "
                      f"ids_match={'yes' if ids_match else 'NO'}  {status_str}{extra}")
                if j == 2 and len(rank_keys) > 4:
                    print(f"    ... ({len(rank_keys) - 4} more)")

            # Cleanup
            reader_store.remove_eagle3_tensors(key, has_last_hidden_states=lhs is not None)

    avg_get = total_get_ms / n_samples if n_samples else 0

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  Real-World HS Collector Test Summary")
    print(f"{'=' * 70}")
    print(f"  Model:           {args.model}")
    print(f"  Hidden size:     {hidden_size} ({num_layers} layers)")
    print(f"  Aux layers:      {aux_layer_ids}")
    print(f"  Prompts:         {args.num_prompts}")
    print(f"  DP size:         {args.dp_size}")
    print(f"")
    print(f"  vLLM init:       {init_sec:.1f}s")
    print(f"  vLLM generate:   {gen_sec:.2f}s ({len(outputs)} prefills)")
    print(f"  Mooncake get:    {total_get_ms:.1f}ms total ({avg_get:.1f}ms avg)")
    print(f"  Total HS data:   {fmt_bytes(total_bytes)}")
    print(f"")
    print(f"  Samples stored:  {len(sample_metas)}")
    print(f"  Samples fetched: {n_samples}")
    print(f"  Shapes match:    {'yes' if all_ok else 'NO'}")
    print(f"  Input IDs match: {'yes' if all_ok else 'NO'}")
    print(f"  Non-zero HS:     {'yes' if all_ok else 'NO'}")
    print(f"")
    print(f"  Result: {'ALL PASSED' if all_ok else 'FAILURES DETECTED'}")
    print(f"{'=' * 70}")

    # Cleanup
    wait = args.lease_ttl + 1.5
    print(f"\n  Waiting {wait:.1f}s for deferred deletes...")
    time.sleep(wait)

    reader_store.close()

    # Shutdown vLLM
    del engine

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Real-world HS collector test (vLLM + Mooncake + controller)"
    )
    parser.add_argument("--model", default="Qwen/Qwen2-7B-Instruct",
                        help="HuggingFace model path")
    parser.add_argument("--gpu", type=int, default=0, help="Base GPU device ID")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--gpu-mem", type=float, default=0.5,
                        help="vLLM gpu_memory_utilization")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Max sequence length for vLLM")
    parser.add_argument("--num-prompts", type=int, default=8,
                        help="Number of prompts to send")
    parser.add_argument("--dp-size", type=int, default=2,
                        help="Simulated DP size for mesh dispatch")
    parser.add_argument("--master-host", default="localhost")
    parser.add_argument("--master-port", type=int, default=50051)
    parser.add_argument("--metadata-port", type=int, default=8090)
    parser.add_argument("--lease-ttl", type=float, default=5.0,
                        help="Mooncake lease TTL in seconds")
    parser.add_argument("--skip-master", action="store_true",
                        help="Don't launch mooncake_master (use existing)")
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"  Real-World HS Collector Test")
    print(f"  vLLM {args.model} + MooncakeHiddenStatesConnector")
    print(f"  GPU cuda:{args.gpu} | TP={args.tp_size} | dp_size={args.dp_size} | {args.num_prompts} prompts")
    print(f"{'=' * 70}")

    master_proc = None
    if not args.skip_master:
        print("\n[Setup] Launching mooncake_master...")
        lease_ttl_ms = int(args.lease_ttl * 1000)
        master_proc = launch_master(
            port=args.master_port,
            http_port=args.metadata_port,
            lease_ttl_ms=lease_ttl_ms,
        )
    else:
        print("\n[Setup] Skipping master launch (--skip-master)")

    ok = False
    try:
        ok = run_test(args)
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
    except Exception:
        logger.exception("Test failed with exception")
    finally:
        if master_proc and master_proc.poll() is None:
            print("\n[Cleanup] Stopping mooncake_master...")
            time.sleep(0.5)
            master_proc.terminate()
            try:
                master_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                master_proc.kill()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
