#!/usr/bin/env python3
"""
End-to-end HS collector + drafter trainer test using verl's single-controller
infrastructure.

Two separate RayWorkerGroups mimic the real architecture:
  - HSCollectorWorkerGroup (4 GPU actors): vLLM + KV connector → Mooncake put
  - DrafterTrainerWorkerGroup (4 CPU actors): Mooncake get → simulated train → remove

The driver (DrafterDataController) orchestrates both, matching RayPPOTrainer.fit():
  1. Driver creates DrafterDataController + both WorkerGroups
  2. Driver dispatches sequences to HS collectors (DP_COMPUTE_PROTO)
     → each worker runs vLLM generate → KV connector → Mooncake put
     → returns SampleMeta
  3. Driver collects metadata → controller Level 2
  4. Driver drains → DataProto → dispatches to trainer workers (DP_COMPUTE_PROTO)
     → each trainer fetches from Mooncake → simulated train_batch → remove
  5. Driver prints summary

Requires: 4 GPUs, mooncake_master binary, Qwen2-7B-Instruct cached.

Usage:
    python scripts/test_hs_collector_verl.py
    python scripts/test_hs_collector_verl.py --model Qwen/Qwen2-7B-Instruct --num-prompts 64
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("test_hs_verl")


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
        sys.exit(1)
    cmd = [binary, f"--port={port}", f"--http_metadata_server_port={http_port}",
           "--http_metadata_server_host=0.0.0.0", "--enable_http_metadata_server=true",
           f"--default_kv_lease_ttl={lease_ttl_ms}"]
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
        print(f"ERROR: mooncake_master exited with code {proc.returncode}\n  {stderr[:500]}")
        sys.exit(1)
    print(f"  mooncake_master running (PID {proc.pid})")
    return proc


# ─── Worker class (runs on each GPU as a Ray actor) ────────

@ray.remote
class HSCollectorWorker(Worker):
    """One vLLM HS collector per GPU — mirrors the HS collector in
    ActorRolloutRefDrafterWorker.  Owns vLLM engine + KV connector.
    Writes hidden states to Mooncake during prefill.
    """

    def __init__(self, model: str, max_seq_len: int, gpu_mem: float,
                 mooncake_host: str, mooncake_master_port: int, mooncake_metadata_port: int,
                 lease_ttl: float):
        super().__init__()
        self._model = model
        self._max_seq_len = max_seq_len
        self._gpu_mem = gpu_mem
        self._mooncake_host = mooncake_host
        self._mooncake_master_port = mooncake_master_port
        self._mooncake_metadata_port = mooncake_metadata_port
        self._lease_ttl = lease_ttl
        self._engine = None
        self._aux_layer_ids = None
        self._hidden_size = None

    def _make_mooncake_config(self):
        from verl.utils.mooncake.config import MooncakeConfig
        return MooncakeConfig(
            local_hostname=self._mooncake_host,
            metadata_server=f"http://{self._mooncake_host}:{self._mooncake_metadata_port}/metadata",
            master_server_address=f"{self._mooncake_host}:{self._mooncake_master_port}",
            global_segment_size=2 * 1024 * 1024 * 1024,
            local_buffer_size=512 * 1024 * 1024,
            protocol="tcp",
            max_seq_len=self._max_seq_len,
            hidden_dim=self._hidden_size or 4096,
            async_put_pool_size=4,
            kv_lease_ttl_s=self._lease_ttl,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_engine(self):
        """Initialize vLLM engine on this worker's GPU."""
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        logger.info(f"HSCollector {self.rank}: init (CUDA_VISIBLE_DEVICES={visible})")

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self._model, trust_remote_code=True)
        config = getattr(config, "text_config", config)
        n = config.num_hidden_layers
        self._hidden_size = config.hidden_size
        self._aux_layer_ids = [1, n // 2 - 1, n - 4, n - 1]

        self._make_mooncake_config().export_env()

        from vllm import LLM
        self._engine = LLM(
            model=self._model,
            tensor_parallel_size=1,
            gpu_memory_utilization=self._gpu_mem,
            trust_remote_code=True,
            distributed_executor_backend="mp",
            disable_custom_all_reduce=True,
            enable_prefix_caching=False,
            max_model_len=self._max_seq_len,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": list(self._aux_layer_ids),
                    },
                },
            },
            kv_transfer_config={
                "kv_connector": "MooncakeHiddenStatesConnector",
                "kv_connector_module_path": "verl.utils.mooncake.hidden_states_connector",
                "kv_role": "kv_producer",
            },
        )
        logger.info(f"HSCollector {self.rank}: vLLM ready "
                     f"(aux_layers={self._aux_layer_ids}, hidden={self._hidden_size})")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def collect_hidden_states(self, data: DataProto) -> DataProto:
        """vLLM prefill on this worker's shard → KV connector → Mooncake put → return metadata."""
        from vllm import SamplingParams

        input_ids_list = data.non_tensor_batch['input_ids']
        data_ids = data.non_tensor_batch['data_ids']

        prompts = [{"prompt_token_ids": ids.tolist() if hasattr(ids, 'tolist') else list(ids)}
                   for ids in input_ids_list]

        outputs = self._engine.generate(prompts, SamplingParams(max_tokens=1, temperature=0),
                                        use_tqdm=False)

        mooncake_keys, shapes_list, dtypes_list, seq_lens = [], [], [], []
        for i, output in enumerate(outputs):
            kv_params = getattr(output, "kv_transfer_params", None)
            if kv_params is None:
                logger.error(f"HSCollector {self.rank}: no kv_transfer_params for {data_ids[i]}")
                continue
            mooncake_keys.append(kv_params.get("mooncake_key", data_ids[i]))
            shapes_list.append(kv_params.get("tensor_shapes", {}))
            dtypes_list.append(kv_params.get("tensor_dtypes", {}))
            seq_lens.append(len(output.prompt_token_ids))

        n = len(mooncake_keys)
        return DataProto(
            non_tensor_batch={
                'mooncake_keys': np.array(mooncake_keys, dtype=object),
                'shapes': np.array(shapes_list, dtype=object),
                'dtypes': np.array(dtypes_list, dtype=object),
                'seq_lens': np.array(seq_lens, dtype=object),
            },
            batch=TensorDict({'_len': torch.ones(n)}, batch_size=[n]),
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def shutdown(self):
        if self._engine is not None:
            del self._engine
            self._engine = None
        logger.info(f"HSCollector {self.rank}: shutdown")


# ─── Drafter trainer worker (separate Ray actor group) ────

@ray.remote
class DrafterTrainerWorker(Worker):
    """Simulates FSDPDrafterEngine consumption in a separate process group.

    Each worker:
    - Owns a Mooncake reader store (like the real drafter trainer)
    - Fetches hidden states from Mooncake (simulates train_batch input)
    - Runs a dummy forward pass on the tensors (simulates EAGLE forward KL loss)
    - Cleans up Mooncake keys (simulates post-training removal)

    In the real pipeline this would be FSDPDrafterEngine + TrainingWorker,
    colocated on the same GPU but as a separate engine with its own FSDP group.
    Here we use CPU actors to prove the cross-process Mooncake data flow works.
    """

    def __init__(self, mooncake_host: str, mooncake_master_port: int,
                 mooncake_metadata_port: int, lease_ttl: float,
                 max_seq_len: int, hidden_dim: int):
        super().__init__()
        self._mooncake_host = mooncake_host
        self._mooncake_master_port = mooncake_master_port
        self._mooncake_metadata_port = mooncake_metadata_port
        self._lease_ttl = lease_ttl
        self._max_seq_len = max_seq_len
        self._hidden_dim = hidden_dim
        self._store = None
        self._train_steps = 0

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_store(self):
        """Initialize Mooncake reader store (like FSDPDrafterEngine setup)."""
        from verl.utils.mooncake.config import MooncakeConfig
        from verl.utils.mooncake.eagle_store import EagleMooncakeStore

        # Trainer only reads from Mooncake (get), never writes (put).
        # async_put_pool_size=0 skips host buffer pool allocation.
        # We also force CUDA_VISIBLE_DEVICES="" to prevent torch.cuda.is_available()
        # from trying to init CUDA in this CPU-only actor.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        mc_config = MooncakeConfig(
            local_hostname=self._mooncake_host,
            metadata_server=f"http://{self._mooncake_host}:{self._mooncake_metadata_port}/metadata",
            master_server_address=f"{self._mooncake_host}:{self._mooncake_master_port}",
            global_segment_size=512 * 1024 * 1024,
            local_buffer_size=512 * 1024 * 1024,
            protocol="tcp",
            max_seq_len=self._max_seq_len,
            hidden_dim=self._hidden_dim,
            async_put_pool_size=0,
            kv_lease_ttl_s=self._lease_ttl,
        )
        self._store = EagleMooncakeStore(mc_config)
        self._store.setup(device=None)
        logger.info(f"DrafterTrainer {self.rank}: Mooncake store ready")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_drafter(self, data: DataProto) -> DataProto:
        """Simulate update_drafter: Mooncake get → train_batch → remove.

        This mimics the real flow in ActorRolloutRefDrafterWorker.update_drafter():
        1. Fetch hidden_states, input_ids, last_hidden_states from Mooncake
        2. Run forward pass (here: simulated — compute norm to prove data is real)
        3. Compute loss (here: simulated Forward KL stub)
        4. Clean up Mooncake keys
        """
        mooncake_keys = data.non_tensor_batch['mooncake_keys']
        shapes_list = data.non_tensor_batch['shapes']
        dtypes_list = data.non_tensor_batch['dtypes']
        original_ids = data.non_tensor_batch.get('original_ids', [None] * len(mooncake_keys))

        n_ok = 0
        n_fail = 0
        total_tokens = 0
        total_loss = 0.0

        for i, (key, shapes, dtypes) in enumerate(zip(mooncake_keys, shapes_list, dtypes_list)):
            # ── Step 1: Mooncake get (same as real update_drafter) ──
            resolved_dtypes = {}
            for k, v in dtypes.items():
                resolved_dtypes[k] = getattr(torch, v) if isinstance(v, str) and hasattr(torch, v) else v

            result = self._store.get(key=key, shapes=shapes, dtypes=resolved_dtypes,
                                     device=torch.device("cpu"))

            hs = result.hidden_states          # [seq_len, num_aux_layers * hidden]
            ids = result.input_ids             # [seq_len]
            lhs = result.last_hidden_states    # [seq_len, hidden]

            # ── Step 2: Validate shapes ──
            hs_ok = hs.shape == tuple(shapes["hidden_states"])
            ids_ok = ids.shape == tuple(shapes["input_ids"])
            lhs_ok = lhs is not None
            nonzero = hs.abs().sum().item() > 0

            ids_match = True
            if original_ids[i] is not None:
                orig = original_ids[i]
                if hasattr(orig, 'tolist'):
                    orig = orig.tolist()
                ids_match = ids.tolist() == list(orig)

            # ── Step 3: Simulated train_batch (Forward KL loss stub) ──
            # In the real pipeline: compiled_forward_kl_loss(hs, lhs, ids, ...)
            # Here: compute a dummy "loss" to prove tensors are usable
            if lhs_ok and nonzero:
                # Simulate: RMSNorm → lm_head → Forward KL
                # Dummy: MSE between hs[:, :hidden] and lhs as proxy
                hidden_dim = lhs.shape[-1]
                hs_first_layer = hs[:, :hidden_dim].float()
                lhs_float = lhs.float()
                dummy_loss = torch.nn.functional.mse_loss(hs_first_layer, lhs_float).item()
                total_loss += dummy_loss
                total_tokens += hs.shape[0]
                self._train_steps += 1

            ok = hs_ok and ids_ok and lhs_ok and nonzero and ids_match
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                logger.error(f"DrafterTrainer {self.rank}: FAIL key={key} "
                             f"hs={hs_ok} ids={ids_ok} lhs={lhs_ok} nz={nonzero} idm={ids_match}")

            # ── Step 4: Cleanup (same as real update_drafter) ──
            self._store.remove_eagle3_tensors(key, has_last_hidden_states=lhs_ok)

        avg_loss = total_loss / max(n_ok, 1)
        logger.info(f"DrafterTrainer {self.rank}: {n_ok + n_fail} samples, "
                     f"{n_ok} OK, {n_fail} FAIL, "
                     f"tokens={total_tokens}, avg_loss={avg_loss:.4f}, "
                     f"total_steps={self._train_steps}")

        return DataProto(
            non_tensor_batch={
                'n_ok': np.array([n_ok], dtype=np.int64),
                'n_fail': np.array([n_fail], dtype=np.int64),
                'total_tokens': np.array([total_tokens], dtype=np.int64),
                'avg_loss': np.array([avg_loss], dtype=np.float64),
            },
            batch=TensorDict({'_len': torch.tensor([1])}, batch_size=[1]),
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def shutdown(self):
        if self._store is not None:
            self._store.close()
            self._store = None
        logger.info(f"DrafterTrainer {self.rank}: shutdown")


# ─── Prompts ──────────────────────────────────────────────

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


# ─── Driver (mirrors RayPPOTrainer) ──────────────────────

def run_test(args):
    from verl.trainer.drafter.controller import DrafterDataController, SampleMeta

    num_workers = args.num_workers

    # Discover hidden_dim for trainer worker group
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_config = getattr(model_config, "text_config", model_config)
    hidden_dim = model_config.hidden_size

    # ── Step 1: Create HS collector worker group (GPU actors) ──
    print(f"\n{'=' * 70}")
    print(f"[Step 1a] Creating HS Collector WorkerGroup: {num_workers} GPU actors")
    print(f"{'=' * 70}")

    gpu_pool = RayResourcePool(
        process_on_nodes=[num_workers],
        use_gpu=True,
        max_colocate_count=1,
    )

    hs_cls = RayClassWithInitArgs(
        cls=HSCollectorWorker,
        model=args.model,
        max_seq_len=args.max_seq_len,
        gpu_mem=args.gpu_mem,
        mooncake_host=args.master_host,
        mooncake_master_port=args.master_port,
        mooncake_metadata_port=args.metadata_port,
        lease_ttl=args.lease_ttl,
    )

    t0 = time.time()
    hs_wg = RayWorkerGroup(
        resource_pool=gpu_pool,
        ray_cls_with_init=hs_cls,
        name_prefix="hs_collector_",
    )
    print(f"  HS Collector group: {hs_wg.world_size} workers ({time.time() - t0:.1f}s)")

    # ── Step 1b: Create drafter trainer worker group (CPU actors) ──
    print(f"\n{'=' * 70}")
    print(f"[Step 1b] Creating Drafter Trainer WorkerGroup: {num_workers} CPU actors")
    print(f"{'=' * 70}")

    cpu_pool = RayResourcePool(
        process_on_nodes=[num_workers],
        use_gpu=False,
        max_colocate_count=1,
    )

    trainer_cls = RayClassWithInitArgs(
        cls=DrafterTrainerWorker,
        mooncake_host=args.master_host,
        mooncake_master_port=args.master_port,
        mooncake_metadata_port=args.metadata_port,
        lease_ttl=args.lease_ttl,
        max_seq_len=args.max_seq_len,
        hidden_dim=hidden_dim,
    )

    t0 = time.time()
    trainer_wg = RayWorkerGroup(
        resource_pool=cpu_pool,
        ray_cls_with_init=trainer_cls,
        name_prefix="drafter_trainer_",
    )
    print(f"  Drafter Trainer group: {trainer_wg.world_size} workers ({time.time() - t0:.1f}s)")

    # ── Step 2: Initialize both groups in parallel ──
    print(f"\n{'=' * 70}")
    print(f"[Step 2] Initializing engines + stores on all workers (parallel)")
    print(f"{'=' * 70}")

    t0 = time.time()
    # Fire both init calls — they run in parallel across groups
    hs_wg.init_engine()
    init_hs_sec = time.time() - t0

    t0 = time.time()
    trainer_wg.init_store()
    init_trainer_sec = time.time() - t0

    print(f"  HS Collectors initialized:   {init_hs_sec:.1f}s ({num_workers} vLLM engines)")
    print(f"  Drafter Trainers initialized: {init_trainer_sec:.1f}s ({num_workers} Mooncake stores)")

    # ── Step 3: Prepare prompts + controller ──
    print(f"\n{'=' * 70}")
    print(f"[Step 3] Preparing {args.num_prompts} prompts + DrafterDataController")
    print(f"{'=' * 70}")

    controller = DrafterDataController(dp_size=num_workers)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = [SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)] for i in range(args.num_prompts)]
    tokenized = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    data_ids = [f"rl_step0_seq{i:04d}" for i in range(args.num_prompts)]

    print(f"  {args.num_prompts} prompts tokenized (dp_size={num_workers})")

    # ── Step 4: HS collection (HS collector workers) ──
    print(f"\n{'=' * 70}")
    print(f"[Step 4] HS Collection: HS collectors → vLLM generate → Mooncake put")
    print(f"{'=' * 70}")

    n = args.num_prompts
    collect_proto = DataProto(
        non_tensor_batch={
            'input_ids': np.array(tokenized, dtype=object),
            'data_ids': np.array(data_ids, dtype=object),
        },
        batch=TensorDict({'_len': torch.ones(n)}, batch_size=[n]),
    )

    t0 = time.time()
    result_proto = hs_wg.collect_hidden_states(collect_proto)
    collect_sec = time.time() - t0

    mooncake_keys = result_proto.non_tensor_batch['mooncake_keys']
    shapes_list = result_proto.non_tensor_batch['shapes']
    dtypes_list = result_proto.non_tensor_batch['dtypes']
    seq_lens = result_proto.non_tensor_batch['seq_lens']

    print(f"  Collected {len(mooncake_keys)} samples in {collect_sec:.2f}s")
    for i in range(min(3, len(mooncake_keys))):
        hs_shape = shapes_list[i].get("hidden_states", "?")
        print(f"    [{i}] key={mooncake_keys[i]}  seq_len={seq_lens[i]}  hs={hs_shape}")
    if len(mooncake_keys) > 3:
        print(f"    ... ({len(mooncake_keys) - 3} more)")

    # Push to controller Level 2
    sample_metas = []
    for i in range(len(mooncake_keys)):
        sample_metas.append(SampleMeta(
            mooncake_key=mooncake_keys[i],
            shapes=shapes_list[i],
            dtypes=dtypes_list[i],
            seq_len=int(seq_lens[i]),
            n_tokens=int(seq_lens[i]),
        ))
    controller.push_samples(sample_metas)
    print(f"  Controller status: {controller.get_status()}")

    # ── Step 5: Drain → dispatch to drafter trainers (SEPARATE worker group) ──
    print(f"\n{'=' * 70}")
    print(f"[Step 5] Drain → dispatch update_drafter to trainer workers")
    print(f"{'=' * 70}")

    drain_proto = controller.drain_as_dataproto()
    if drain_proto is None:
        print("  ERROR: drain returned None!")
        return False

    drain_proto.non_tensor_batch['original_ids'] = np.array(tokenized[:len(mooncake_keys)], dtype=object)
    n_drain = len(drain_proto.non_tensor_batch['mooncake_keys'])
    drain_proto.batch = TensorDict({'_len': torch.ones(n_drain)}, batch_size=[n_drain])

    t0 = time.time()
    train_result = trainer_wg.update_drafter(drain_proto)
    train_sec = time.time() - t0

    total_ok = sum(train_result.non_tensor_batch['n_ok'])
    total_fail = sum(train_result.non_tensor_batch['n_fail'])
    total_tokens = sum(train_result.non_tensor_batch['total_tokens'])
    avg_losses = train_result.non_tensor_batch['avg_loss']
    overall_avg_loss = np.mean(avg_losses)

    print(f"  Trained in {train_sec:.2f}s: {total_ok} OK, {total_fail} FAIL")
    print(f"  Total tokens: {total_tokens}, avg dummy loss: {overall_avg_loss:.4f}")

    # ��─ Step 6: Shutdown both groups ──
    print(f"\n{'=' * 70}")
    print(f"[Step 6] Shutdown")
    print(f"{'=' * 70}")

    hs_wg.shutdown()
    trainer_wg.shutdown()

    wait = args.lease_ttl + 1.5
    print(f"  Waiting {wait:.1f}s for deferred deletes...")
    time.sleep(wait)

    # ���─ Summary ──
    all_ok = total_fail == 0
    print(f"\n{'=' * 70}")
    print(f"  verl Single-Controller HS Collector + Drafter Trainer Test")
    print(f"{'=' * 70}")
    print(f"  Model:              {args.model} (hidden={hidden_dim})")
    print(f"  HS Collectors:      {num_workers} GPU actors (TP=1 each)")
    print(f"  Drafter Trainers:   {num_workers} CPU actors (separate processes)")
    print(f"  Prompts:            {args.num_prompts}")
    print(f"")
    print(f"  vLLM init:          {init_hs_sec:.1f}s (parallel across {num_workers} GPUs)")
    print(f"  HS collection:      {collect_sec:.2f}s ({len(mooncake_keys)} samples)")
    print(f"  Drafter training:   {train_sec:.2f}s ({total_tokens} tokens, loss={overall_avg_loss:.4f})")
    print(f"  Passed:             {total_ok}/{total_ok + total_fail}")
    print(f"  Result:             {'ALL PASSED' if all_ok else 'FAILURES DETECTED'}")
    print(f"{'=' * 70}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="verl single-controller HS collector test")
    parser.add_argument("--model", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers (GPUs)")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.5)
    parser.add_argument("--master-host", default="localhost")
    parser.add_argument("--master-port", type=int, default=50051)
    parser.add_argument("--metadata-port", type=int, default=8090)
    parser.add_argument("--lease-ttl", type=float, default=5.0)
    parser.add_argument("--skip-master", action="store_true")
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"  verl Single-Controller HS Collector Test")
    print(f"  {args.num_workers} workers × TP=1 | {args.num_prompts} prompts | {args.model}")
    print(f"{'=' * 70}")

    master_proc = None
    if not args.skip_master:
        print("\n[Setup] Launching mooncake_master...")
        master_proc = launch_master(port=args.master_port, http_port=args.metadata_port,
                                    lease_ttl_ms=int(args.lease_ttl * 1000))

    print("\n[Setup] Initializing Ray...")
    ray.init(ignore_reinit_error=True)
    print(f"  Ray cluster: {ray.cluster_resources()}")

    ok = False
    try:
        ok = run_test(args)
    except KeyboardInterrupt:
        print("\n  Interrupted")
    except Exception:
        logger.exception("Test failed")
    finally:
        ray.shutdown()
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
