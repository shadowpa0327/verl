"""Demo script for multi-GPU SMC speculative decoding with particle-level DP.

The SMC loop runs in the controller (driver process).  Draft workers are
stateless engines distributed across GPUs.  The target engine uses TP.

Usage:
    python scripts/run_smc_sd.py --prompt "What is 15 + 27?"
    python scripts/run_smc_sd.py --n-gpus 4 --n-particles 16 --prompt "Explain quantum computing."
"""

import argparse

import ray

from transformers import AutoTokenizer

from verl.single_controller.ray.base import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.utils.device import get_device_name, get_visible_devices_keyword
from verl.workers.smc.config import MultiGPUSMCConfig, SMCConfig
from verl.workers.smc.controller import SMCController
from verl.workers.smc.smc_worker import SMCDraftWorker
from verl.workers.smc.target_engine import TargetEngine


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU SMC Speculative Decoding (Particle-Level DP)")
    parser.add_argument("--prompt", type=str, default="Explain how to perform the integral with 10 experiments?")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--draft-model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--target-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument("--target-tp-size", type=int, default=4)
    parser.add_argument("--n-particles", type=int, default=16)
    parser.add_argument("--gamma", type=int, default=8)
    parser.add_argument("--draft-mem", type=float, default=0.25)
    parser.add_argument("--target-mem", type=float, default=0.55)
    args = parser.parse_args()

    ray.init()

    config = MultiGPUSMCConfig(
        smc=SMCConfig(
            n_particles=args.n_particles,
            gamma=args.gamma,
        ),
        draft_model=args.draft_model,
        target_model=args.target_model,
        n_gpus=args.n_gpus,
        target_tp_size=args.target_tp_size,
        draft_mem_fraction=args.draft_mem,
        target_mem_fraction=args.target_mem,
    )

    # 1. Create the shared target engine (TP across all GPUs)
    visible_devices_keyword = get_visible_devices_keyword()
    cuda_visible_devices = ",".join(str(i) for i in range(args.n_gpus))

    print(f"Creating TargetEngine with tp={args.target_tp_size}, gpus={cuda_visible_devices}")
    target_handle = TargetEngine.options(
        num_gpus=0,
        runtime_env={"env_vars": {f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}": "1"}},
    ).remote(
        model_path=args.target_model,
        tp_size=args.target_tp_size,
        mem_fraction=args.target_mem,
        cuda_visible_devices=cuda_visible_devices,
        quantization=None,
    )
    # Wait for target engine to be ready
    ray.get(target_handle.score.remote([[0]], [0]))
    print("TargetEngine ready.")

    # 2. Create the SMCDraftWorker WorkerGroup (DP across GPUs)
    draft_cls = RayClassWithInitArgs(
        cls=SMCDraftWorker,
        config=config,
    )
    resource_pool = RayResourcePool(
        process_on_nodes=[args.n_gpus],
        max_colocate_count=2,  # share GPUs with target engine
    )
    draft_wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=draft_cls,
        device_name=get_device_name(),
    )
    print(f"DraftWorkerGroup created with {draft_wg.world_size} workers.")

    # 3. Initialize draft engines on all workers
    draft_wg.init_engine()
    print("Draft engines initialized on all workers.")

    # 4. Create the controller
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    controller = SMCController(
        config=config,
        draft_worker_group=draft_wg,
        target_engine_handle=target_handle,
        tokenizer=tokenizer,
    )

    # 5. Verify all engines are ready
    controller.ensure_ready()

    # 6. Format prompt and decode
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )

    print(f"\nDecoding prompt: {args.prompt}")
    print(f"Config: n_particles={args.n_particles}, gamma={args.gamma}, "
          f"n_workers={draft_wg.world_size}")
    print(f"Prompt tokens: {len(prompt_ids)}")
    print("-" * 60)

    text, stats = controller.decode(prompt_ids, max_tokens=args.max_tokens)

    print(f"\nOutput: {text}")
    print(f"Tokens/sec: {stats['tokens_per_second']:.1f}")
    print(f"Steps: {stats['steps']}, Resamples: {stats['resample_count']}")
    print(f"Draft tokens: {stats['total_draft_tokens']}, "
          f"Target scores: {stats['total_target_scores']}")

    # Cleanup
    draft_wg.shutdown()
    ray.get(target_handle.shutdown.remote())
    ray.shutdown()


if __name__ == "__main__":
    main()
