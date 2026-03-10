"""Target engine actor for SMC speculative decoding.

A plain Ray actor wrapping sglang Engine with tensor parallelism.
"""

import os
from typing import List, Optional

import ray

from verl.utils.device import get_visible_devices_keyword


@ray.remote
class TargetEngine:
    """Hosts the target sglang Engine with TP across multiple GPUs."""

    def __init__(
        self,
        model_path: str,
        tp_size: int,
        mem_fraction: float,
        cuda_visible_devices: str,
        base_gpu_id: int = 0,
        quantization: Optional[str] = None,
    ):
        # Set GPU visibility inside the actor process (verl pattern)
        visible_devices_keyword = get_visible_devices_keyword()
        os.environ[visible_devices_keyword] = cuda_visible_devices

        # Set NCCL env vars (from verl's _set_envs_and_config pattern)
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"

        from sglang.srt.entrypoints.engine import Engine

        engine_kwargs = {
            "model_path": model_path,
            "tp_size": tp_size,
            "mem_fraction_static": mem_fraction,
            "base_gpu_id": base_gpu_id,
            "log_level": "warning",
            "enable_cache_report": True,
        }
        if quantization:
            engine_kwargs["quantization"] = quantization

        print(f"[TargetEngine] Loading {model_path} with tp={tp_size}, "
              f"mem={mem_fraction}, gpus={cuda_visible_devices}")
        self.engine = Engine(**engine_kwargs)

    def score(
        self,
        input_ids: List[List[int]],
        logprob_start_lens: List[int],
    ) -> list:
        """Score sequences and return logprobs (no generation)."""
        result = self.engine.generate(
            input_ids=input_ids,
            sampling_params={"max_new_tokens": 0},
            return_logprob=True,
            logprob_start_len=logprob_start_lens,
        )
        if isinstance(result, list):
            return result
        return [result]

    def generate(
        self,
        input_ids: List[List[int]],
        sampling_params: dict,
        return_logprob: bool = True,
    ) -> list:
        """Generate tokens (used for fallback in sd_like mode)."""
        result = self.engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
        )
        if isinstance(result, list):
            return result
        return [result]

    def shutdown(self):
        if hasattr(self, "engine"):
            self.engine.shutdown()
