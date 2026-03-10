"""SMC Draft Worker — a stateless draft engine wrapper.

Each worker owns a draft sglang Engine on one GPU. The controller shards
particles across workers and collects results.  Workers have no persistent
particle state and no target engine reference.
"""

import os
from typing import Dict, List, Optional

import ray
from transformers import AutoTokenizer

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register


@ray.remote
class SMCDraftWorker(Worker):
    """Stateless draft engine wrapper for particle-level DP."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.draft_engine = None
        self.tokenizer = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_engine(self):
        """Create the local draft sglang Engine on this worker's GPU."""
        from sglang.srt.entrypoints.engine import Engine

        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.target_model)

        draft_kwargs = {
            "model_path": self.config.draft_model,
            "mem_fraction_static": self.config.draft_mem_fraction,
            "log_level": "warning",
            "enable_cache_report": True,
        }
        if self.config.draft_quantization:
            draft_kwargs["quantization"] = self.config.draft_quantization

        print(
            f"[SMCDraftWorker rank={self.rank}] Loading draft model: "
            f"{self.config.draft_model} (mem={self.config.draft_mem_fraction})"
        )
        self.draft_engine = Engine(**draft_kwargs)
        print(f"[SMCDraftWorker rank={self.rank}] Draft engine ready.")

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def draft_generate(self, particle_ids_shard, gamma, temperature):
        """Generate draft tokens for a shard of particles.

        Args:
            particle_ids_shard: List[List[int]] — token IDs for this worker's particles.
            gamma: int — number of tokens to draft.
            temperature: float — sampling temperature.

        Returns:
            dict with:
                - output_ids: List[List[int]] — new tokens per particle
                - draft_logprobs: List[float] — sum of draft logprobs per particle
                - finished: List[bool] — whether each particle hit EOS / stop
        """
        from .utils import sum_logprobs

        if self.draft_engine is None:
            raise RuntimeError(
                f"[SMCDraftWorker rank={self.rank}] draft_generate called before init_engine()"
            )

        if not particle_ids_shard:
            return {"output_ids": [], "draft_logprobs": [], "finished": []}

        draft_result = self.draft_engine.generate(
            input_ids=particle_ids_shard,
            sampling_params={
                "max_new_tokens": gamma,
                "temperature": temperature,
            },
            return_logprob=True,
        )

        if not isinstance(draft_result, list):
            draft_result = [draft_result]

        output_ids = []
        draft_logprobs = []
        finished = []

        for local_idx, out in enumerate(draft_result):
            out_ids = list(out.get("output_ids", []) or [])
            meta = out.get("meta_info", {})
            finish_reason = meta.get("finish_reason", {})

            # sglang may echo back prompt-tail tokens in output_ids.
            # Use finish_reason to determine the actual generated count
            # and strip any leading input-echo tokens.
            n_generated = None
            if isinstance(finish_reason, dict):
                n_generated = finish_reason.get("length")
            if n_generated is not None and n_generated < len(out_ids):
                out_ids = out_ids[-n_generated:]

            output_ids.append(out_ids)

            # Logprobs — take only those matching the generated tokens
            output_lps = meta.get("output_token_logprobs", [])
            if len(output_lps) > len(out_ids):
                output_lps = output_lps[-len(out_ids):]
            draft_logprobs.append(sum_logprobs(output_lps))

            # Determine if this particle has finished generating.
            # Use sglang's finish_reason exclusively — do NOT scan out_ids
            # for eos_id, because sglang may include prompt-tail tokens in
            # output_ids which contain <|eot_id|> from the chat template.
            is_stop = finish_reason == "stop" or (
                isinstance(finish_reason, dict) and finish_reason.get("type") == "stop"
            )
            finished.append(is_stop)

        return {
            "output_ids": output_ids,
            "draft_logprobs": draft_logprobs,
            "finished": finished,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def shutdown(self):
        if self.draft_engine:
            self.draft_engine.shutdown()

    @property
    def rank(self):
        return self._rank
