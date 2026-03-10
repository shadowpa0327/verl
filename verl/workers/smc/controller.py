"""SMC Controller — orchestrates particle-level DP across draft workers.

The controller owns all particle state and runs the SMC loop.  Each step it
shards particles across K draft workers (DP_COMPUTE), collects draft results,
calls the target engine once for all N particles, updates weights, and
resamples.
"""

import time
from typing import List, Optional, Tuple

import numpy as np
import ray

from .config import MultiGPUSMCConfig
from .utils import (
    effective_sample_size,
    normalize_weights,
    resample,
    sum_logprobs,
)


class SMCController:
    """Orchestrates the SMC decode loop with particle-level data parallelism."""

    def __init__(
        self,
        config: MultiGPUSMCConfig,
        draft_worker_group,
        target_engine_handle,
        tokenizer,
    ):
        self.config = config
        self.cfg = config.smc
        self.draft_wg = draft_worker_group
        self.target_handle = target_engine_handle
        self.tokenizer = tokenizer
        self.n_workers = draft_worker_group.world_size
        self._engines_ready = False

    def ensure_ready(self):
        """Verify that draft and target engines are initialized.

        Call this once after init_engine() on the worker group. It fires a
        lightweight probe to each draft worker (via DP_COMPUTE) and to the
        target engine to confirm they respond.
        """
        # Draft workers — send an empty shard; each worker returns immediately
        empty_shards = [[] for _ in range(self.n_workers)]
        probe_results = self.draft_wg.draft_generate(
            empty_shards,
            [1] * self.n_workers,
            [1.0] * self.n_workers,
        )
        for i, r in enumerate(probe_results):
            if not isinstance(r, dict) or "output_ids" not in r:
                raise RuntimeError(f"Draft worker {i} not ready: probe returned {r}")

        # Target engine — lightweight score call
        ray.get(self.target_handle.score.remote([[0]], [0]))

        self._engines_ready = True
        print("[SMCController] All engines ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode(
        self,
        prompt_ids: List[int],
        max_tokens: int = 128,
    ) -> Tuple[str, dict]:
        """Run the full SMC decode loop with particle-level DP.

        Args:
            prompt_ids: Pre-tokenized prompt token IDs.  Use
                ``tokenizer.apply_chat_template(..., tokenize=True)`` to
                produce these so that special tokens are encoded correctly.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            (generated_text, stats_dict)
        """
        if not self._engines_ready:
            raise RuntimeError(
                "SMCController.decode() called before ensure_ready(). "
                "Call controller.ensure_ready() after draft_wg.init_engine()."
            )

        prompt_len = len(prompt_ids)

        # Particle state — owned by the controller
        n = self.cfg.n_particles
        particle_ids: List[List[int]] = [prompt_ids[:] for _ in range(n)]
        particle_token_lens = [prompt_len] * n
        log_weights = np.zeros(n)
        finished = [False] * n

        stats = {
            "total_draft_tokens": 0,
            "total_target_scores": 0,
            "resample_count": 0,
            "steps": 0,
        }

        tokens_generated = 0
        start_time = time.time()

        while tokens_generated < max_tokens:
            if all(finished):
                break

            active_indices = [i for i, f in enumerate(finished) if not f]
            active_input_ids = [particle_ids[i] for i in active_indices]

            # -------------------------------------------------------
            # 1. Draft generate — shard active particles across workers
            # -------------------------------------------------------
            shards = self._shard_particles(active_input_ids)

            # DP_COMPUTE: each arg is a list[world_size]
            shard_results = self.draft_wg.draft_generate(
                shards,
                [self.cfg.gamma] * self.n_workers,
                [self.cfg.draft_temperature] * self.n_workers,
            )

            # Reassemble into global order (active indices only)
            draft_output_ids, draft_logprobs_list, draft_finished = (
                self._unshard_draft_results(shard_results, len(active_input_ids))
            )

            # Map back to full particle arrays
            full_output_ids: List[List[int]] = [[] for _ in range(n)]
            full_draft_logprobs = np.zeros(n)
            full_token_counts = [0] * n

            for j, i in enumerate(active_indices):
                full_output_ids[i] = draft_output_ids[j]
                full_draft_logprobs[i] = draft_logprobs_list[j]
                full_token_counts[i] = len(draft_output_ids[j])
                if draft_finished[j]:
                    finished[i] = True

            # Check if all active particles produced identical extensions (skip scoring)
            skip_target_scoring = False
            if active_indices:
                active_prefixes = [tuple(particle_ids[i]) for i in active_indices]
                active_extensions = [tuple(full_output_ids[i]) for i in active_indices]
                if (
                    len(set(active_prefixes)) == 1
                    and len(set(active_extensions)) == 1
                    and len(active_extensions[0]) > 0
                ):
                    skip_target_scoring = True

            if active_indices and not skip_target_scoring:
                active_prefix_lens = [particle_token_lens[i] for i in active_indices]
                logprob_start_lens = [max(0, pl - 1) for pl in active_prefix_lens]

            # Extend particles in-place
            for i in range(n):
                particle_ids[i].extend(full_output_ids[i])
            particle_token_lens = [len(ids) for ids in particle_ids]
            stats["total_draft_tokens"] += sum(full_token_counts)

            # -------------------------------------------------------
            # 2. Target scoring — single batched call for all active particles
            # -------------------------------------------------------
            target_logprobs = np.zeros(n)

            if active_indices and not skip_target_scoring:
                active_new_input_ids = [particle_ids[i] for i in active_indices]
                active_token_counts = [full_token_counts[i] for i in active_indices]

                target_outputs = ray.get(
                    self.target_handle.score.remote(
                        active_new_input_ids,
                        logprob_start_lens,
                    )
                )

                for j, i in enumerate(active_indices):
                    out = target_outputs[j]
                    input_lps = out.get("meta_info", {}).get("input_token_logprobs", [])
                    n_cont_tokens = active_token_counts[j]

                    if n_cont_tokens > 0 and len(input_lps) >= n_cont_tokens:
                        cont_lps = input_lps[-n_cont_tokens:]
                        target_logprobs[i] = sum_logprobs(cont_lps)
                        target_logprobs[i] /= self.cfg.target_lhts_temperature

                stats["total_target_scores"] += len(active_indices)
            elif skip_target_scoring:
                # All particles identical — target logprobs = draft logprobs
                target_logprobs = full_draft_logprobs.copy()

            # -------------------------------------------------------
            # 3. Update importance weights
            # -------------------------------------------------------
            log_importance = target_logprobs - full_draft_logprobs
            for i in active_indices:
                log_weights[i] += log_importance[i]

            # -------------------------------------------------------
            # 4. Resample if needed
            # -------------------------------------------------------
            weights = normalize_weights(log_weights)
            ess = effective_sample_size(weights)

            if ess < n * self.cfg.resample_threshold:
                old_finished = finished.copy()
                old_token_lens = particle_token_lens[:]
                particle_ids, log_weights, resampled_indices = resample(
                    particle_ids, weights, method=self.cfg.resample_method
                )
                finished = [old_finished[idx] for idx in resampled_indices]
                particle_token_lens = [old_token_lens[idx] for idx in resampled_indices]
                stats["resample_count"] += 1

            tokens_generated += self.cfg.gamma
            stats["steps"] += 1

        end_time = time.time()
        stats["elapsed_time"] = end_time - start_time
        stats["tokens_per_second"] = (
            tokens_generated / stats["elapsed_time"] if stats["elapsed_time"] > 0 else 0.0
        )

        # Return highest weight particle
        best_idx = int(np.argmax(log_weights))
        best_ids = particle_ids[best_idx]
        generated_ids = best_ids[prompt_len:]
        generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated, stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shard_particles(
        self, particle_ids_list: List[List[int]]
    ) -> List[List[List[int]]]:
        """Split a list of particle token-ID sequences into K shards.

        Returns a list of length ``self.n_workers``.  Each element is the
        subset of particles assigned to that worker.  If there are fewer
        particles than workers, some shards will be empty lists.
        """
        k = self.n_workers
        n = len(particle_ids_list)
        shards: List[List[List[int]]] = [[] for _ in range(k)]
        for idx, ids in enumerate(particle_ids_list):
            shards[idx % k].append(ids)
        return shards

    def _unshard_draft_results(
        self,
        shard_results: List[dict],
        n_active: int,
    ) -> Tuple[List[List[int]], List[float], List[bool]]:
        """Reassemble per-worker draft results into global active-particle order.

        The sharding in ``_shard_particles`` assigns particle ``idx`` to
        worker ``idx % K`` at position ``idx // K`` within that shard.
        This function reverses that mapping.
        """
        k = self.n_workers
        output_ids: List[Optional[List[int]]] = [None] * n_active
        draft_logprobs: List[float] = [0.0] * n_active
        finished_flags: List[bool] = [False] * n_active

        for worker_idx, result in enumerate(shard_results):
            worker_out_ids = result["output_ids"]
            worker_logprobs = result["draft_logprobs"]
            worker_finished = result["finished"]
            for local_idx in range(len(worker_out_ids)):
                global_idx = local_idx * k + worker_idx
                output_ids[global_idx] = worker_out_ids[local_idx]
                draft_logprobs[global_idx] = worker_logprobs[local_idx]
                finished_flags[global_idx] = worker_finished[local_idx]

        # Replace any None entries (shouldn't happen) with empty
        output_ids = [ids if ids is not None else [] for ids in output_ids]

        return output_ids, draft_logprobs, finished_flags
