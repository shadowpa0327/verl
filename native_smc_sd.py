"""
Native SMC Speculative Decoding using sglang's Engine API.

Eliminates HTTP overhead and enables RadixAttention for KV-cache reuse.

Usage:
    python smc_native.py --prompt "What is 15 + 27?"
    python smc_native.py --eval --num-samples 20
"""

import argparse
import time
import numpy as np
import nvtx
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from transformers import AutoTokenizer


@dataclass
class SMCConfig:
    """Configuration for SMC decoding."""
    mode: str = "smc"  # smc | sd_like
    n_particles: int = 4
    gamma: int = 8
    draft_temperature: float = 0.7
    target_lhts_temperature: float = 1.0
    resample_threshold: float = 0.5
    rejuvenate: bool = False
    rejuvenation_k: int = 16
    rejuvenation_frac: float = 0.5
    rejuvenation_logp_threshold: float = -3.0
    rejuvenation_logp_ema_alpha: float = 0.3
    rejuvenation_trigger_steps: int = 2
    resample_method: str = "systematic"  # systematic | multinomial


class NativeSMCDecoder:
    """
    SMC decoder using sglang's native Engine API.

    Benefits over HTTP version:
    - No HTTP/JSON overhead
    - RadixAttention automatically caches shared prefixes
    - Direct in-process communication
    """

    def __init__(
        self,
        draft_model: str = "meta-llama/Llama-3.2-1B-Instruct",
        target_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        config: Optional[SMCConfig] = None,
        draft_mem_fraction: float = 0.25,
        target_mem_fraction: float = 0.55,
        draft_quantization: Optional[str] = None,
        target_quantization: Optional[str] = None,
    ):
        from sglang.srt.entrypoints.engine import Engine

        self.config = config or SMCConfig()
        self.target_model_name = target_model

        print(f"Loading tokenizer from {target_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(target_model)

        draft_kwargs = {
            "model_path": draft_model,
            "mem_fraction_static": draft_mem_fraction,
            "log_level": "warning",
            "enable_cache_report": True,
        }
        if draft_quantization:
            draft_kwargs["quantization"] = draft_quantization

        print(
            f"Loading draft model: {draft_model} (mem={draft_mem_fraction})"
            + (f", quant={draft_quantization}" if draft_quantization else "")
            + "..."
        )
        self.draft_engine = Engine(**draft_kwargs)

        target_kwargs = {
            "model_path": target_model,
            "mem_fraction_static": target_mem_fraction,
            "log_level": "warning",
            "enable_cache_report": True,
        }
        if target_quantization:
            target_kwargs["quantization"] = target_quantization

        print(
            f"Loading target model: {target_model} (mem={target_mem_fraction})"
            + (f", quant={target_quantization}" if target_quantization else "")
            + "..."
        )
        self.target_engine = Engine(**target_kwargs)

    def format_user_prompt(self, user_text: str) -> str:
        """
        Format a user prompt for chat models.

        Uses tokenizer chat template when available; otherwise falls back to a
        Vicuna-style USER/ASSISTANT format that works for older chat checkpoints.
        """
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback for tokenizers without a built-in chat template (e.g., Vicuna).
        return (
            "A chat between a curious user and an assistant. "
            "The assistant gives helpful, detailed, and polite answers.\n"
            f"USER: {user_text}\n"
            "ASSISTANT:"
        )

        print(f"SMC Config: mode={self.config.mode}, N={self.config.n_particles}, γ={self.config.gamma}")

    def decode(
        self,
        prompt: str,
        max_tokens: int = 128,
        verbose: bool = False,
        return_particles: bool = False,
        viz: bool = False,
    ) -> Tuple[str, dict]:
        """Generate text using SMC with native Engine API."""
        if self.config.mode == "sd_like":
            return self._decode_sd_like(
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=verbose,
                return_particles=return_particles,
            )

        cfg = self.config
        viz_steps = [] if viz else None

        # Tokenize prompt ONCE and keep particles as token IDs (no string concat / re-tokenization)
        prompt_ids: List[int] = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # Particle state = token IDs
        particle_ids: List[List[int]] = [prompt_ids[:] for _ in range(cfg.n_particles)]
        particle_token_lens = [prompt_len] * cfg.n_particles
        log_weights = np.zeros(cfg.n_particles)
        finished = [False] * cfg.n_particles

        stats = {
            "total_draft_tokens": 0,
            "total_target_scores": 0,
            "resample_count": 0,
            "steps": 0,
            "rejuvenation_count": 0,
            "rejuvenation_attempts": 0,
            "rejuvenation_accepts": 0,
            # Cache diagnostics (sglang radix/prefix cache)
            "draft_cached_tokens": 0,
            "draft_total_tokens": 0,
            "target_cached_tokens": 0,
            "target_total_tokens": 0,
            # Profiling: timing breakdown
            "time_draft_gen": 0.0,
            "time_target_score": 0.0,
            "time_weight_update": 0.0,
            "time_resample": 0.0,
            "time_rejuvenation": 0.0,
            # Optimization stats
            "skipped_target_calls": 0,
        }

        tokens_generated = 0
        start_time = time.time()
        first_step_end = None
        first_step_tokens = 0

        logp_ema = None
        low_logp_steps = 0

        while tokens_generated < max_tokens:
            if all(finished):
                if verbose:
                    print(f"All particles finished at step {stats['steps']}")
                break

            nvtx.push_range(f"smc_step_{stats['steps']}", color="orange")

            # 1. Generate from draft model - only active particles
            active_indices = [i for i, f in enumerate(finished) if not f]
            active_input_ids = [particle_ids[i] for i in active_indices]

            full_draft_logprobs = np.zeros(cfg.n_particles)
            full_token_counts = [0] * cfg.n_particles
            full_output_ids: List[List[int]] = [[] for _ in range(cfg.n_particles)]
            draft_lps_per_token: List[List[float]] = [[] for _ in range(cfg.n_particles)]

            if active_input_ids:
                # Batched call to draft engine
                t0 = time.perf_counter()
                with nvtx.annotate("draft_generate", color="blue"):
                    draft_result = self.draft_engine.generate(
                        input_ids=active_input_ids,
                        sampling_params={
                            "max_new_tokens": cfg.gamma,
                            "temperature": cfg.draft_temperature,
                        },
                        return_logprob=True,
                    )
                stats["time_draft_gen"] += time.perf_counter() - t0

                # Handle response (could be list or single dict)
                if isinstance(draft_result, list):
                    draft_outputs = draft_result
                else:
                    draft_outputs = [draft_result]

                for idx, active_idx in enumerate(active_indices):
                    out = draft_outputs[idx]
                    out_ids = out.get("output_ids", []) or []
                    full_output_ids[active_idx] = list(out_ids)

                    # Sum output token logprobs (generated tokens)
                    output_lps = out.get("meta_info", {}).get("output_token_logprobs", [])
                    full_draft_logprobs[active_idx] = self._sum_logprobs(output_lps)

                    if viz:
                        draft_lps_per_token[active_idx] = self._extract_logprobs(output_lps, len(out_ids))

                    # Track token count from output IDs (no tokenization!)
                    n_output_tokens = len(out_ids)
                    full_token_counts[active_idx] = n_output_tokens

                    # Cache stats (may be 0 / absent depending on backend/config)
                    meta_info = out.get("meta_info", {}) or {}
                    cached = meta_info.get("cached_tokens", 0)
                    prompt_tokens = meta_info.get("prompt_tokens", len(active_input_ids[idx]))
                    if isinstance(cached, (int, float)):
                        stats["draft_cached_tokens"] += int(cached)
                    if isinstance(prompt_tokens, (int, float)):
                        stats["draft_total_tokens"] += int(prompt_tokens)

                    # Check for EOS
                    meta = out.get("meta_info", {})
                    finish_reason = meta.get("finish_reason", {})
                    is_stop = (
                        finish_reason == "stop" or
                        (isinstance(finish_reason, dict) and finish_reason.get("type") == "stop")
                    )
                    # Also treat eos_token_id as stop if present in output ids (more explicit)
                    eos_id = self.tokenizer.eos_token_id
                    hit_eos = (eos_id is not None and eos_id in out_ids)
                    if is_stop or hit_eos or n_output_tokens < cfg.gamma:
                        finished[active_idx] = True
                        if verbose:
                            print(f"  Particle {active_idx} hit EOS")
            # Create new particles (token IDs) and update token lengths
            new_particle_ids: List[List[int]] = [
                particle_ids[i] + full_output_ids[i] for i in range(cfg.n_particles)
            ]
            new_particle_token_lens = [len(ids) for ids in new_particle_ids]
            stats["total_draft_tokens"] += sum(full_token_counts)

            # 2. Score with target model - only active particles
            target_logprobs = np.zeros(cfg.n_particles)
            target_lps_per_token: List[List[float]] = [[] for _ in range(cfg.n_particles)]

            # NOTE: Do NOT skip target scoring just because extensions match.
            # The importance ratio p_target/p_draft generally depends on each particle's prefix.
            # Skipping is only safe if all active particle prefixes are identical (then ratios are constant).
            skip_target_scoring = False
            if active_indices:
                active_prefixes = [tuple(particle_ids[i]) for i in active_indices]
                active_extensions = [tuple(full_output_ids[i]) for i in active_indices]
                if len(set(active_prefixes)) == 1 and len(set(active_extensions)) == 1 and len(active_extensions[0]) > 0:
                    skip_target_scoring = True
                    stats["skipped_target_calls"] += 1
                    # Constant factor across particles; we can set log-importance to 0 safely.
                    for i in active_indices:
                        target_logprobs[i] = full_draft_logprobs[i]
                        if viz:
                            target_lps_per_token[i] = list(draft_lps_per_token[i])
                    if verbose:
                        print(f"  Skipping target scoring (all active particles truly identical)")

            # Compute logprob_start_lens from OLD prefix lengths before in-place extend
            if active_indices and not skip_target_scoring:
                active_prefix_lens = [particle_token_lens[i] for i in active_indices]
                logprob_start_lens = [max(0, pl - 1) for pl in active_prefix_lens]

            # Extend particles in-place (avoids O(L*N) full-prefix copy every step)
            for i in range(cfg.n_particles):
                particle_ids[i].extend(full_output_ids[i])
            particle_token_lens = [len(ids) for ids in particle_ids]
            stats["total_draft_tokens"] += sum(full_token_counts)

            if active_indices and not skip_target_scoring:
                active_new_input_ids = [particle_ids[i] for i in active_indices]
                active_token_counts = [full_token_counts[i] for i in active_indices]

                # Batched call to target engine
                t0 = time.perf_counter()
                with nvtx.annotate("target_score", color="red"):
                    target_result = self.target_engine.generate(
                        input_ids=active_new_input_ids,
                        sampling_params={
                            "max_new_tokens": 0,
                        },
                        return_logprob=True,
                        logprob_start_len=logprob_start_lens,
                    )
                stats["time_target_score"] += time.perf_counter() - t0

                if isinstance(target_result, list):
                    target_outputs = target_result
                else:
                    target_outputs = [target_result]

                for j, i in enumerate(active_indices):
                    out = target_outputs[j]
                    input_lps = out.get("meta_info", {}).get("input_token_logprobs", [])
                    n_cont_tokens = active_token_counts[j]

                    if n_cont_tokens > 0 and len(input_lps) >= n_cont_tokens:
                        cont_lps = input_lps[-n_cont_tokens:]
                        target_logprobs[i] = self._sum_logprobs(cont_lps)
                        target_logprobs[i] /= cfg.target_lhts_temperature

                        if viz:
                            per_tok = self._extract_logprobs(cont_lps, n_cont_tokens)
                            if cfg.lhts:
                                per_tok = [lp * cfg.target_temperature for lp in per_tok]
                            target_lps_per_token[i] = per_tok

                    meta_info = out.get("meta_info", {}) or {}
                    cached = meta_info.get("cached_tokens", 0)
                    prompt_tokens = meta_info.get("prompt_tokens", len(active_new_input_ids[j]))
                    if isinstance(cached, (int, float)):
                        stats["target_cached_tokens"] += int(cached)
                    if isinstance(prompt_tokens, (int, float)):
                        stats["target_total_tokens"] += int(prompt_tokens)

                stats["total_target_scores"] += len(active_indices)

            # Track average target logprob per token (for rejuvenation trigger)
            step_logps = []
            for i in active_indices:
                n_cont = full_token_counts[i]
                if n_cont > 0:
                    step_logps.append(target_logprobs[i] / n_cont)
            if step_logps:
                avg_step_logp = float(np.mean(step_logps))
                if logp_ema is None:
                    logp_ema = avg_step_logp
                else:
                    alpha = cfg.rejuvenation_logp_ema_alpha
                    logp_ema = alpha * avg_step_logp + (1 - alpha) * logp_ema
                stats["logp_ema"] = logp_ema
                if logp_ema < cfg.rejuvenation_logp_threshold:
                    low_logp_steps += 1
                else:
                    low_logp_steps = 0

            # 3. Update importance weights
            t0 = time.perf_counter()
            with nvtx.annotate("weight_update", color="green"):
                log_importance = target_logprobs - full_draft_logprobs
                for i in active_indices:
                    log_weights[i] += log_importance[i]
            stats["time_weight_update"] += time.perf_counter() - t0

            if verbose:
                weights = self._normalize_weights(log_weights)
                ess = self._effective_sample_size(weights)
                n_finished = sum(finished)
                print(f"Step {stats['steps']}: ESS={ess:.2f}, "
                      f"max_weight={weights.max():.3f}, "
                      f"finished={n_finished}/{cfg.n_particles}, "
                      f"cached(draft/target)={stats['draft_cached_tokens']}/{stats['target_cached_tokens']}")

            # 4. Resample if needed
            weights = self._normalize_weights(log_weights)
            ess = self._effective_sample_size(weights)

            did_resample = ess < cfg.n_particles * cfg.resample_threshold
            resampled_indices = None

            if did_resample:
                t0 = time.perf_counter()
                with nvtx.annotate("resample", color="yellow"):
                    old_finished = finished.copy()
                    old_token_lens = particle_token_lens[:]
                    # Resample token-id particles (clones lists to break aliasing)
                    particle_ids, log_weights, resampled_indices = self._resample(
                        particle_ids, weights
                    )
                    finished = [old_finished[idx] for idx in resampled_indices]
                    particle_token_lens = [old_token_lens[idx] for idx in resampled_indices]
                stats["time_resample"] += time.perf_counter() - t0
                stats["resample_count"] += 1
                if verbose:
                    print(f"  -> Resampled (ESS={ess:.2f})")

                # Rejuvenation after resampling if logp trigger fires.
                if (
                    cfg.rejuvenate
                    and low_logp_steps >= cfg.rejuvenation_trigger_steps
                    and any(not f for f in finished)
                ):
                    t0 = time.perf_counter()
                    with nvtx.annotate("rejuvenation", color="purple"):
                        self._rejuvenate_particles(
                            particle_ids=particle_ids,
                            particle_token_lens=particle_token_lens,
                            finished=finished,
                            prompt_len=prompt_len,
                            cfg=cfg,
                            stats=stats,
                        )
                    stats["time_rejuvenation"] += time.perf_counter() - t0
                    stats["rejuvenation_count"] += 1
            else:
                pass  # particle_ids and particle_token_lens already updated in-place

            if viz:
                step_record = {
                    "step": stats["steps"],
                    "particles": [
                        {
                            "particle_id": i,
                            "tokens": [
                                {
                                    "token_id": int(tid),
                                    "token_str": self.tokenizer.decode([tid]),
                                    "log_p_draft": float(draft_lps_per_token[i][j]) if j < len(draft_lps_per_token[i]) else 0.0,
                                    "log_p_target": float(target_lps_per_token[i][j]) if j < len(target_lps_per_token[i]) else 0.0,
                                }
                                for j, tid in enumerate(full_output_ids[i])
                            ],
                            "finished": bool(finished[i]),
                        }
                        for i in range(cfg.n_particles)
                    ],
                    "log_weights": log_weights.tolist(),
                    "weights": self._normalize_weights(log_weights).tolist(),
                    "ess": float(ess),
                    "resampled": bool(did_resample),
                    "resampled_indices": [int(x) for x in resampled_indices] if did_resample else None,
                }
                viz_steps.append(step_record)

            tokens_generated += cfg.gamma
            stats["steps"] += 1
            if first_step_end is None:
                first_step_end = time.time()
                first_step_tokens = cfg.gamma
            nvtx.pop_range()

        end_time = time.time()
        stats["elapsed_time"] = end_time - start_time
        stats["tokens_per_second"] = tokens_generated / stats["elapsed_time"]

        # TTFT = time of first complete step (includes prefill for draft + target)
        stats["ttft"] = (first_step_end - start_time) if first_step_end else stats["elapsed_time"]
        decode_time = stats["elapsed_time"] - stats["ttft"]
        decode_tokens = tokens_generated - first_step_tokens
        stats["decode_tps"] = decode_tokens / decode_time if decode_time > 0 else 0.0

        # Compute timing percentages
        total_profiled = (stats["time_draft_gen"] + stats["time_target_score"] +
                          stats["time_weight_update"] + stats["time_resample"])
        if total_profiled > 0:
            stats["pct_draft_gen"] = 100 * stats["time_draft_gen"] / total_profiled
            stats["pct_target_score"] = 100 * stats["time_target_score"] / total_profiled
            stats["pct_weight_update"] = 100 * stats["time_weight_update"] / total_profiled
            stats["pct_resample"] = 100 * stats["time_resample"] / total_profiled
            stats["pct_other"] = 100 * (stats["elapsed_time"] - total_profiled) / stats["elapsed_time"]

        # Compute cache hit ratios
        if stats["draft_total_tokens"] > 0:
            stats["draft_cache_hit_ratio"] = stats["draft_cached_tokens"] / stats["draft_total_tokens"]
        if stats["target_total_tokens"] > 0:
            stats["target_cache_hit_ratio"] = stats["target_cached_tokens"] / stats["target_total_tokens"]

        # Return highest weight particle
        best_idx = np.argmax(log_weights)
        best_ids = particle_ids[best_idx]
        generated_ids = best_ids[prompt_len:]
        stats["output_token_count"] = len(generated_ids)
        generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if return_particles:
            # Expose final particle completions for oracle/selection analyses.
            weights = self._normalize_weights(log_weights)
            stats["particle_texts"] = [
                self.tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
                for ids in particle_ids
            ]
            stats["particle_log_weights"] = log_weights.tolist()
            stats["particle_weights"] = weights.tolist()
            stats["selected_particle_idx"] = int(best_idx)

        if viz:
            stats["viz_data"] = {
                "config": {
                    "mode": cfg.mode,
                    "n_particles": cfg.n_particles,
                    "gamma": cfg.gamma,
                    "draft_temperature": cfg.draft_temperature,
                    "target_temperature": cfg.target_temperature,
                    "lhts": cfg.lhts,
                    "resample_threshold": cfg.resample_threshold,
                    "rejuvenate": cfg.rejuvenate,
                },
                "prompt": prompt,
                "steps": viz_steps,
            }

        return generated, stats

    def _decode_sd_like(
        self,
        prompt: str,
        max_tokens: int = 128,
        verbose: bool = False,
        return_particles: bool = False,
    ) -> Tuple[str, dict]:
        """
        SD-like decoding in this framework:
        - single particle
        - no SMC weighting / ESS / resampling / rejuvenation
        - draft proposes a chunk, target token-wise accepts/rejects
        """
        cfg = self.config
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        current_ids = prompt_ids[:]

        stats = {
            "mode": "sd_like",
            "steps": 0,
            "total_draft_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "total_target_scores": 0,
            "resample_count": 0,
            "rejuvenation_count": 0,
            "time_draft_gen": 0.0,
            "time_target_score": 0.0,
            "time_weight_update": 0.0,
            "time_resample": 0.0,
            "time_rejuvenation": 0.0,
        }

        start_time = time.time()
        tokens_generated = 0
        first_step_end = None
        first_step_tokens = 0
        eos_id = self.tokenizer.eos_token_id

        while tokens_generated < max_tokens:
            remaining = max_tokens - tokens_generated
            n_prop = min(cfg.gamma, remaining)

            # 1) Draft proposes chunk
            t0 = time.perf_counter()
            draft_out = self.draft_engine.generate(
                input_ids=[current_ids],
                sampling_params={
                    "max_new_tokens": n_prop,
                    "temperature": cfg.draft_temperature,
                },
                return_logprob=True,
            )
            stats["time_draft_gen"] += time.perf_counter() - t0
            d = draft_out[0] if isinstance(draft_out, list) else draft_out
            draft_ids = list(d.get("output_ids", []) or [])
            if not draft_ids:
                break
            draft_lps_raw = d.get("meta_info", {}).get("output_token_logprobs", []) or []
            draft_lps = self._extract_logprobs(draft_lps_raw, len(draft_ids))
            stats["total_draft_tokens"] += len(draft_ids)

            # 2) Target scores proposed chunk
            t0 = time.perf_counter()
            target_out = self.target_engine.generate(
                input_ids=[current_ids + draft_ids],
                sampling_params={
                    "max_new_tokens": 0,
                },
                return_logprob=True,
                logprob_start_len=max(0, len(current_ids) - 1),
            )
            stats["time_target_score"] += time.perf_counter() - t0
            t = target_out[0] if isinstance(target_out, list) else target_out
            input_lps = t.get("meta_info", {}).get("input_token_logprobs", []) or []
            target_lps = self._extract_logprobs(input_lps[-len(draft_ids):], len(draft_ids))
            stats["total_target_scores"] += 1

            # 3) Token-wise accept/reject based on target/draft ratio
            accepted_prefix = 0
            for j, tok in enumerate(draft_ids):
                log_ratio = target_lps[j] - draft_lps[j]
                if np.log(np.random.uniform()) < min(0.0, log_ratio):
                    current_ids.append(tok)
                    accepted_prefix += 1
                    stats["accepted_tokens"] += 1
                    tokens_generated += 1
                    if eos_id is not None and tok == eos_id:
                        break
                else:
                    stats["rejected_tokens"] += 1
                    # Fallback token from target model
                    fb = self.target_engine.generate(
                        input_ids=[current_ids],
                        sampling_params={
                            "max_new_tokens": 1,
                        },
                    )
                    f = fb[0] if isinstance(fb, list) else fb
                    fb_ids = list(f.get("output_ids", []) or [])
                    if fb_ids:
                        current_ids.append(fb_ids[0])
                        tokens_generated += 1
                    break

            if verbose:
                print(
                    f"Step {stats['steps']}: proposed={len(draft_ids)}, "
                    f"accepted_prefix={accepted_prefix}, total_gen={tokens_generated}"
                )
            stats["steps"] += 1
            if first_step_end is None:
                first_step_end = time.time()
                first_step_tokens = tokens_generated

            if eos_id is not None and current_ids and current_ids[-1] == eos_id:
                break

        generated_ids = current_ids[len(prompt_ids):]
        generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        stats["elapsed_time"] = time.time() - start_time
        stats["tokens_per_second"] = (
            tokens_generated / stats["elapsed_time"] if stats["elapsed_time"] > 0 else 0.0
        )
        stats["ttft"] = (first_step_end - start_time) if first_step_end else stats["elapsed_time"]
        decode_time = stats["elapsed_time"] - stats["ttft"]
        decode_tokens = tokens_generated - first_step_tokens
        stats["decode_tps"] = decode_tokens / decode_time if decode_time > 0 else 0.0
        stats["output_token_count"] = len(generated_ids)

        if return_particles:
            stats["particle_texts"] = [generated]
            stats["particle_log_weights"] = [0.0]
            stats["particle_weights"] = [1.0]
            stats["selected_particle_idx"] = 0

        return generated, stats

    def _sum_logprobs(self, logprobs_list) -> float:
        """Sum logprobs from sglang output format. Optimized with fast paths."""
        if not logprobs_list:
            return 0.0

        # Fast path: check first item to determine format
        first = logprobs_list[0]

        if first is None:
            # Fallback to slow path if first is None
            return self._sum_logprobs_slow(logprobs_list)
        elif isinstance(first, (int, float)):
            # Direct numeric values - fastest path
            return float(sum(x for x in logprobs_list if x is not None))
        elif isinstance(first, (list, tuple)):
            # Tuple format [(logprob, token_id), ...] - extract first element
            return float(sum(x[0] for x in logprobs_list if x and x[0] is not None))
        else:
            return self._sum_logprobs_slow(logprobs_list)

    def _sum_logprobs_slow(self, logprobs_list) -> float:
        """Fallback slow path with full type checking."""
        total = 0.0
        for item in logprobs_list:
            if item is None:
                continue
            elif isinstance(item, (int, float)):
                total += float(item)
            elif isinstance(item, (list, tuple)) and len(item) > 0:
                if item[0] is not None:
                    total += float(item[0])
        return total

    def _extract_logprobs(self, logprobs_list, n_expected: int) -> List[float]:
        """Extract scalar logprobs from sglang list/tuple/mixed formats."""
        vals: List[float] = []
        for item in logprobs_list:
            if item is None:
                continue
            if isinstance(item, (int, float)):
                vals.append(float(item))
            elif isinstance(item, (list, tuple)) and len(item) > 0 and item[0] is not None:
                vals.append(float(item[0]))
        if len(vals) < n_expected:
            vals = vals + [0.0] * (n_expected - len(vals))
        return vals[:n_expected]

    def _normalize_weights(self, log_weights: np.ndarray) -> np.ndarray:
        """Normalize log weights to probabilities."""
        max_lw = np.max(log_weights)
        weights = np.exp(log_weights - max_lw)
        return weights / weights.sum()

    def _effective_sample_size(self, weights: np.ndarray) -> float:
        """Compute effective sample size."""
        return 1.0 / np.sum(weights ** 2)

    def _resample(
        self,
        particles: List,
        weights: np.ndarray,
    ) -> Tuple[List, np.ndarray, List[int]]:
        """Resample particles using the configured method."""
        if self.config.resample_method == "multinomial":
            return self._resample_multinomial(particles, weights)
        return self._resample_systematic(particles, weights)

    def _resample_systematic(
        self,
        particles: List,
        weights: np.ndarray,
    ) -> Tuple[List, np.ndarray, List[int]]:
        """Systematic resampling."""
        n = len(particles)
        positions = (np.arange(n) + np.random.uniform()) / n
        cumsum = np.cumsum(weights)
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, n - 1)

        new_particles = [list(particles[i]) for i in indices]
        new_log_weights = np.zeros(n)
        return new_particles, new_log_weights, list(indices)

    def _resample_multinomial(
        self,
        particles: List,
        weights: np.ndarray,
    ) -> Tuple[List, np.ndarray, List[int]]:
        """Multinomial resampling."""
        n = len(particles)
        indices = np.random.choice(n, size=n, replace=True, p=weights)

        new_particles = [list(particles[i]) for i in indices]
        new_log_weights = np.zeros(n)
        return new_particles, new_log_weights, list(indices)

    def _rejuvenate_particles(
        self,
        particle_ids: List[List[int]],
        particle_token_lens: List[int],
        finished: List[bool],
        prompt_len: int,
        cfg: SMCConfig,
        stats: dict,
    ) -> None:
        """Rejuvenate a subset of particles by re-sampling a suffix with MH."""
        # Select subset of active particles
        active_indices = [i for i, f in enumerate(finished) if not f]
        if not active_indices:
            return

        n_active = len(active_indices)
        n_rejuv = max(1, int(cfg.rejuvenation_frac * n_active))
        rejuv_indices = list(np.random.choice(active_indices, size=n_rejuv, replace=False))

        # Build prefixes and propose suffixes with draft model
        prefixes = []
        prefix_lens = []
        for i in rejuv_indices:
            ids = particle_ids[i]
            cut = max(prompt_len, len(ids) - cfg.rejuvenation_k)
            prefix = ids[:cut]
            prefixes.append(prefix)
            prefix_lens.append(len(prefix))

        draft_result = self.draft_engine.generate(
            input_ids=prefixes,
            sampling_params={
                "max_new_tokens": cfg.rejuvenation_k,
                "temperature": cfg.draft_temperature,
            },
            return_logprob=True,
        )
        draft_outputs = draft_result if isinstance(draft_result, list) else [draft_result]

        # Build candidate full sequences and draft logprobs
        cand_full_ids = []
        cand_suffix_lens = []
        draft_logps = []
        for out in draft_outputs:
            out_ids = out.get("output_ids", []) or []
            output_lps = out.get("meta_info", {}).get("output_token_logprobs", [])
            draft_logps.append(self._sum_logprobs(output_lps))
            cand_suffix_lens.append(len(out_ids))
            cand_full_ids.append(out_ids)

        # Score proposed suffixes with target model
        target_inputs = []
        logprob_start_lens = []
        for idx, prefix in enumerate(prefixes):
            suffix = cand_full_ids[idx]
            target_inputs.append(prefix + suffix)
            logprob_start_lens.append(max(0, len(prefix) - 1))

        target_result = self.target_engine.generate(
            input_ids=target_inputs,
            sampling_params={
                "max_new_tokens": 0,
            },
            return_logprob=True,
            logprob_start_len=logprob_start_lens,
        )
        target_outputs = target_result if isinstance(target_result, list) else [target_result]

        for j, i in enumerate(rejuv_indices):
            n_cont = cand_suffix_lens[j]
            stats["rejuvenation_attempts"] += 1
            if n_cont <= 0:
                continue

            out = target_outputs[j]
            input_lps = out.get("meta_info", {}).get("input_token_logprobs", [])
            if len(input_lps) < n_cont:
                continue

            cont_lps = input_lps[-n_cont:]
            target_logp = self._sum_logprobs(cont_lps)
            target_logp /= cfg.target_lhts_temperature
            draft_logp = draft_logps[j]

            # MH accept
            log_accept_ratio = target_logp - draft_logp
            if np.log(np.random.uniform()) < min(0.0, log_accept_ratio):
                # Accept: replace suffix
                prefix = prefixes[j]
                suffix = cand_full_ids[j]
                new_ids = prefix + suffix
                particle_ids[i] = new_ids
                particle_token_lens[i] = len(new_ids)
                stats["rejuvenation_accepts"] += 1

    def shutdown(self):
        """Cleanup engines."""
        print("Shutting down engines...")
        if hasattr(self, 'draft_engine'):
            self.draft_engine.shutdown()
        if hasattr(self, 'target_engine'):
            self.target_engine.shutdown()


def evaluate_gsm8k(
    decoder: NativeSMCDecoder,
    num_samples: int = 20,
    selection_mode: str = "argmax",
) -> Dict:
    """Evaluate on GSM8K dataset."""
    from datasets import load_dataset
    import re

    def extract_answer(text: str) -> Optional[str]:
        match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(",", "")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        last_line = lines[-1] if lines else text.strip()
        numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", last_line)
        return numbers[-1].replace(",", "") if numbers else None

    print(f"Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")

    correct = 0
    argmax_correct_total = 0
    oracle_correct_total = 0
    total_output_tokens = 0
    total_time = 0
    total_decode_time = 0
    details = []

    print(f"Evaluating on {num_samples} samples...")

    for i, sample in enumerate(dataset.select(range(num_samples))):
        instruction = (
            "Solve this math problem step by step.\n"
            "At the very end, output ONLY the final numeric answer on a new line in the exact format:\n"
            "#### <number>\n\n"
            f"Problem:\n{sample['question']}\n"
        )
        prompt = decoder.format_user_prompt(instruction)

        generated, stats = decoder.decode(
            prompt,
            max_tokens=512,
            verbose=False,
            return_particles=(selection_mode == "oracle"),
        )

        pred = extract_answer(generated)
        gold = extract_answer(sample["answer"])

        argmax_correct = pred == gold
        argmax_correct_total += int(argmax_correct)

        oracle_correct = argmax_correct
        selected_particle_idx = stats.get("selected_particle_idx")
        if selection_mode == "oracle":
            particle_texts = stats.get("particle_texts", [])
            particle_preds = [extract_answer(txt) for txt in particle_texts]
            correct_idxs = [idx for idx, p in enumerate(particle_preds) if p == gold]
            oracle_correct = len(correct_idxs) > 0
            if oracle_correct:
                selected_particle_idx = correct_idxs[0]
                generated = particle_texts[selected_particle_idx]
                pred = particle_preds[selected_particle_idx]
            else:
                pred = extract_answer(generated)
        oracle_correct_total += int(oracle_correct)

        is_correct = oracle_correct if selection_mode == "oracle" else argmax_correct
        if is_correct:
            correct += 1

        output_tokens = stats.get("output_token_count", 0)
        total_output_tokens += output_tokens
        total_time += stats["elapsed_time"]
        total_decode_time += (stats["elapsed_time"] - stats.get("ttft", 0))
        decode_tps = stats.get("decode_tps", 0.0)

        print(f"[{i+1}/{num_samples}] Pred: {pred}, Gold: {gold}, "
              f"Correct: {is_correct}, Decode TPS: {decode_tps:.1f}, TTFT: {stats.get('ttft', 0):.3f}s")

        details.append({
            "question": sample["question"],
            "gold_answer": sample["answer"],
            "generated": generated,
            "predicted": pred,
            "gold": gold,
            "correct": is_correct,
            "selection_mode": selection_mode,
            "argmax_correct": argmax_correct,
            "oracle_correct": oracle_correct,
            "selected_particle_idx": selected_particle_idx,
            "num_tokens": output_tokens,
            "decode_tps": decode_tps,
            "ttft": stats.get("ttft", 0),
            "generation_time": stats["elapsed_time"],
        })

    per_sample_tps = [d["decode_tps"] for d in details]
    # Filter: remove zeros, samples above median + 1000 TPS, and the lowest 3
    nonzero_tps = [t for t in per_sample_tps if t > 0]
    median_tps = float(np.median(nonzero_tps)) if nonzero_tps else 0
    filtered_tps = sorted([t for t in nonzero_tps if t <= median_tps + 1000])
    filtered_tps = filtered_tps[3:] if len(filtered_tps) > 3 else filtered_tps
    avg_decode_tps = float(np.mean(filtered_tps)) if filtered_tps else 0

    results = {
        "selection_mode": selection_mode,
        "accuracy": correct / num_samples,
        "correct": correct,
        "total": num_samples,
        "argmax_accuracy": argmax_correct_total / num_samples,
        "oracle_at_n": oracle_correct_total / num_samples,
        "decode_tps": avg_decode_tps,
        "total_output_tokens": total_output_tokens,
        "total_time": total_time,
        "details": details,
    }

    print(f"\n{'='*50}")
    print(f"NATIVE SMC RESULTS: N={decoder.config.n_particles}, γ={decoder.config.gamma}")
    print(f"{'='*50}")
    print(f"Selection mode: {selection_mode}")
    print(f"Accuracy: {correct}/{num_samples} ({100*results['accuracy']:.1f}%)")
    if selection_mode == "oracle":
        print(f"Argmax accuracy: {100*results['argmax_accuracy']:.1f}%")
        print(f"Oracle@N: {100*results['oracle_at_n']:.1f}%")
    print(f"Decode TPS: {avg_decode_tps:.1f} (filtered: dropped >median+1000 TPS and lowest 3)")
    print(f"Total Time: {total_time:.1f}s")

    return results


def benchmark_target_only(target_engine, tokenizer, num_samples: int = 20) -> Dict:
    """Benchmark target model only for comparison."""
    from datasets import load_dataset
    import re

    def extract_answer(text: str) -> Optional[str]:
        match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(",", "")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        last_line = lines[-1] if lines else text.strip()
        numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", last_line)
        return numbers[-1].replace(",", "") if numbers else None

    print(f"\nBenchmarking target model only...")
    dataset = load_dataset("gsm8k", "main", split="test")

    correct = 0
    total_output_tokens = 0
    start_time = time.time()
    details = []

    for i, sample in enumerate(dataset.select(range(num_samples))):
        instruction = (
            "Solve this math problem step by step.\n"
            "At the very end, output ONLY the final numeric answer on a new line in the exact format:\n"
            "#### <number>\n\n"
            f"Problem:\n{sample['question']}\n"
        )
        # Apply chat template so the instruct model sees proper formatting
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )

        t0 = time.time()
        result = target_engine.generate(
            prompt=prompt,
            sampling_params={
                "max_new_tokens": 512,
                "temperature": 0.7,
            },
        )
        gen_time = time.time() - t0

        generated = result.get("text", "")

        pred = extract_answer(generated)
        gold = extract_answer(sample["answer"])

        is_correct = pred == gold
        if is_correct:
            correct += 1

        output_tokens = len(tokenizer.encode(generated, add_special_tokens=False))
        total_output_tokens += output_tokens
        output_tps = output_tokens / gen_time if gen_time > 0 else 0

        elapsed = time.time() - start_time
        tps = total_output_tokens / elapsed if elapsed > 0 else 0
        print(f"[{i+1}/{num_samples}] Pred: {pred}, Gold: {gold}, "
              f"Correct: {is_correct}, Output TPS: {output_tps:.1f}")

        details.append({
            "question": sample["question"],
            "gold_answer": sample["answer"],
            "generated": generated,
            "predicted": pred,
            "gold": gold,
            "correct": is_correct,
            "num_tokens": output_tokens,
            "tps": output_tps,
            "generation_time": gen_time,
        })

    total_time = time.time() - start_time
    per_sample_tps = [d["tps"] for d in details]
    # Filter: remove zeros, samples above median + 1000 TPS, and the lowest 3
    nonzero_tps = [t for t in per_sample_tps if t > 0]
    median_tps = float(np.median(nonzero_tps)) if nonzero_tps else 0
    filtered_tps = sorted([t for t in nonzero_tps if t <= median_tps + 1000])
    filtered_tps = filtered_tps[3:] if len(filtered_tps) > 3 else filtered_tps
    avg_output_tps = float(np.mean(filtered_tps)) if filtered_tps else 0

    print(f"\n{'='*50}")
    print(f"TARGET ONLY RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {correct}/{num_samples} ({100*correct/num_samples:.1f}%)")
    print(f"Output TPS: {avg_output_tps:.1f} (filtered: dropped >median+1000 TPS and lowest 3)")
    print(f"Total Time: {total_time:.1f}s")

    return {
        "accuracy": correct / num_samples,
        "output_tps": avg_output_tps,
        "total_time": total_time,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Native SMC Speculative Decoding")

    # Model configuration
    parser.add_argument("--draft-model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--draft-mem", type=float, default=0.25,
                        help="GPU memory fraction for draft model")
    parser.add_argument("--target-mem", type=float, default=0.55,
                        help="GPU memory fraction for target model")

    # SMC configuration
    parser.add_argument("--mode", choices=["smc", "sd_like"], default="smc",
                        help="Decoding mode: full SMC or sd_like single-particle draft/target accept.")
    parser.add_argument("--n-particles", "-N", type=int, default=4)
    parser.add_argument("--gamma", "-g", type=int, default=8)
    parser.add_argument("--draft-temperature", type=float, default=0.7)
    parser.add_argument("--target-lhts-temperature", type=float, default=1.0,
                        help="LHTS temperature scaling for target logprobs (1.0 = no scaling)")
    parser.add_argument(
        "--resample-threshold",
        type=float,
        default=0.5,
        help="Resampling threshold as a fraction of N (resample when ESS < N * threshold).",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rejuvenate", action="store_true",
                        help="Enable rejuvenation after resampling when logp is low")
    parser.add_argument("--rejuvenation-k", type=int, default=16,
                        help="Suffix length to re-sample during rejuvenation")
    parser.add_argument("--rejuvenation-frac", type=float, default=0.5,
                        help="Fraction of active particles to rejuvenate")
    parser.add_argument("--rejuvenation-logp-threshold", type=float, default=-3.0,
                        help="EMA logprob threshold to trigger rejuvenation")
    parser.add_argument("--rejuvenation-logp-ema-alpha", type=float, default=0.3,
                        help="EMA smoothing for logprob trigger")
    parser.add_argument("--rejuvenation-trigger-steps", type=int, default=2,
                        help="Consecutive low-logp steps required to trigger rejuvenation")
    parser.add_argument("--resample", choices=["systematic", "multinomial"], default="systematic",
                        help="Resampling method: systematic or multinomial")

    # Mode
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--eval", action="store_true", help="Run GSM8K evaluation")
    parser.add_argument("--compare", action="store_true", help="Compare with target-only")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument(
        "--selection-mode",
        choices=["argmax", "oracle"],
        default="argmax",
        help="How to select final output in SMC eval: argmax weight or oracle@N (any particle correct).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Seed for numpy resampling (for reproducible runs)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    parser.add_argument("--viz", action="store_true", help="Save per-token viz data to out/")

    args = parser.parse_args()

    config = SMCConfig(
        mode=args.mode,
        n_particles=args.n_particles,
        gamma=args.gamma,
        draft_temperature=args.draft_temperature,
        target_lhts_temperature=args.target_lhts_temperature,
        resample_threshold=args.resample_threshold,
        rejuvenate=args.rejuvenate,
        rejuvenation_k=args.rejuvenation_k,
        rejuvenation_frac=args.rejuvenation_frac,
        rejuvenation_logp_threshold=args.rejuvenation_logp_threshold,
        rejuvenation_logp_ema_alpha=args.rejuvenation_logp_ema_alpha,
        rejuvenation_trigger_steps=args.rejuvenation_trigger_steps,
        resample_method=args.resample,
    )

    if args.mode == "sd_like":
        print("Mode sd_like: resampling/rejuvenation-related knobs are ignored.")

    decoder = NativeSMCDecoder(
        draft_model=args.draft_model,
        target_model=args.target_model,
        config=config,
        draft_mem_fraction=args.draft_mem,
        target_mem_fraction=args.target_mem,
    )

    try:
        if args.seed is not None:
            np.random.seed(args.seed)
        all_results = {}

        if args.compare:
            # Run target-only baseline first
            target_results = benchmark_target_only(
                decoder.target_engine, decoder.tokenizer, args.num_samples
            )
            all_results["target_only"] = target_results

            # Run SMC
            print()
            smc_results = evaluate_gsm8k(
                decoder,
                args.num_samples,
                selection_mode=args.selection_mode,
            )
            all_results["smc"] = smc_results

            # Summary
            print("\n" + "="*60)
            print("COMPARISON SUMMARY (Native Engine)")
            print("="*60)
            speedup = smc_results['decode_tps'] / target_results['output_tps'] if target_results['output_tps'] > 0 else 0
            print(f"{'Method':<25} {'Accuracy':<12} {'Output TPS':<12} {'Speedup':<10}")
            print("-"*60)
            print(f"{'Target-only':<25} {100*target_results['accuracy']:.1f}%{'':<7} {target_results['output_tps']:.1f}{'':<8} 1.00x")
            print(f"{'SMC N='+str(config.n_particles)+' γ='+str(config.gamma):<25} {100*smc_results['accuracy']:.1f}%{'':<7} {smc_results['decode_tps']:.1f}{'':<8} {speedup:.2f}x")
            print("="*60)

        elif args.eval:
            smc_results = evaluate_gsm8k(
                decoder,
                args.num_samples,
                selection_mode=args.selection_mode,
            )
            all_results["smc"] = smc_results

        elif args.prompt:
            print(f"\nPrompt: {args.prompt}\n")

            prompt = decoder.format_user_prompt(args.prompt)

            generated, stats = decoder.decode(
                prompt,
                max_tokens=args.max_tokens,
                verbose=args.verbose,
                viz=args.viz,
            )

            if args.viz and "viz_data" in stats:
                import json, os, datetime
                os.makedirs("out", exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                viz_data = stats.pop("viz_data")
                viz_data["timestamp"] = datetime.datetime.now().isoformat()
                viz_path = f"out/smc-native-{ts}.json"
                with open(viz_path, "w") as f:
                    json.dump(viz_data, f, indent=2)
                print(f"Viz data saved to {viz_path}")

            print(f"Generated: {generated}")
            print(f"\n--- Stats ---")
            print(f"Decode TPS: {stats.get('decode_tps', 0):.2f} (excludes prefill)")
            print(f"TTFT: {stats.get('ttft', 0):.3f}s")
            print(f"Time: {stats['elapsed_time']:.2f}s")
            print(f"Resample count: {stats['resample_count']}")
            print(f"Steps: {stats['steps']}")
            print(f"Skipped target calls: {stats.get('skipped_target_calls', 0)}")

            # Profiling breakdown
            if "pct_draft_gen" in stats:
                print(f"\n--- Timing Breakdown ---")
                print(f"Draft generation: {stats['time_draft_gen']:.3f}s ({stats['pct_draft_gen']:.1f}%)")
                print(f"Target scoring:   {stats['time_target_score']:.3f}s ({stats['pct_target_score']:.1f}%)")
                print(f"Weight update:    {stats['time_weight_update']:.3f}s ({stats['pct_weight_update']:.1f}%)")
                print(f"Resampling:       {stats['time_resample']:.3f}s ({stats['pct_resample']:.1f}%)")
                print(f"Other overhead:   {stats.get('pct_other', 0):.1f}%")

            # Cache stats
            if stats.get("draft_cache_hit_ratio") is not None or stats.get("target_cache_hit_ratio") is not None:
                print(f"\n--- Cache Hit Ratios ---")
                if stats.get("draft_cache_hit_ratio") is not None:
                    print(f"Draft cache:  {100*stats['draft_cache_hit_ratio']:.1f}%")
                if stats.get("target_cache_hit_ratio") is not None:
                    print(f"Target cache: {100*stats['target_cache_hit_ratio']:.1f}%")

        else:
            print("Please provide --prompt, --eval, or --compare")

        # Save results to JSON if requested
        if args.output and all_results:
            import json
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    finally:
        decoder.shutdown()


if __name__ == "__main__":
    main()