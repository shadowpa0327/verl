"""Pure helper functions extracted from NativeSMCDecoder."""

from typing import List, Tuple

import numpy as np


def sum_logprobs(logprobs_list) -> float:
    """Sum logprobs from sglang output format. Optimized with fast paths."""
    if not logprobs_list:
        return 0.0

    first = logprobs_list[0]

    if first is None:
        return _sum_logprobs_slow(logprobs_list)
    elif isinstance(first, (int, float)):
        return float(sum(x for x in logprobs_list if x is not None))
    elif isinstance(first, (list, tuple)):
        return float(sum(x[0] for x in logprobs_list if x and x[0] is not None))
    else:
        return _sum_logprobs_slow(logprobs_list)


def _sum_logprobs_slow(logprobs_list) -> float:
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


def extract_logprobs(logprobs_list, n_expected: int) -> List[float]:
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


def normalize_weights(log_weights: np.ndarray) -> np.ndarray:
    """Normalize log weights to probabilities."""
    max_lw = np.max(log_weights)
    weights = np.exp(log_weights - max_lw)
    return weights / weights.sum()


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute effective sample size."""
    return 1.0 / np.sum(weights**2)


def resample(
    particles: List,
    weights: np.ndarray,
    method: str = "systematic",
) -> Tuple[List, np.ndarray, List[int]]:
    """Resample particles using the specified method."""
    if method == "multinomial":
        return resample_multinomial(particles, weights)
    return resample_systematic(particles, weights)


def resample_systematic(
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


def resample_multinomial(
    particles: List,
    weights: np.ndarray,
) -> Tuple[List, np.ndarray, List[int]]:
    """Multinomial resampling."""
    n = len(particles)
    indices = np.random.choice(n, size=n, replace=True, p=weights)

    new_particles = [list(particles[i]) for i in indices]
    new_log_weights = np.zeros(n)
    return new_particles, new_log_weights, list(indices)
