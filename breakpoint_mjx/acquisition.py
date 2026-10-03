"""Active-learning acquisition (learned layer, M2).

Black-box search (CMA-ES, MAP-Elites) spends an expensive GPU rollout on every theta
it tries. With a surrogate ENSEMBLE (M1) we can instead score a huge pool of candidate
worlds for FREE on the CPU and simulate only the most INFORMATIVE ones: worlds that sit
on the predicted failure boundary AND where the ensemble disagrees (high epistemic
uncertainty). That is where a real rollout resolves the most uncertainty, so the failure
boundary gets mapped in far fewer GPU evaluations.

acquisition = boundary_proximity * uncertainty, with greedy diversity so the batch is
not N near-duplicates. Pure numpy; unit-tested against a known boundary.
"""
from __future__ import annotations

import numpy as np


def boundary_uncertainty_score(mean: np.ndarray, std: np.ndarray, threshold: float) -> np.ndarray:
    """High when a candidate is BOTH near the failure threshold and uncertain.

    boundary term = exp(-((mean - threshold) / scale)^2): peaks at the decision boundary.
    uncertainty term = std (ensemble disagreement). Product = classic BALD-style value.
    """
    scale = max(threshold, 1e-3)
    boundary = np.exp(-(((mean - threshold) / scale) ** 2))
    return boundary * std


def select_informative(ensemble, candidates, n: int, *, diversity_frac: float = 0.15):
    """Pick n candidate rows to simulate next. Greedy: repeatedly take the highest-value
    remaining candidate, then suppress others too close to it (bound-normalized L2 <
    diversity_frac) so the batch spans the boundary instead of clustering.

    Returns the selected row indices (into `candidates`)."""
    cand = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
    mean, std = ensemble.predict(cand)
    value = boundary_uncertainty_score(mean, std, ensemble.threshold)

    lo = ensemble.members[0].lo
    span = ensemble.members[0].span
    norm = (cand - lo) / span  # [0,1] per dim for a scale-free distance

    chosen: list[int] = []
    available = np.ones(len(cand), dtype=bool)
    order = np.argsort(-value)
    for idx in order:
        if len(chosen) >= n:
            break
        if not available[idx]:
            continue
        chosen.append(int(idx))
        d = np.linalg.norm(norm - norm[idx], axis=1) / np.sqrt(norm.shape[1])
        available &= d >= diversity_frac
        available[idx] = False
    # if diversity suppressed too many, fill by raw value
    if len(chosen) < n:
        for idx in order:
            if len(chosen) >= n:
                break
            if idx not in chosen:
                chosen.append(int(idx))
    return chosen[:n]


def sample_candidates(space, n: int, seed: int = 0) -> np.ndarray:
    """Uniform candidate pool over the theta bounds (the free CPU pool to score)."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(space.lower(), dtype=np.float64)
    hi = np.asarray(space.upper(), dtype=np.float64)
    return lo + rng.random((n, len(lo))) * (hi - lo)
