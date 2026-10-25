"""Space-filling baseline samplers that the engine does not already provide.

Uniform and LHS + boundary (RoboGate-style) already live in ``breakpoint_mjx.equal_budget``
and are reused unchanged. This module adds only the scrambled Sobol baseline.
"""
from __future__ import annotations

import warnings

import numpy as np

SOBOL_STREAM = 99   # distinct from equal_budget's per-arm streams (indices 0..3)


def sobol_points(space, n: int, seed: int) -> np.ndarray:
    """``n`` scrambled-Sobol points mapped onto the space's box, (n, D).

    Scrambling (Owen) with a seeded generator makes repetitions independent. ``n`` need
    not be a power of two; SciPy warns that balance properties weaken in that case, which
    is accepted and recorded rather than changing the shared budget for one method."""
    from scipy.stats import qmc

    lo = np.asarray(space.lower(), dtype=float)
    hi = np.asarray(space.upper(), dtype=float)
    eng = qmc.Sobol(d=len(lo), scramble=True,
                    seed=np.random.default_rng([int(seed), SOBOL_STREAM]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        u = eng.random(int(n))
    return lo + u * (hi - lo)


def sobol_rollout_key(seed: int) -> int:
    """Integer seed for the Sobol method's rollout-key stream (disjoint from others)."""
    return int(seed) * 7919 + 101 * (SOBOL_STREAM + 1)
