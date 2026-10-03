"""Cross-policy failure embedding (learned layer, M4).

Each run gets a fixed-length "failure fingerprint" that is independent of how many
families it found, computed from WHERE in parameter space the policy fails and how
severely. Two runs that fail the same way land close together even across different
robots, policy versions, or search modes. That unlocks:

  * "this Go1 checkpoint fails the same way that G1 one did" (cross-policy comparison),
  * warm-starting a new policy's search from the hot regions of its nearest neighbors
    instead of from scratch (transfer), and
  * a library that gets more predictive as more policies are tested.

Pure numpy; unit-tested that same-mode runs embed close and different-mode runs far.
"""
from __future__ import annotations

import numpy as np

_FAILURE_MODES = None  # imported lazily to keep this module import-light


def _norm_theta(theta_named, space):
    lo = np.asarray(space.lower(), dtype=np.float64)
    hi = np.asarray(space.upper(), dtype=np.float64)
    nominal = np.asarray(space.nominal_vector(), dtype=np.float64)
    span = np.where(hi > lo, hi - lo, 1.0)
    vec = np.array([float(theta_named.get(n, nominal[i])) for i, n in enumerate(space.names())])
    return (vec - nominal) / span  # signed deviation from nominal, in bound units


def run_embedding(families, space, failure_rate: float = 0.0) -> np.ndarray:
    """Fixed-length failure fingerprint (length 2*D + 1).

    Per theta dim: (1) the severity-weighted signed deviation from nominal across the
    run's families, and (2) the severity-weighted |deviation| (dominance magnitude).
    Plus the overall failure rate. Severity = family mean_failure_score.
    """
    names = space.names()
    d = len(names)
    if not families:
        return np.zeros(2 * d + 1)
    devs = np.stack([_norm_theta(f.triggering_parameters, space) for f in families], axis=0)
    w = np.array([max(float(f.mean_failure_score), 1e-6) for f in families])
    w = w / w.sum()
    signed = (devs * w[:, None]).sum(axis=0)
    magnitude = (np.abs(devs) * w[:, None]).sum(axis=0)
    return np.concatenate([signed, magnitude, [float(failure_rate)]])


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def nearest_runs(query, library, k: int = 3):
    """Return [(index, similarity), ...] of the k most similar library embeddings."""
    sims = [(i, cosine_similarity(query, emb)) for i, emb in enumerate(library)]
    sims.sort(key=lambda t: t[1], reverse=True)
    return sims[:k]


def transfer_hot_regions(neighbor_families, space, top: int = 8):
    """Warm-start seeds for a new search: the highest-severity representative thetas from
    a set of neighbor runs' families, as raw theta dicts (feed to a search's seeding)."""
    ranked = sorted(neighbor_families, key=lambda f: f.mean_failure_score, reverse=True)
    return [dict(f.triggering_parameters) for f in ranked[:top]]
