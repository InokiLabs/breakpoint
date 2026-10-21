"""Benchmark metrics. Pure numpy; every function here is unit-tested in benchmark/tests.

Definitions are fixed in BENCHMARK_PLAN.md section 8.
"""
from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------- normalization
def box_normalize(X, lo, hi):
    """Rows of X mapped to the unit box of [lo, hi] (a zero-width dim maps to 0)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    lo = np.asarray(lo, dtype=float)
    span = np.asarray(hi, dtype=float) - lo
    span = np.where(span > 0, span, 1.0)
    return (X - lo) / span


# --------------------------------------------------------------------------- clustering
def leader_clusters(U, eps, order=None):
    """Greedy leader clustering in discovery order.

    ``U``: (N, D) failures in normalized coordinates, in the order they were found.
    A point joins the first existing leader within L-infinity distance ``eps``;
    otherwise it becomes a new leader. Returns (labels (N,), leader_indices list).
    Deterministic given the order, so 'the k-th distinct failure was found at rollout t'
    is well defined.
    """
    U = np.atleast_2d(np.asarray(U, dtype=float))
    n = U.shape[0] if U.size else 0
    idx = np.arange(n) if order is None else np.asarray(order)
    labels = np.full(n, -1, dtype=int)
    leaders: list[int] = []
    L = np.zeros((0, U.shape[1] if U.ndim == 2 and U.size else 0))
    for i in idx:
        if len(leaders):
            d = np.abs(L - U[i]).max(axis=1)
            j = int(np.argmin(d))
            if d[j] <= eps:
                labels[i] = j
                continue
        labels[i] = len(leaders)
        leaders.append(int(i))
        L = np.vstack([L, U[i][None, :]]) if L.size else U[i][None, :].copy()
    return labels, leaders


def discovery_curve(failure_mask, U_all, eps):
    """Cumulative distinct failure clusters after each rollout (length N).

    ``failure_mask``: (N,) bool over ALL evaluated rows in evaluation order.
    ``U_all``: (N, D) normalized coordinates of all rows."""
    failure_mask = np.asarray(failure_mask, dtype=bool)
    U_all = np.atleast_2d(np.asarray(U_all, dtype=float))
    curve = np.zeros(failure_mask.shape[0], dtype=int)
    L = None
    count = 0
    for t in range(failure_mask.shape[0]):
        if failure_mask[t]:
            u = U_all[t]
            if L is None or np.abs(L - u).max(axis=1).min() > eps:
                L = u[None, :].copy() if L is None else np.vstack([L, u])
                count += 1
        curve[t] = count
    return curve


def mode_count(modes) -> int:
    return len(set(modes))


# ---------------------------------------------------------------------------- intervals
def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Two-sided exact (1 - alpha) interval for a binomial proportion."""
    from scipy.stats import beta

    if n <= 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def cp_lower_one_sided(k: int, n: int, alpha: float = 0.05) -> float:
    from scipy.stats import beta

    if k <= 0 or n <= 0:
        return 0.0
    return float(beta.ppf(alpha, k, n - k + 1))


def is_validated(k: int, n: int, bar: float = 0.5, alpha: float = 0.05) -> bool:
    """Predefined validation rule: one-sided 95% CP lower bound >= bar."""
    return cp_lower_one_sided(k, n, alpha) >= bar


def min_k_to_validate(n: int, bar: float = 0.5, alpha: float = 0.05):
    for k in range(n + 1):
        if is_validated(k, n, bar, alpha):
            return k
    return None


# ------------------------------------------------------------------ candidate selection
def select_candidates(failure_order, cluster_leaders, first_k, n_leaders, seed):
    """Fixed selection rule (no outcome-dependent choice).

    ``failure_order``: row indices of failures in discovery order.
    ``cluster_leaders``: row indices of cluster leaders.
    Returns (first, leaders_sampled, all_leaders_validated: bool)."""
    first = list(map(int, failure_order[:first_k]))
    leaders = list(map(int, cluster_leaders))
    if len(leaders) <= n_leaders:
        return first, leaders, True
    rng = np.random.default_rng([int(seed), 31337])
    pick = sorted(rng.choice(len(leaders), size=n_leaders, replace=False).tolist())
    return first, [leaders[i] for i in pick], False


def time_to_first_validated(first_rows, validated: dict):
    """Rollouts spent (1-based row index) before the first in-order failure that
    validated, or None if none of the checked ones did."""
    for r in first_rows:
        if validated.get(int(r)):
            return int(r) + 1
    return None


def estimated_validated_distinct(n_clusters: int, k_valid: int, n_checked: int,
                                 exact: bool):
    """Point estimate and 95% CP-derived interval for distinct validated failures."""
    if n_checked == 0:
        return {"estimate": 0.0 if n_clusters == 0 else None, "lo": None, "hi": None,
                "exact": exact}
    if exact:
        return {"estimate": float(k_valid), "lo": float(k_valid), "hi": float(k_valid),
                "exact": True}
    p = k_valid / n_checked
    lo, hi = clopper_pearson(k_valid, n_checked)
    return {"estimate": n_clusters * p, "lo": n_clusters * lo, "hi": n_clusters * hi,
            "exact": False}


# -------------------------------------------------------------------- across repetitions
def summarize_reps(values):
    v = np.asarray([x for x in values if x is not None and not
                    (isinstance(x, float) and math.isnan(x))], dtype=float)
    if v.size == 0:
        return {"mean": None, "min": None, "max": None, "sd": None, "n": 0}
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
            "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}


def bootstrap_ratio(a, b, n_boot=10000, seed=0):
    """Ratio of means mean(a)/mean(b) with a percentile bootstrap 95% interval over
    repetitions (resampling a and b independently). None when b's mean is 0."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0 or b.mean() == 0:
        return {"ratio": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a.size, (n_boot, a.size))
    ib = rng.integers(0, b.size, (n_boot, b.size))
    mb = b[ib].mean(axis=1)
    ok = mb > 0
    r = a[ia].mean(axis=1)[ok] / mb[ok]
    return {"ratio": float(a.mean() / b.mean()),
            "lo": float(np.percentile(r, 2.5)) if r.size else None,
            "hi": float(np.percentile(r, 97.5)) if r.size else None}


def paired_fall_difference(base_falls, cand_falls):
    """Candidate minus baseline fall rate on paired rollouts, with a Newcombe-style
    interval from the per-pair differences (normal approximation on the paired mean;
    reported alongside the gate's exact sign test, never instead of it)."""
    a = np.asarray(base_falls, dtype=float)
    b = np.asarray(cand_falls, dtype=float)
    if a.size == 0 or a.shape != b.shape:
        return {"diff": None, "lo": None, "hi": None, "n": int(a.size)}
    d = b - a
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else 0.0
    return {"diff": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se, "n": int(d.size)}
