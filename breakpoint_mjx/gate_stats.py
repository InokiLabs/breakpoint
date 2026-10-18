"""Exact tests behind the CI gate's verdict (pure Python, no numpy or JAX).

The gate runs the baseline and the candidate on the same cells and the same seeds, so each
(cell, seed) is a PAIR of outcomes. A pair where exactly one checkpoint falls is
discordant: ``up`` when the candidate falls and the baseline survived, ``down`` the other
way round. Under "the candidate is no worse than the baseline", a discordant pair is at
most a fair coin flip, which gives an exact one-sided sign test (McNemar's test in its
exact form) that needs no normal approximation and is valid at n = 2.

The verdict combines two families, each at alpha / 2, so the build's false-alarm
probability stays at or below alpha whatever the number of cells:

  * pooled: one sign test over every discordant pair in the battery (most power when a
    regression is spread thin, or concentrated in one cell while the rest are quiet);
  * per cell: one sign test per cell, Holm-adjusted across cells (catches one badly
    regressed cell even when other cells improved and cancel it out of the pooled count).

When an evaluator only reports per-cell fall counts (no per-seed outcomes), the pairing is
unknown and each test falls back to the one-sided Fisher exact test on the counts, which
is valid but has less power. The report names the method used.
"""
from __future__ import annotations

import math

DEFAULT_ALPHA = 0.05


def binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p), computed in log space (stable for large n)."""
    k, n = int(k), int(n)
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + i * lp + (n - i) * lq for i in range(k, n + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def sign_test_p(up: int, down: int) -> float:
    """One-sided exact sign test: P(at least ``up`` of the ``up + down`` discordant pairs
    go against the candidate, if each pair were a fair coin). 1.0 when nothing is
    discordant (identical outcomes are never evidence of a regression)."""
    up, down = int(up), int(down)
    if up < 0 or down < 0:
        raise ValueError("discordant counts must be >= 0")
    return binom_sf(up, up + down, 0.5)


def fisher_one_sided(cand_falls: int, cand_n: int, base_falls: int, base_n: int) -> float:
    """One-sided Fisher exact test that the candidate's fall rate exceeds the baseline's:
    P(candidate falls >= observed | both margins), hypergeometric upper tail."""
    a, n1, b, n2 = int(cand_falls), int(cand_n), int(base_falls), int(base_n)
    if not (0 <= a <= n1 and 0 <= b <= n2):
        raise ValueError("falls must lie in [0, n]")
    k, n = a + b, n1 + n2
    if k == 0 or k == n:
        return 1.0

    def log_pmf(x):
        return (_log_comb(n1, x) + _log_comb(n2, k - x) - _log_comb(n, k))

    hi = min(n1, k)
    terms = [log_pmf(x) for x in range(a, hi + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def holm_adjust(pvals) -> list:
    """Holm step-down adjusted p-values (same order as the input). Controls the
    family-wise error rate under any dependence between the tests."""
    p = [float(x) for x in pvals]
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def min_detectable_up(n_pairs: int, alpha: float) -> int | None:
    """Smallest number of discordant 'candidate falls' pairs (with zero the other way)
    that a sign test over ``n_pairs`` pairs can call significant at ``alpha``; None when
    even a clean sweep cannot (too few seeds for that alpha)."""
    for up in range(1, int(n_pairs) + 1):
        if sign_test_p(up, 0) < alpha:
            return up
    return None


def cell_test(base: dict, cand: dict) -> dict:
    """Test one cell. ``base``/``cand`` carry ``falls``, ``n`` and optionally ``outcomes``
    (a per-seed 0/1 list in seed order). Returns the discordant counts (None when the
    pairing is unknown), the one-sided p-value and the method."""
    bo, co = base.get("outcomes"), cand.get("outcomes")
    if bo is not None and co is not None and len(bo) == len(co):
        up = sum(1 for b, c in zip(bo, co) if c and not b)
        down = sum(1 for b, c in zip(bo, co) if b and not c)
        return {"up": up, "down": down, "p": sign_test_p(up, down),
                "method": "paired-sign"}
    return {"up": None, "down": None,
            "p": fisher_one_sided(cand["falls"], cand["n"], base["falls"], base["n"]),
            "method": "fisher"}


def escalation_candidates(decision: dict, tolerance: float = 0.0) -> list:
    """Cells worth re-running on fresh seeds: the candidate fell more often but the first
    stage could not call it (too few seeds). Significant cells are already decided."""
    return [t["id"] for t in decision["cells"]
            if t["delta"] > tolerance + 1e-12 and not t["significant"]]


def two_stage(stage1_pairs, run_stage2, alpha: float = DEFAULT_ALPHA,
              tolerance: float = 0.0) -> dict:
    """Adaptive gate: decide on the battery's seeds, then re-run only the suspect cells on
    FRESH seeds and decide again.

    Each stage gets alpha/2. Stage 2 sees only new, independent seeds, so choosing which
    cells to escalate from stage-1 data does not bias it, and by the union bound the
    overall false-alarm probability is at most alpha. ``run_stage2(cell_ids)`` returns the
    stage-2 pairs (same shape as ``stage1_pairs``) for those cells; it is not called when
    nothing is suspect."""
    s1 = decide(stage1_pairs, alpha=alpha / 2.0, tolerance=tolerance)
    suspects = escalation_candidates(s1, tolerance)
    s2 = None
    if suspects:
        s2 = decide(run_stage2(suspects), alpha=alpha / 2.0, tolerance=tolerance)
    return {"stage1": s1, "stage2": s2, "escalated": suspects,
            "regressed": bool(s1["regressed"] or (s2 is not None and s2["regressed"]))}


def decide(pairs, alpha: float = DEFAULT_ALPHA, tolerance: float = 0.0) -> dict:
    """The gate verdict over all cells.

    ``pairs`` is a list of (cell_id, base_cell, cand_cell). A cell is flagged when its
    Holm-adjusted p is below alpha/2 and its fall-rate increase exceeds ``tolerance``; the
    battery is flagged when the pooled p is below alpha/2 and the pooled fall-rate
    increase exceeds ``tolerance``. ``regressed`` is either. Returns per-cell results in
    input order plus the pooled block."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    half = alpha / 2.0
    cells = []
    for cid, b, c in pairs:
        t = cell_test(b, c)
        n_b, n_c = max(int(b["n"]), 1), max(int(c["n"]), 1)
        t["id"] = cid
        t["delta"] = c["falls"] / n_c - b["falls"] / n_b
        cells.append(t)
    adj = holm_adjust([t["p"] for t in cells]) if cells else []
    for t, a in zip(cells, adj):
        t["p_holm"] = a
        t["significant"] = bool(a < half and t["delta"] > tolerance + 1e-12)

    paired = all(t["method"] == "paired-sign" for t in cells) and bool(cells)
    tot_b = sum(int(b["falls"]) for _, b, _ in pairs)
    tot_c = sum(int(c["falls"]) for _, _, c in pairs)
    n_b = sum(int(b["n"]) for _, b, _ in pairs)
    n_c = sum(int(c["n"]) for _, _, c in pairs)
    if paired:
        up = sum(t["up"] for t in cells)
        down = sum(t["down"] for t in cells)
        p_pool = sign_test_p(up, down)
    else:
        up = down = None
        p_pool = fisher_one_sided(tot_c, n_c, tot_b, n_b) if n_b and n_c else 1.0
    pooled_delta = (tot_c / n_c - tot_b / n_b) if n_b and n_c else 0.0
    pooled_sig = bool(p_pool < half and pooled_delta > tolerance + 1e-12)
    per_seed_pairs = (n_c // max(len(cells), 1)) if cells else 0
    return {
        "method": ("paired exact sign test (pooled + Holm per cell)" if paired else
                   "one-sided Fisher exact test (pooled + Holm per cell; per-seed "
                   "outcomes unavailable)"),
        "alpha": alpha,
        "cells": cells,
        "pooled": {"up": up, "down": down, "p": p_pool, "delta": pooled_delta,
                   "significant": pooled_sig},
        "regressed": pooled_sig or any(t["significant"] for t in cells),
        # what this run could have detected at all, so a pass is read for what it is
        "min_up_per_cell": min_detectable_up(per_seed_pairs, half / max(len(cells), 1)),
        "min_up_pooled": min_detectable_up(n_c, half),
    }
