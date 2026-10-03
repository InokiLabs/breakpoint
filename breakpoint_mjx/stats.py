"""Workstream 1b (T2): statistical rigor for BreakPoint reports.

Pure numpy + stdlib, fully CPU-unit-tested and ALWAYS-ON (cheap): every failure rate
the report quotes gets a confidence interval, and families are ranked by
``severity x probability`` (expected risk) so a rare-but-catastrophic fall is not
buried under a common-but-mild stumble.

Two proportion tools:
  * ``wilson_interval`` -- the Wilson *score* interval for a binomial proportion. Better
    than the naive normal (Wald) interval near 0/1 and for small n, and it never leaves
    ``[0, 1]``. Used for both the overall failure rate and each family's member-fraction.
  * ``bootstrap_ci``   -- a percentile bootstrap CI for a non-proportion metric (e.g. the
    mean failure_score of a family's members), for when a proportion CI does not fit.

Design notes:
  * All CIs degrade cleanly: ``n == 0`` -> ``(0.0, 1.0)`` (no data, maximal uncertainty)
    rather than dividing by zero.
  * ``severity`` for the ``severity x probability`` product is the family's *continuous*
    ``mean_failure_score`` (monotone + discriminative); the coarse label from
    ``transfer.severity_from_score`` is reported alongside for humans.
  * Nothing here depends on validity/margins, so the display is purely additive: a report
    with neither still gets failure-rate CIs + the sev x prob ranking.
"""
from __future__ import annotations

import math

import numpy as np

from breakpoint_mjx.transfer import severity_from_score


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns ``(lo, hi)`` in ``[0, 1]``.

    ``z`` is the standard-normal quantile (1.96 -> 95% two-sided). With ``n == 0`` there is
    no evidence, so the honest interval is the whole unit range ``(0.0, 1.0)`` -- this also
    avoids the div-by-zero the closed form would otherwise hit. ``successes`` is clamped to
    ``[0, n]`` defensively.
    """
    n = int(n)
    k = int(successes)
    if n <= 0:
        return (0.0, 1.0)
    k = max(0, min(k, n))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (float(lo), float(hi))


def proportion_ci(successes: int, n: int, z: float = 1.96) -> dict:
    """A proportion + its Wilson interval + the raw counts, as a JSON-safe dict.

    ``rate`` is ``successes / n`` (0.0 when ``n == 0``); ``lo``/``hi`` are the Wilson bounds.
    """
    lo, hi = wilson_interval(successes, n, z)
    rate = (int(successes) / int(n)) if n > 0 else 0.0
    return {
        "rate": float(rate),
        "lo": lo,
        "hi": hi,
        "successes": int(successes),
        "n": int(n),
        "z": float(z),
    }


def failure_rate(scores, threshold: float, z: float = 1.96) -> dict:
    """Failure rate over raw failure_scores: fraction ``>= threshold`` with a Wilson CI.

    ``rate = |{s : s >= threshold}| / |scores|`` -- the same ``>=`` rule the search uses to
    call an env a failure (see ``crossval.reproduces``). Returns a ``proportion_ci`` dict.
    """
    arr = np.asarray(list(scores), dtype=float)
    n = int(arr.size)
    k = int(np.count_nonzero(arr >= float(threshold))) if n else 0
    return proportion_ci(k, n, z)


def bootstrap_ci(samples, statistic=None, n_resamples: int = 2000,
                 ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for an arbitrary ``statistic`` (default: the mean).

    Resamples ``samples`` with replacement ``n_resamples`` times, computes ``statistic`` on
    each resample, and returns the central ``ci`` percentile band. Deterministic given
    ``seed``. Empty input -> ``(0.0, 0.0)``; a single unique value -> a degenerate band at
    that value.
    """
    arr = np.asarray(list(samples), dtype=float)
    m = int(arr.size)
    if m == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(int(n_resamples), m))
    resamples = arr[idx]                       # (n_resamples, m)
    if statistic is None:
        boot = resamples.mean(axis=1)
    else:
        boot = np.array([statistic(row) for row in resamples], dtype=float)
    alpha = 1.0 - float(ci)
    lo = float(np.percentile(boot, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def family_probability(family, n_total: int, z: float = 1.96) -> dict:
    """A family's failure rate as its member-fraction of all evaluated envs, with Wilson CI.

    ``probability = |member_ids| / n_total`` -- the chance a random evaluated env lands in
    this failure family. ``n_total`` is the count of ALL evaluated envs (not just failures),
    so families are on the same probability scale as the overall failure rate.
    """
    return proportion_ci(len(family.member_ids), int(n_total), z)


def _family_stat_rows(families, n_total=None, z: float = 1.96) -> list[dict]:
    """Build the per-family stat dicts (unsorted) shared by both rankings.

    * ``probability`` = each family's member-fraction of ``n_total`` evaluated envs, with a
      Wilson CI. When ``n_total`` is None it defaults to the total member count across the
      given families, i.e. a *relative* member-fraction (still a valid proportion) -- this
      keeps the rankings self-contained when the caller has only the families.
    * ``severity``     = the coarse label from ``severity_from_score``.
    * ``severity_score`` = the family's continuous ``mean_failure_score`` (monotone + more
      discriminative than the 4-way label; used as the severity magnitude).
    * ``sev_prob``     = ``severity_score * probability`` (expected-value risk).
    """
    fams = list(families)
    if n_total is None:
        n_total = sum(len(f.member_ids) for f in fams)
    rows = []
    for f in fams:
        pc = family_probability(f, n_total, z) if n_total else proportion_ci(0, 0, z)
        sev_score = float(f.mean_failure_score)
        rows.append({
            "family_id": f.id,
            "name": f.name,
            "severity": severity_from_score(sev_score),
            "severity_score": sev_score,
            "probability": pc["rate"],
            "prob_lo": pc["lo"],
            "prob_hi": pc["hi"],
            "successes": pc["successes"],
            "n": pc["n"],
            "sev_prob": float(sev_score * pc["rate"]),
        })
    return rows


def severity_probability_rank(families, n_total=None, z: float = 1.96) -> list[dict]:
    """Rank families by ``severity x probability`` = EXPECTED-VALUE risk, highest first.

    This is the standard risk metric ``probability x severity`` (severity =
    ``mean_failure_score``). Being an expectation, it does NOT guarantee that a rare but
    catastrophic family outranks a common but mild one -- a frequent mild failure can carry
    more expected cost than an infrequent severe one. Use ``severity_primary_rank`` for the
    worst-case-first (catastrophic-first) view; the report surfaces both.

    Returns a list of JSON-safe dicts (see ``_family_stat_rows``) sorted by ``sev_prob``
    descending (ties broken by severity, then name for determinism).
    """
    rows = _family_stat_rows(families, n_total, z)
    rows.sort(key=lambda d: (d["sev_prob"], d["severity_score"], d["name"]), reverse=True)
    return rows


def severity_primary_rank(families, n_total=None, z: float = 1.96) -> list[dict]:
    """Rank families by SEVERITY first (worst-case first), probability as the tiebreak.

    The complement to ``severity_probability_rank``: this GUARANTEES a higher-severity
    family (a catastrophic fall) ranks ahead of a lower-severity one no matter how rare it
    is -- the ordering an operator asking "what's the worst that can happen?" wants -- which
    expected-value risk does not. Same per-family dict shape; sorted by ``severity_score``
    descending, then ``probability``, then name (determinism).
    """
    rows = _family_stat_rows(families, n_total, z)
    rows.sort(key=lambda d: (d["severity_score"], d["probability"], d["name"]), reverse=True)
    return rows


def sim_validity_counts(records) -> dict:
    """Count simulator-artifact rollouts (the physical-validity gate) among ``records``.

    ``records`` are FailureRecords (or dicts) whose ``component_signals`` may carry the
    ``sim_invalid`` / ``sim_invalid_reason`` keys written by ``failure.component_signals``.
    Records without them count as valid (older runs / backends without the gate), and
    ``gate_active`` reports whether ANY record carried the keys, so the report can say
    "gate not active" instead of implying 0 artifacts were checked.

    Returns ``{"n_evaluated", "n_sim_invalid", "n_valid", "reasons": {name: count},
    "gate_active"}``. A record can carry several reasons, so reason counts may sum to more
    than ``n_sim_invalid``.
    """
    from breakpoint_mjx.failure import SIM_INVALID_REASONS

    n = 0
    n_bad = 0
    active = False
    reasons = {name: 0 for _, name in SIM_INVALID_REASONS}
    for r in records or []:
        n += 1
        sig = (r.get("component_signals") if isinstance(r, dict)
               else getattr(r, "component_signals", None)) or {}
        if "sim_invalid" in sig:
            active = True
        if float(sig.get("sim_invalid", 0.0) or 0.0):
            n_bad += 1
            bits = int(float(sig.get("sim_invalid_reason", 0.0) or 0.0))
            for bit, name in SIM_INVALID_REASONS:
                if bits & bit:
                    reasons[name] += 1
    return {"n_evaluated": n, "n_sim_invalid": n_bad, "n_valid": n - n_bad,
            "reasons": reasons, "gate_active": active}


def compute_report_stats(families, n_total: int, n_failures: int,
                         threshold: float | None = None, z: float = 1.96,
                         coverage: dict | None = None, records=None) -> dict:
    """Bundle the always-on report statistics.

    Returns ``{"n_total", "threshold", "overall", "ranking", "severity_ranking"}`` where:
      * ``overall``          -- ``proportion_ci`` for the OVERALL failure rate
        (``n_failures / n_valid``, the fraction of physically valid evaluated envs that
        failed; ``n_valid == n_total`` when ``records`` is not given).
      * ``ranking``          -- ``severity_probability_rank`` (expected-value risk, headline).
      * ``severity_ranking`` -- ``severity_primary_rank`` (worst-case first, second view).

    ``threshold`` is accepted for provenance/labelling; the failure counts are passed in
    pre-computed (run_pipeline already partitions all_records at ``cfg.failure_threshold``).
    ``probability`` here is the true fraction of the valid evaluated envs, not the relative
    member-fraction the ``stats=None`` render fallback uses.

    ``records`` (optional) are ALL evaluated FailureRecords. When given, rollouts the
    physical-validity gate flagged as simulator artifacts (``component_signals
    ["sim_invalid"]``, see ``failure.component_signals``) are counted and EXCLUDED from the
    denominators, and ``out["validity"]`` / ``out["n_sim_invalid"]`` carry the counts
    (``sim_validity_counts``) for the report's "Physical validity" line. Flagged rollouts
    score 0, so they are already absent from ``n_failures`` and from every family.

    ``coverage`` (optional) is the campaign-coverage certificate from
    ``coverage.run_until_saturated`` (BP_COVERAGE=1); it is passed through unchanged under
    ``"coverage"`` so the report can render its one-line release-review sentence.
    """
    n_valid = int(n_total)
    validity = None
    if records is not None:
        validity = sim_validity_counts(records)
        n_valid = max(int(n_total) - validity["n_sim_invalid"], 0)
    out = {
        "n_total": int(n_total),
        "threshold": (float(threshold) if threshold is not None else None),
        "overall": proportion_ci(n_failures, n_valid, z),
        "ranking": severity_probability_rank(families, n_valid, z),
        "severity_ranking": severity_primary_rank(families, n_valid, z),
    }
    if validity is not None:
        out["n_sim_invalid"] = validity["n_sim_invalid"]
        out["n_valid"] = n_valid
        out["validity"] = validity
    if coverage is not None:
        out["coverage"] = coverage
    return out
