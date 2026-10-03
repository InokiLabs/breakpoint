"""CPU unit tests for T2 statistical rigor (breakpoint_mjx/stats.py).

Pure numpy: Wilson interval known values + n=0 edge, failure-rate + proportion CIs, the
bootstrap CI, and the severity x probability ranking (member-fraction probability with CI).
"""
from __future__ import annotations

import numpy as np
import pytest

from breakpoint_mjx.schemas import FailureFamily
from breakpoint_mjx.stats import (
    bootstrap_ci,
    compute_report_stats,
    failure_rate,
    family_probability,
    proportion_ci,
    severity_primary_rank,
    severity_probability_rank,
    wilson_interval,
)


# --------------------------------------------------------------------------- #
# Wilson interval: known reference values + edges
# --------------------------------------------------------------------------- #
def test_wilson_known_values_95pct():
    # 5/10 at 95% -> [0.2366, 0.7634] (standard reference).
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=5e-4)
    assert hi == pytest.approx(0.7634, abs=5e-4)
    # 1/1 at 95% -> [0.2065, 1.0]; the upper bound clamps to 1.
    lo1, hi1 = wilson_interval(1, 1)
    assert lo1 == pytest.approx(0.2065, abs=5e-4)
    assert hi1 == pytest.approx(1.0, abs=1e-9)


def test_wilson_stays_in_unit_interval_at_extremes():
    lo0, hi0 = wilson_interval(0, 20)      # zero successes
    assert lo0 == pytest.approx(0.0, abs=1e-9)   # Wilson lo at 0 successes is 0 (clamped)
    assert 0.0 <= lo0 <= hi0 <= 1.0
    loN, hiN = wilson_interval(20, 20)     # all successes
    assert hiN == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= loN <= hiN <= 1.0


def test_wilson_n_zero_is_maximally_uncertain():
    """n=0 has no evidence -> the honest interval is the whole unit range (no div-by-zero)."""
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(3, 0) == (0.0, 1.0)   # defensive: garbage successes, still safe


def test_wilson_wider_z_gives_wider_interval():
    lo95, hi95 = wilson_interval(5, 10, z=1.96)
    lo99, hi99 = wilson_interval(5, 10, z=2.576)
    assert lo99 < lo95 and hi99 > hi95


# --------------------------------------------------------------------------- #
# proportion_ci + failure_rate
# --------------------------------------------------------------------------- #
def test_proportion_ci_reports_rate_and_counts():
    d = proportion_ci(3, 12)
    assert d["rate"] == pytest.approx(0.25)
    assert d["successes"] == 3 and d["n"] == 12
    assert d["lo"] <= d["rate"] <= d["hi"]


def test_proportion_ci_n_zero():
    d = proportion_ci(0, 0)
    assert d["rate"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 1.0


def test_failure_rate_counts_at_or_above_threshold():
    scores = [0.9, 0.4, 0.55, 0.2, 0.5]           # >= 0.5 : {0.9, 0.55, 0.5} = 3
    d = failure_rate(scores, threshold=0.5)
    assert d["successes"] == 3 and d["n"] == 5
    assert d["rate"] == pytest.approx(3 / 5)
    assert d["lo"] <= 0.6 <= d["hi"]


def test_failure_rate_empty():
    d = failure_rate([], threshold=0.5)
    assert d["n"] == 0 and d["rate"] == 0.0 and d["hi"] == 1.0


# --------------------------------------------------------------------------- #
# bootstrap CI
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_constant_samples_is_degenerate():
    lo, hi = bootstrap_ci([2.0, 2.0, 2.0, 2.0])
    assert lo == pytest.approx(2.0) and hi == pytest.approx(2.0)


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    rng = np.random.default_rng(0)
    samples = rng.normal(5.0, 1.0, size=200)
    lo, hi = bootstrap_ci(samples, seed=123)
    assert lo < samples.mean() < hi
    # deterministic given the seed
    assert (lo, hi) == bootstrap_ci(samples, seed=123)


def test_bootstrap_ci_empty():
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_bootstrap_ci_custom_statistic():
    lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], statistic=np.max, seed=1)
    assert 1.0 <= lo <= hi <= 5.0


# --------------------------------------------------------------------------- #
# severity x probability ranking
# --------------------------------------------------------------------------- #
def _fam(fid, rank, mean_score, n_members):
    return FailureFamily(
        id=fid, name=f"fam {fid}", member_ids=[f"{fid}-{i}" for i in range(n_members)],
        representative_id=f"{fid}-0", triggering_parameters={"friction_floor": 0.3},
        mean_failure_score=mean_score, mean_transfer_score=0.6, priority_rank=rank,
    )


def test_family_probability_is_member_fraction_with_ci():
    f = _fam("f", 1, 0.9, n_members=5)
    pc = family_probability(f, n_total=100)
    assert pc["rate"] == pytest.approx(0.05)
    assert pc["successes"] == 5 and pc["n"] == 100
    assert pc["lo"] <= 0.05 <= pc["hi"]


def test_sev_prob_lets_severe_outrank_more_common_but_milder():
    """The point of severity x probability: a MORE severe family can outrank a MORE common
    but milder one -- i.e. it is not pure-frequency ranking."""
    severe = _fam("severe", rank=2, mean_score=1.0, n_members=30)   # prob 0.30, sev*prob 0.30
    common = _fam("common", rank=1, mean_score=0.5, n_members=40)   # prob 0.40, sev*prob 0.20
    ranking = severity_probability_rank([common, severe], n_total=100)
    # common has MORE members (would win on pure probability) ...
    assert len(common.member_ids) > len(severe.member_ids)
    # ... but the severe family wins on severity x probability.
    assert ranking[0]["family_id"] == "severe"
    assert ranking[0]["sev_prob"] == pytest.approx(0.30)
    assert ranking[1]["sev_prob"] == pytest.approx(0.20)
    # probabilities are member-fractions with CIs.
    assert ranking[0]["probability"] == pytest.approx(0.30)
    assert ranking[0]["prob_lo"] <= 0.30 <= ranking[0]["prob_hi"]


def test_sev_prob_rank_relative_when_n_total_absent():
    """Without n_total, probability is the relative member fraction across the families."""
    a = _fam("a", 1, 0.8, n_members=3)
    b = _fam("b", 2, 0.8, n_members=1)
    ranking = severity_probability_rank([a, b])   # n_total defaults to 3+1 = 4
    by_id = {d["family_id"]: d for d in ranking}
    assert by_id["a"]["probability"] == pytest.approx(3 / 4)
    assert by_id["b"]["probability"] == pytest.approx(1 / 4)
    # equal severity -> the one with more members ranks first
    assert ranking[0]["family_id"] == "a"


def test_sev_prob_rank_empty_families():
    assert severity_probability_rank([]) == []


def test_severity_primary_rank_puts_worst_case_first():
    """severity_primary_rank guarantees a rare-but-catastrophic family ranks first BY
    SEVERITY, even though expected-value risk (sev x prob) ranks the common mild one first."""
    severe_rare = _fam("severe", rank=2, mean_score=0.95, n_members=2)   # prob 0.02
    mild_common = _fam("mild", rank=1, mean_score=0.50, n_members=40)    # prob 0.40
    fams = [mild_common, severe_rare]
    # Expected-value risk: the common mild family wins (0.5*0.40 = 0.20 > 0.95*0.02 = 0.019).
    ev = severity_probability_rank(fams, n_total=100)
    assert ev[0]["family_id"] == "mild"
    # Worst-case-first: the rare catastrophic family wins on severity regardless of rarity.
    sev = severity_primary_rank(fams, n_total=100)
    assert sev[0]["family_id"] == "severe"
    assert sev[0]["severity_score"] == pytest.approx(0.95)


def test_severity_primary_rank_probability_tiebreak():
    a = _fam("a", 1, mean_score=0.8, n_members=5)
    b = _fam("b", 2, mean_score=0.8, n_members=2)   # same severity, fewer members
    sev = severity_primary_rank([b, a], n_total=100)
    assert [d["family_id"] for d in sev] == ["a", "b"]   # equal severity -> more common first


# --------------------------------------------------------------------------- #
# compute_report_stats bundle
# --------------------------------------------------------------------------- #
def test_compute_report_stats_bundles_overall_and_ranking():
    fams = [_fam("f0", 1, 0.9, 4), _fam("f1", 2, 0.6, 6)]
    stats = compute_report_stats(fams, n_total=50, n_failures=10, threshold=0.5)
    assert stats["n_total"] == 50
    assert stats["overall"]["rate"] == pytest.approx(10 / 50)
    assert stats["overall"]["lo"] <= 0.2 <= stats["overall"]["hi"]
    assert {d["family_id"] for d in stats["ranking"]} == {"f0", "f1"}
    # ranking probabilities are fractions of n_total (not just the failures)
    by_id = {d["family_id"]: d for d in stats["ranking"]}
    assert by_id["f0"]["probability"] == pytest.approx(4 / 50)
    assert by_id["f1"]["probability"] == pytest.approx(6 / 50)
    # both views are bundled; the severity view is worst-case first (f0 has higher severity)
    assert {d["family_id"] for d in stats["severity_ranking"]} == {"f0", "f1"}
    assert stats["severity_ranking"][0]["family_id"] == "f0"
