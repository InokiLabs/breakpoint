"""Exact tests behind the gate verdict (breakpoint_mjx/gate_stats.py). Pure Python."""
from __future__ import annotations

import itertools
import math

import pytest

from breakpoint_mjx import gate_stats as S


def _brute_binom_sf(k, n, p=0.5):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


@pytest.mark.parametrize("n", [0, 1, 2, 8, 30, 120])
def test_binom_sf_matches_brute_force(n):
    for k in range(0, n + 2):
        assert S.binom_sf(k, n) == pytest.approx(_brute_binom_sf(k, n) if k <= n else 0.0,
                                                 abs=1e-12)


def test_binom_sf_large_n_does_not_overflow():
    # mincause's float version overflowed past n ~ 1030; this one is log-space
    assert S.binom_sf(1500, 3000) == pytest.approx(0.5, abs=0.01)
    assert S.binom_sf(1000, 1000) > 0.0              # 0.5^1000 ~ 1e-301, still a float


def test_sign_test_known_values():
    assert S.sign_test_p(0, 0) == 1.0
    assert S.sign_test_p(8, 0) == pytest.approx(0.5 ** 8)
    assert S.sign_test_p(3, 0) == pytest.approx(0.125)
    assert S.sign_test_p(0, 5) == 1.0
    assert S.sign_test_p(2, 2) == pytest.approx(11 / 16)


def test_fisher_matches_brute_force_hypergeometric():
    def brute(a, n1, b, n2):
        k = a + b
        tot = math.comb(n1 + n2, k)
        return sum(math.comb(n1, x) * math.comb(n2, k - x)
                   for x in range(a, min(n1, k) + 1)) / tot

    for a, b in itertools.product(range(9), range(9)):
        if 0 < a + b < 16:
            assert S.fisher_one_sided(a, 8, b, 8) == pytest.approx(brute(a, 8, b, 8))
    assert S.fisher_one_sided(0, 8, 0, 8) == 1.0
    assert S.fisher_one_sided(8, 8, 8, 8) == 1.0


def test_holm_adjust():
    assert S.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert S.holm_adjust([]) == []
    assert S.holm_adjust([0.9, 0.9]) == [1.0, 1.0]


def _cell(outcomes):
    return {"falls": sum(outcomes), "n": len(outcomes), "outcomes": list(outcomes)}


def test_identical_outcomes_never_regress():
    pairs = [(f"c{i}", _cell([1, 0, 1, 0]), _cell([1, 0, 1, 0])) for i in range(15)]
    d = S.decide(pairs)
    assert d["regressed"] is False
    assert d["pooled"]["p"] == 1.0


def test_one_catastrophic_cell_is_caught_by_the_pooled_test():
    # 0/8 -> 8/8 in one cell: p = 0.0039 per cell, which Holm across 15 cells cannot
    # reach (needs < 0.025/15), but the pooled sign test (8 up, 0 down) does
    pairs = [(f"c{i}", _cell([0] * 8), _cell([0] * 8)) for i in range(14)]
    pairs.append(("bad", _cell([0] * 8), _cell([1] * 8)))
    d = S.decide(pairs)
    assert d["pooled"]["significant"] is True
    assert d["regressed"] is True


def test_regression_masked_in_pool_is_caught_per_cell():
    # one cell regresses hard while improvements elsewhere cancel the pooled count
    n = 40
    pairs = [("bad", _cell([0] * n), _cell([1] * 20 + [0] * 20))]
    pairs += [(f"good{i}", _cell([1] * 10 + [0] * 30), _cell([0] * n)) for i in range(2)]
    d = S.decide(pairs)
    assert d["pooled"]["significant"] is False
    bad = next(t for t in d["cells"] if t["id"] == "bad")
    assert bad["significant"] is True and d["regressed"] is True


def test_baseline_already_failing_cell_getting_worse_is_tested():
    # the old rule only looked at cells the baseline PASSED; 2/8 -> 8/8 used to be
    # "fails (baseline too)" and passed the build
    pairs = [("slope", _cell([1, 1, 0, 0, 0, 0, 0, 0]), _cell([1] * 8))]
    d = S.decide(pairs)
    assert d["cells"][0]["up"] == 6 and d["regressed"] is True


def test_small_noise_does_not_regress():
    # 1 more fall out of 8 in two cells: noise, not evidence
    pairs = [(f"c{i}", _cell([0] * 8), _cell([0] * 8)) for i in range(13)]
    pairs += [("a", _cell([0] * 8), _cell([1] + [0] * 7)),
              ("b", _cell([0] * 8), _cell([0, 1] + [0] * 6))]
    assert S.decide(pairs)["regressed"] is False


def test_tolerance_filters_practically_small_increases():
    pairs = [("c", _cell([0] * 40), _cell([1] * 6 + [0] * 34))]   # +15 pp, p = 1/64
    assert S.decide(pairs, tolerance=0.0)["regressed"] is True
    assert S.decide(pairs, tolerance=0.2)["regressed"] is False


def test_unpaired_counts_fall_back_to_fisher():
    pairs = [("c", {"falls": 0, "n": 8}, {"falls": 8, "n": 8})]
    d = S.decide(pairs)
    assert d["cells"][0]["method"] == "fisher"
    assert "Fisher" in d["method"]
    assert d["regressed"] is True


def test_false_alarm_rate_is_controlled_under_the_null():
    """Exact enumeration: two policies with the same independent per-seed fall rate p.
    The family-wise false-alarm probability of decide() must stay <= alpha. We check it
    by simulation with a fixed RNG over many batteries (15 cells, 8 seeds)."""
    import random

    rng = random.Random(0)
    alpha, trials, alarms = 0.05, 2000, 0
    for _ in range(trials):
        pairs = []
        for i in range(15):
            p = 0.1 if i % 2 else 0.3
            b = [int(rng.random() < p) for _ in range(8)]
            c = [int(rng.random() < p) for _ in range(8)]
            pairs.append((f"c{i}", _cell(b), _cell(c)))
        alarms += S.decide(pairs, alpha=alpha)["regressed"]
    assert alarms / trials <= alpha


def test_min_detectable():
    assert S.min_detectable_up(8, 0.025) == 6        # 0.5^6 = 0.0156 < 0.025
    assert S.min_detectable_up(2, 0.025) is None     # 0.25 is the best two seeds can do
    assert S.min_detectable_up(0, 0.05) is None


def test_bad_arguments():
    with pytest.raises(ValueError):
        S.decide([], alpha=0.0)
    with pytest.raises(ValueError):
        S.decide([], tolerance=-0.1)
    with pytest.raises(ValueError):
        S.sign_test_p(-1, 0)


def test_two_stage_false_alarm_rate_is_controlled_under_the_null():
    """Escalation picks cells from stage-1 data; stage 2 must still keep the build's
    false-alarm probability <= alpha (fresh, independent seeds + alpha/2 per stage)."""
    import random

    rng = random.Random(1)
    alpha, trials, alarms, escalations = 0.05, 1500, 0, 0

    def draw(p, n):
        return [int(rng.random() < p) for _ in range(n)]

    rates = [0.0, 0.02, 0.1, 0.2, 0.4] * 3
    for _ in range(trials):
        s1 = [(str(i), _cell(draw(p, 8)), _cell(draw(p, 8))) for i, p in enumerate(rates)]

        def stage2(ids):
            return [(i, _cell(draw(rates[int(i)], 32)), _cell(draw(rates[int(i)], 32)))
                    for i in ids]

        res = S.two_stage(s1, stage2, alpha=alpha)
        alarms += res["regressed"]
        escalations += bool(res["escalated"])
    assert escalations > trials // 2          # the null escalates often, and still...
    assert alarms / trials <= alpha           # ...stays under the false-alarm budget


def test_two_stage_skips_stage2_when_nothing_is_suspect():
    called = []
    res = S.two_stage([("a", _cell([0] * 8), _cell([0] * 8))],
                      lambda ids: called.append(ids) or [])
    assert called == [] and res["stage2"] is None and res["regressed"] is False


# ------------------------------------------------------------------ degradation_test

def test_degradation_identical_errors_are_never_evidence():
    from breakpoint_mjx.gate_stats import degradation_test
    d = degradation_test([(0.2, 0.2)] * 40)
    assert (d["up"], d["down"], d["p"], d["significant"]) == (0, 0, 1.0, False)


def test_degradation_deadband_is_symmetric_in_the_ratio():
    from breakpoint_mjx.gate_stats import degradation_test
    # 1.05x worse is inside a 10% deadband; 1.2x worse / 1.2x better both count
    d = degradation_test([(1.0, 1.05), (1.0, 1.2), (1.2, 1.0)], margin=0.10)
    assert (d["up"], d["down"]) == (1, 1)


def test_degradation_consistently_worse_is_significant():
    from breakpoint_mjx.gate_stats import degradation_test
    d = degradation_test([(0.2, 0.3)] * 12 + [(0.3, 0.2)] * 1, alpha=0.05)
    assert d["up"] == 12 and d["down"] == 1 and d["significant"] is True
    assert d["median_ratio"] == pytest.approx(1.5)
