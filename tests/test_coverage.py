"""CPU tests for breakpoint_mjx.coverage: exact arithmetic on hand-built inputs, the
segment-connectivity mode definition on a two-ball toy, and the stopping rule on fake
searches. No physics runs here."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from breakpoint_mjx import coverage as cov


# ---------------------------------------------------------------- Chao2 / Chao1 / BRK

def test_chao2_closed_form_with_doubletons():
    # counts [1,1,2,3], m=10: S=4 Q1=2 Q2=1, (m-1)/m=0.9 -> 4 + 0.9*4/2 = 5.8
    r = cov.chao2([1, 1, 2, 3], m=10)
    assert (r["S_obs"], r["Q1"], r["Q2"], r["U"]) == (4, 2, 1, 7)
    assert r["estimate"] == pytest.approx(5.8)
    # Chao & Jost deficit: (2/7) * (9*2) / (9*2 + 2*1) = 0.257142...
    assert r["new_mode_prob"] == pytest.approx((2 / 7) * 18 / 20)


def test_chao2_bias_corrected_branch_when_no_doubletons():
    # counts [1,1,1], m=5: Q2=0 -> 3 + 0.8*3*2/2 = 5.4
    r = cov.chao2([1, 1, 1], m=5)
    assert r["Q2"] == 0
    assert r["estimate"] == pytest.approx(5.4)


def test_chao2_no_singletons_is_saturated_and_degenerate_ci():
    r = cov.chao2([2, 3, 0], m=6)
    assert r["S_obs"] == 2 and r["estimate"] == 2.0
    assert r["ci"] == [2.0, 2.0] and r["new_mode_prob"] == 0.0


@pytest.mark.parametrize("counts,m", [([1, 1, 2, 3], 10), ([1, 1, 1], 5),
                                      ([1] * 6 + [2, 5], 30), ([1, 4, 4], 12)])
def test_chao2_ci_lower_never_below_observed(counts, m):
    r = cov.chao2(counts, m)
    assert r["ci"][0] >= r["S_obs"] - 1e-12
    assert r["ci"][0] <= r["estimate"] <= r["ci"][1] + 1e-12


def test_chao1_bias_corrected():
    # f=[1,1,2]: n=4 S=3 f1=2 f2=1 -> 3 + (3/4)*2*1/(2*2) = 3.375
    assert cov.chao1_bc([1, 1, 2])["estimate"] == pytest.approx(3.375)


def test_boender_rinnooy_kan_closed_form_and_rule():
    r = cov.boender_rinnooy_kan(2, 10)          # 2*9/6 = 3.0 -> keep going
    assert r["w_est"] == pytest.approx(3.0) and r["stop"] is False
    assert r["unseen_basin_mass"] == pytest.approx(6 / 90)
    r = cov.boender_rinnooy_kan(2, 40)          # 2*39/36 = 2.1667 -> stop
    assert r["stop"] is True
    assert cov.boender_rinnooy_kan(3, 5)["stop"] is False   # N < w + 3: undefined


# ---------------------------------------------------------------- Clopper-Pearson

def test_clopper_pearson_rule_of_three_arithmetic():
    # k=0: upper = 1 - alpha^(1/n) exactly
    assert cov.clopper_pearson_upper(0, 2048) == pytest.approx(1 - 0.05 ** (1 / 2048),
                                                               rel=1e-6)
    assert cov.clopper_pearson_upper(5, 5) == 1.0


@pytest.mark.parametrize("k,n", [(0, 10), (1, 10), (3, 100), (40, 2048), (900, 2048)])
def test_clopper_pearson_matches_scipy_beta(k, n):
    beta = pytest.importorskip("scipy.stats").beta
    assert cov.clopper_pearson_upper(k, n) == pytest.approx(
        beta.ppf(0.95, k + 1, n - k), rel=1e-6)
    lo, hi = cov.clopper_pearson_interval(k, n)
    assert hi == pytest.approx(beta.ppf(0.975, k + 1, n - k), rel=1e-6)
    if k:
        assert lo == pytest.approx(beta.ppf(0.025, k, n - k + 1), rel=1e-6)


# ---------------------------------------------------------------- units / incumbents

def test_restart_unit_parsing():
    assert cov.restart_unit("cmaes-r3-40") == (0, 3)
    assert cov.restart_unit("cmaes-w2-r11-9") == (2, 11)
    assert cov.restart_unit("random-4") is None
    assert cov.restart_unit(None) is None


def _rec(rid, theta, score):
    return SimpleNamespace(id=rid, theta_vector=list(theta), failure_score=float(score))


def test_incumbents_exclude_restart0_and_nonfailing_units():
    recs = [_rec("cmaes-r0-0", [0, 0], 0.9), _rec("cmaes-r1-1", [1, 1], 0.2),
            _rec("cmaes-r1-2", [2, 2], 0.7), _rec("cmaes-r2-3", [3, 3], 0.4),
            _rec("random-1", [9, 9], 1.0)]
    units, inc = cov.restart_incumbents(recs, 0.5)
    assert units == [(0, 1), (0, 2)]           # restart 0 and random excluded
    assert list(inc) == [(0, 1)] and inc[(0, 1)].theta_vector == [2, 2]
    units0, _ = cov.restart_incumbents(recs, 0.5, exclude_restart0=False)
    assert (0, 0) in units0


# ---------------------------------------------------------------- mode graph + audit

CENTRES = np.array([[0.2, 0.2], [0.8, 0.8]])


def two_ball_score(thetas):
    d = np.linalg.norm(np.atleast_2d(thetas)[:, None, :] - CENTRES[None], axis=2).min(1)
    return np.where(d < 0.15, 1.0, 0.0)


class Box:
    def names(self):
        return ["a", "b"]

    def lower(self):
        return [0.0, 0.0]

    def upper(self):
        return [1.0, 1.0]


def test_segments_split_disjoint_balls_and_join_within_one():
    ok = cov.segments_connected(two_ball_score, [[0.15, 0.2], [0.2, 0.2]],
                                [[0.25, 0.22], [0.8, 0.8]], 0.5)
    assert ok.tolist() == [True, False]


def test_mode_graph_components_are_order_independent_and_cached():
    g = cov.ModeGraph(two_ball_score, 0.5)
    for k, p in [("a", [0.8, 0.8]), ("b", [0.2, 0.25]), ("c", [0.85, 0.8]),
                 ("d", [0.2, 0.15])]:
        g.add(k, p)
    lab = g.labels()
    assert lab["a"] == lab["c"] and lab["b"] == lab["d"] and lab["a"] != lab["b"]
    before = g.probe_evals
    g.labels()
    assert g.probe_evals == before              # no re-evaluation of known pairs


def test_audit_counts_failures_in_unfound_mode_as_unseen():
    g = cov.ModeGraph(two_ball_score, 0.5)
    g.add("a", [0.2, 0.2])                      # only the lower-left mode was found
    labels = g.labels()
    th = np.array([[0.21, 0.19], [0.79, 0.81], [0.5, 0.5], [0.81, 0.8]])
    a = cov.residual_audit(th, two_ball_score(th), g, labels, 0.5, space=Box())
    assert (a["n"], a["k_fail"], a["k_unseen"]) == (4, 3, 2)
    assert a["unseen_index"] == [1, 3]
    assert a["cp_upper"] == pytest.approx(cov.clopper_pearson_upper(2, 4))


# ---------------------------------------------------------------- stopping rule

def _fake_search(mode_of_restart):
    """A search whose restart r lands its incumbent on mode_of_restart(wave_seed, r)."""
    def search(evaluate, space, cfg, key, restarts=16, seed_theta=None, objective=None):
        recs = []
        for r in range(restarts):
            c = mode_of_restart(cfg.seed, r)
            recs.append(_rec(f"cmaes-r{r}-{2 * r}", c, 1.0))
            recs.append(_rec(f"cmaes-r{r}-{2 * r + 1}", [0.5, 0.05], 0.0))
        return recs
    return search


CFG = SimpleNamespace(seed=0, failure_threshold=0.5, popsize=4, n_eval=64)


def _run(search, max_waves=4, rule="chao2"):
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class C:
        seed: int = 0
        failure_threshold: float = 0.5
        popsize: int = 4
        n_eval: int = 64
    return cov.run_until_saturated(None, Box(), C(), None, wave=16, max_waves=max_waves,
                                   rule=rule, score_fn=two_ball_score, search_fn=search,
                                   n_audit=64)


def test_stopping_rule_halts_on_saturated_search():
    recs, c = _run(_fake_search(lambda s, r: CENTRES[r % 2]))
    assert c["stopped_reason"] == "saturated (chao2)" and c["waves"] == 1
    assert c["S_obs"] == 2 and c["chao2"] == 2.0
    assert len({r.id for r in recs}) == len(recs)          # ids unique after re-tag
    assert c["audit"]["n"] == 64 and c["audit"]["k_unseen"] == 0


def test_default_rule_none_never_stops_early_and_makes_no_completeness_claim():
    _, c = _run(_fake_search(lambda s, r: CENTRES[r % 2]), max_waves=3, rule="none")
    assert c["waves"] == 3 and c["stopped_reason"] == "fixed budget, no stop rule"
    assert c["history"][0]["stop"]["chao2"] is True      # would have fired; ignored
    s = cov.coverage_sentence(c)
    assert s.startswith("Uniform audit of 64 random worlds")
    assert "not a completeness claim" in s


def test_stopping_rule_keeps_going_when_every_restart_is_new():
    # every restart lands on its own disjoint mode: all singletons, never saturated
    def far(seed, r):
        k = seed // 100003 * 16 + r
        return [0.02 + 0.03 * (k % 30), 0.97 - 0.03 * (k // 30)]

    def score(th):
        th = np.atleast_2d(th)
        return np.zeros(th.shape[0])            # segments between them never fail

    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class C:
        seed: int = 0
        failure_threshold: float = 0.5
        popsize: int = 4
        n_eval: int = 64
    _, c = cov.run_until_saturated(None, Box(), C(), None, wave=16, max_waves=3,
                                   rule="chao2", score_fn=score, search_fn=_fake_search(far), n_audit=0)
    assert c["stopped_reason"] == "max_waves (stop rule did not fire)" and c["waves"] == 3
    assert c["S_obs"] == 45 and c["Q1"] == 45 and c["chao2"] > 45
    assert c["history"][-1]["stop"] == {"chao2": False, "brk": False, "chao1": False}


def test_coverage_sentence_carries_scope():
    _, c = _run(_fake_search(lambda s, r: CENTRES[r % 2]))
    s = cov.coverage_sentence(c)
    assert "Clopper-Pearson" in s and "declared DR box" in s and "simulation only" in s
    assert cov.coverage_sentence(None) == ""


# ---------------------------------------------------------------- planted-landscape smoke

def test_planted_landscape_smoke():
    """SYNTHETIC: real multi_cmaes_search + real scorer on a 3-mode planted landscape,
    tiny budget. Checks plumbing only (fields present, modes <= restarts), not quality."""
    pytest.importorskip("jax")
    pytest.importorskip("evosax")
    import sys
    from dataclasses import replace
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from bench_search_scaling import make_evaluate, make_landscape

    from breakpoint_mjx.config import load_search_config, load_theta_space
    from breakpoint_mjx.paths import config_path

    space = load_theta_space(config_path("go1/theta_space.yaml"))
    cfg = replace(load_search_config(config_path("search.yaml")), popsize=16, horizon=8,
                  generations=3, n_eval=64)
    ev = make_evaluate(space, make_landscape(space, 3, 0))
    import jax
    recs, c = cov.run_until_saturated(ev, space, cfg, jax.random.PRNGKey(0), wave=4,
                                      max_waves=2, n_audit=64)
    assert len(recs) == c["waves"] * 4 * 16 * 3
    assert 0 <= c["S_obs"] <= 3 * c["waves"]
    assert c["audit"]["n"] == 64 and 0.0 <= c["audit"]["cp_upper"] <= 1.0
    assert math.isfinite(c["chao2"])


# ---------------------------------------------------------------- report wiring

def test_report_renders_coverage_line_and_search_selected_caveat():
    from tests.test_report import _make_report

    from breakpoint_mjx.report import render_html
    from breakpoint_mjx.stats import compute_report_stats

    _, c = _run(_fake_search(lambda s, r: CENTRES[r % 2]))
    report = _make_report()
    stats = compute_report_stats(report.families, n_total=100, n_failures=5,
                                 threshold=0.5, coverage=c)
    assert stats["coverage"] is c
    html = render_html(report, stats=stats)
    assert "Campaign coverage:" in html and "Clopper-Pearson" in html
    assert "chosen by the search" in html
    plain = compute_report_stats(report.families, n_total=100, n_failures=5)
    assert "coverage" not in plain
    assert "Campaign coverage:" not in render_html(report, stats=plain)


def test_pipeline_writes_coverage_outside_the_learned_layer_try():
    """The certificate must not be swallowed by the learned layer's broad except."""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "breakpoint_mjx" / "pipeline.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            seg = ast.get_source_segment(src, node) or ""
            assert 'families_payload["coverage"]' not in seg
    assert 'families_payload["coverage"] = coverage' in src
    assert 'os.environ.get("BP_COVERAGE") == "1"' in src
