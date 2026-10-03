"""Cause-Stratified Operational Risk (breakpoint_mjx/famrisk.py). Pure numpy, CPU."""
from __future__ import annotations

import json

import numpy as np
import pytest

from breakpoint_mjx import famrisk as fr
from breakpoint_mjx import mincause as mc
from breakpoint_mjx.config import ThetaDim, ThetaSpace
from breakpoint_mjx.report import render_html
from breakpoint_mjx.riskcert import OperationalDist
from breakpoint_mjx.schemas import BreakPointReport, FailureFamily, FailureRecord

D = 6


def _space():
    return ThetaSpace(dims={
        f"d{i}": ThetaDim(f"d{i}", "model", (-1.0, 1.0), 0.0, (-0.5, 0.5)) for i in range(D)
    })


def _region(space, cause):
    return fr.region_from_cause(cause, space.names())


def _fam(fid, rank=1, **kw):
    f = FailureFamily(id=fid, name=f"fam {fid}", member_ids=["r1"], representative_id="r1",
                      triggering_parameters={}, mean_failure_score=0.9,
                      mean_transfer_score=0.5, priority_rank=rank)
    for k, v in kw.items():
        setattr(f, k, v)
    return f


# ------------------------------------------------------------------ exact mass
def test_uniform_mass_is_exact_product_of_tails():
    space = _space()
    p = OperationalDist.from_space(space)            # uniform on [-0.5, 0.5]
    r = _region(space, [{"dim": "d1", "value": 0.3, "direction": "high"},
                        {"dim": "d4", "value": -0.4, "direction": "low"}])
    m, per = fr.stratum_mass(p, r)
    assert per == pytest.approx([0.2, 0.1])
    assert m == pytest.approx(0.02)


def test_region_beyond_the_odd_has_zero_mass_and_spends_no_rollouts():
    space = _space()
    p = OperationalDist.from_space(space)
    r = _region(space, [{"dim": "d2", "value": 0.7, "direction": "high"}])  # outside [-.5,.5]
    calls = []

    def fail_fn(X, seeds):
        calls.append(len(X))
        return np.ones(len(X), bool)
    o = fr.family_risk(fail_fn, p, r, 64)
    assert o["mass"] == 0.0 and o["n"] == 0 and not calls
    assert o["class"] == "outside_odd" and o["risk_hi"] == 0.0


def test_truncnorm_mass_matches_monte_carlo_and_far_tail_is_positive():
    space = _space()
    block = {n: {"mean": 0.0, "sd": 0.1} for n in space.names()}
    p = OperationalDist.from_space(space, block)
    r = _region(space, [{"dim": "d0", "value": 0.15, "direction": "high"}])
    m, _ = fr.stratum_mass(p, r)
    X = p.sample(200_000, np.random.default_rng(0))
    assert m == pytest.approx(float((X[:, 0] >= 0.15).mean()), abs=2e-3)
    far = _region(space, [{"dim": "d0", "value": 0.49, "direction": "high"}])
    mf, _ = fr.stratum_mass(p, far)
    assert 0.0 < mf < 1e-5                          # erfc keeps a 4.9-sigma tail


def test_conditional_sampler_stays_in_region_and_matches_p_elsewhere():
    space = _space()
    for block in (None, {n: {"mean": 0.1, "sd": 0.2} for n in space.names()}):
        p = OperationalDist.from_space(space, block)
        r = _region(space, [{"dim": "d3", "value": 0.35, "direction": "high"},
                            {"dim": "d1", "value": -0.3, "direction": "low"}])
        X = fr.sample_stratum(p, r, 5000, np.random.default_rng(1))
        assert fr.in_region(X, r).all()
        assert np.all(X >= p.lo) and np.all(X <= p.hi)
        # an unconstrained dim keeps p's marginal
        ref = p.sample(5000, np.random.default_rng(2))[:, 0]
        assert abs(X[:, 0].mean() - ref.mean()) < 0.02


def test_integer_dim_uses_the_effective_rounded_threshold():
    space = ThetaSpace(dims={
        "latency_steps": ThetaDim("latency_steps", "rollout", (0.0, 4.0), 0.0, (0.0, 2.0)),
        "x": ThetaDim("x", "model", (0.0, 1.0), 0.0, (0.0, 1.0)),
    })
    r = fr.region_from_cause([{"dim": "latency_steps", "value": 0.516, "direction": "high"}],
                             space.names(), integer_dims=frozenset({"latency_steps"}))
    assert r[0]["bound"] == 0.5 and r[0]["value"] == 1
    m, _ = fr.stratum_mass(OperationalDist.from_space(space), r)
    assert m == pytest.approx(0.75)                  # round(x) >= 1 on U[0, 2]
    assert "latency_steps >= 1" in fr.region_string(r)


# ------------------------------------------------------------------ risk + classes
def test_risk_interval_is_mass_times_cp_and_covers_truth():
    space = _space()
    p = OperationalDist.from_space(space)
    r = _region(space, [{"dim": "d0", "value": 0.4, "direction": "high"}])  # mass 0.1

    def fail_fn(X, seeds):
        return np.random.default_rng(int(seeds[0])).random(len(X)) < 0.5
    o = fr.family_risk(fail_fn, p, r, 400, rng=np.random.default_rng(0))
    assert o["mass"] == pytest.approx(0.1)
    assert o["risk_lo"] == pytest.approx(0.1 * o["cond_lo"])
    assert o["risk_lo"] <= 0.05 <= o["risk_hi"]
    assert o["class"] == "in_domain_risk" and o["label"] == "SIMULATION"


def test_classify_thresholds():
    assert fr.classify(0.0, 0.0, 0.0) == "outside_odd"
    assert fr.classify(1e-4, 0.0, 5e-7) == "outside_odd"
    assert fr.classify(0.1, 2e-3, 0.02) == "in_domain_risk"
    assert fr.classify(0.1, 1e-4, 0.02) == "in_domain_low"


def test_resolve_odd_order(tmp_path):
    space = _space()
    risk = tmp_path / "risk.yaml"
    risk.write_text("operational:\n  d0: {range: [0.0, 0.1]}\n")
    contract = tmp_path / "contract.yaml"
    contract.write_text("obs: {}\noperational:\n  d0: {range: [-0.2, 0.2]}\n")
    p, src = fr.resolve_odd(space, {"BP_RISK_DIST": str(risk), "BP_OBS_SPEC": str(contract)})
    assert p.hi[0] == pytest.approx(0.1) and "BP_RISK_DIST" in src
    p, src = fr.resolve_odd(space, {"BP_OBS_SPEC": str(contract)})
    assert p.hi[0] == pytest.approx(0.2) and "contract" in src
    plain = tmp_path / "plain.yaml"
    plain.write_text("obs: {}\n")
    p, src = fr.resolve_odd(space, {"BP_OBS_SPEC": str(plain)})
    assert p.hi[0] == pytest.approx(0.5) and "plausible" in src


def test_settings_budget_presets_and_override():
    assert fr.settings({"BP_BUDGET": "smoke"})[1] == 32
    assert fr.settings({"BP_BUDGET": "standard"})[1] == 256
    assert fr.settings({})[1] == 1024
    assert fr.settings({"BP_BUDGET": "smoke", "BP_FAMRISK_N": "7"})[1] == 7
    assert fr.settings({"BP_FAMRISK": "0"})[0] is False
    # opt-in: the out-of-ODD label failed its safety bar in the unsafe direction
    assert fr.settings({})[0] is False
    assert fr.settings({"BP_FAMRISK": "1"})[0] is True


# ------------------------------------------------------------------ stage
def _and_fn(X, seeds):
    return (X[:, 2] > 0.3) & (X[:, 5] < -0.2)


def test_run_stage_sets_odd_risk_only_on_certified_families_and_bonferroni_splits():
    space = _space()
    a = _fam("a", cause_certified=True, minimal_cause=[
        {"dim": "d2", "value": 0.31, "direction": "high"},
        {"dim": "d5", "value": -0.21, "direction": "low"}])
    b = _fam("b", 2, cause_certified=True, minimal_cause=[
        {"dim": "d0", "value": 0.9, "direction": "high"}])          # outside the ODD
    c = _fam("c", 3, cause_certified=False)
    s = fr.run_stage([a, b, c], space, _and_fn, environ={"BP_FAMRISK": "1", "BP_FAMRISK_N": "200"}, log=None)
    assert s["status"] == "ran" and s["n_certified"] == 2
    assert c.odd_risk is None
    assert b.odd_risk["class"] == "outside_odd" and b.odd_risk["n"] == 0
    # only one family spends rollouts, so it gets the whole delta
    assert s["delta_family"] == pytest.approx(0.05)
    assert a.odd_risk["n"] == 200 and a.odd_risk["k"] == 200
    assert a.odd_risk["mass"] == pytest.approx(0.19 * 0.29)
    assert s["counts"] == {"in_domain_risk": 1, "in_domain_low": 0, "outside_odd": 1}
    json.dumps(s)
    json.dumps(a.odd_risk)


def test_run_stage_disabled_no_certified_and_error_safe():
    space = _space()
    f = _fam("a", cause_certified=True, minimal_cause=[
        {"dim": "d2", "value": 0.31, "direction": "high"}])
    assert fr.run_stage([f], space, _and_fn, environ={"BP_FAMRISK": "0"},
                        log=None)["status"] == "disabled"
    assert f.odd_risk is None
    assert fr.run_stage([_fam("x")], space, _and_fn, environ={"BP_FAMRISK": "1"},
                        log=None)["status"] == "no_certified"

    def boom(X, seeds):
        raise RuntimeError("gpu gone")
    s = fr.run_stage([f], space, boom, environ={"BP_FAMRISK": "1"}, log=None)
    assert s["status"] == "error" and f.odd_risk is None


def test_mincause_run_stage_calls_famrisk_on_the_real_path():
    space = _space()
    names = space.names()
    th = np.random.default_rng(1).uniform(-1, 1, D)
    th[2], th[5] = 0.8, -0.9
    rec = FailureRecord(theta_vector=list(th), theta_named=dict(zip(names, th)),
                        failure_score=1.0, component_signals={}, rollout_signature=[],
                        seed=0, id="r1")
    fam = _fam("fam-0", triggering_parameters=dict(zip(names, th)))
    out_summary = {}
    out, ran = mc.run_stage([fam], {"r1": rec}, space, lambda: _and_fn,
                            environ={"BP_FAMRISK": "1", "BP_FAMRISK_N": "64"}, log=None,
                            famrisk_out=out_summary)
    assert ran and out[0].cause_certified is True
    assert out_summary["status"] == "ran"
    o = out[0].odd_risk
    assert o is not None and o["n"] == 64 and o["class"] == "in_domain_risk"
    # the region CMC certified is inside the true AND-region, so every conditional draw fails
    assert o["k"] == 64
    out2, _ = mc.run_stage([fam], {"r1": rec}, space, lambda: _and_fn,
                           environ={"BP_FAMRISK": "0"}, log=None, famrisk_out=out_summary)
    assert out2[0].odd_risk is None and out_summary["status"] == "disabled"


def test_residual_only_subtracts_under_the_same_domain():
    space = _space()
    p = OperationalDist.from_space(space)
    summ = {"status": "ran", "odd": p.spec(), "union_upper": 0.01}
    risk = {"dist": p.spec(), "estimate": 0.05, "lo": 0.04, "hi": 0.06}
    r = fr.residual(summ, risk)
    assert r["comparable"] and r["unexplained_mass_lower"] == pytest.approx(0.03)
    other = OperationalDist.from_space(space, {"d0": {"range": [0.0, 0.1]}}).spec()
    r2 = fr.residual(summ, {**risk, "dist": other})
    assert r2["comparable"] is False and "unexplained_mass_lower" not in r2
    assert fr.residual(summ, None) is None


# ------------------------------------------------------------------ report
def _report(fams):
    return BreakPointReport(run_id="t", policy_descriptor="pol", comparison={}, families=fams)


def test_report_orders_in_domain_by_upper_bound_and_collapses_out_of_odd():
    lo = _fam("lo", cause_certified=True, name="low one")
    hi = _fam("hi", 2, cause_certified=True, name="high one")
    out = _fam("out", 3, cause_certified=True, name="far one")
    base = {"region": [], "mass": 0.1, "mass_per_dim": {}, "k": 1, "n": 32,
            "cond_lo": 0.0, "cond_hi": 0.2, "risk": 0.003, "label": "SIMULATION"}
    lo.odd_risk = {**base, "region_str": "d0 >= 0.1", "risk_lo": 0.0, "risk_hi": 0.002,
                   "class": "in_domain_low"}
    hi.odd_risk = {**base, "region_str": "d1 >= 0.1", "risk_lo": 0.002, "risk_hi": 0.05,
                   "class": "in_domain_risk"}
    out.odd_risk = {**base, "region_str": "d2 >= 0.9", "mass": 0.0, "k": 0, "n": 0,
                    "risk_lo": 0.0, "risk_hi": 0.0, "class": "outside_odd"}
    summ = {"status": "ran", "odd_source": "plausible bands", "union_upper": 0.052,
            "rollouts": 64, "min_risk": 1e-3, "out_eps": 1e-6}
    html = render_html(_report([lo, hi, out]), famrisk=summ)
    sec = html[html.index("Operational risk by family"):]
    assert sec.index("high one") < sec.index("low one") < sec.index("<details>") \
        < sec.index("far one")
    assert "Outside your operating domain (1 family)" in sec
    assert "SIMULATION" in sec
    assert html.count("Operational risk (declared ODD):") == 3


def test_report_says_so_when_no_family_certified():
    html = render_html(_report([_fam("a")]),
                       famrisk={"status": "no_certified", "odd_source": "plausible bands"})
    assert "No family was certified by the minimal-cause stage at this budget" in html
    assert "Operational risk (declared ODD):" not in html
    assert "Operational risk by family" not in render_html(_report([_fam("a")]))


# ------------------------------------------------------------------ CP + relaxation
def test_cp_interval_symmetry_matches_direct_clopper_pearson():
    from breakpoint_mjx.riskcert import clopper_pearson
    for k, n in ((0, 50), (3, 50), (40, 50), (50, 50), (180, 200)):
        assert fr.cp_interval(k, n, 0.05) == pytest.approx(clopper_pearson(k, n, 0.05),
                                                           abs=1e-9)


def test_relax_region_moves_toward_nominal_by_bisection_resolution_only():
    space = _space()                                        # span 2, nominal 0
    r = _region(space, [{"dim": "d0", "value": 0.6, "direction": "high"},
                        {"dim": "d1", "value": -0.01, "direction": "low"}])
    rr = fr.relax_region(r, space, 6)
    assert rr[0]["bound"] == pytest.approx(0.6 - 2 / 64)
    assert rr[1]["bound"] == 0.0                            # clamped at nominal


def test_zero_mass_certified_region_falls_back_to_relaxed_region():
    space = _space()
    p = OperationalDist.from_space(space)                  # band [-0.5, 0.5]
    r = _region(space, [{"dim": "d0", "value": 0.51, "direction": "high"}])
    rr = fr.relax_region(r, space, 6)                      # 0.51 - 0.03125 < 0.5

    def fail_fn(X, seeds):
        return X[:, 0] >= 0.49
    o = fr.family_risk(fail_fn, p, r, 100, relaxed=rr)
    assert o["region_basis"] == "relaxed" and o["certified_region_mass"] == 0.0
    assert o["mass"] > 0 and o["n"] == 100 and o["class"] != "outside_odd"
    # the relaxation is off => the family is called outside the ODD with no rollouts
    s = fr.run_stage([_fam("a", cause_certified=True, minimal_cause=[
        {"dim": "d0", "value": 0.51, "direction": "high"}])], space, fail_fn,
        environ={"BP_FAMRISK": "1", "BP_FAMRISK_RELAX": "0", "BP_FAMRISK_N": "50"}, log=None)
    assert s["counts"]["outside_odd"] == 1 and s["boundary_relaxation"] is None
