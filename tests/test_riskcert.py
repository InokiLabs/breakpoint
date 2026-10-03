"""CPU tests for riskcert: exact densities, weight bound, intervals, estimators, gates.

All landscapes here are SYNTHETIC with analytic ground truth. Coverage checks are sanity
checks (betting CSs are conservative by construction), not evidence of usefulness; the
width comparison lives in scripts/bench_riskcert.py and docs/riskcert.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from breakpoint_mjx import riskcert as rc
from breakpoint_mjx.config import ThetaDim, ThetaSpace


def _space(d=3):
    return ThetaSpace(dims={f"x{i}": ThetaDim(f"x{i}", "model", (0.0, 1.0), 0.5, (0.1, 0.9))
                            for i in range(d)})


# ---------------------------------------------------------------- CP ------------
def test_clopper_pearson_matches_reference_values():
    # reference values from scipy.stats.beta.ppf (computed once, hard-coded)
    lo, hi = rc.clopper_pearson(2, 2000)
    assert lo == pytest.approx(0.00012112759055682972, rel=1e-6)
    assert hi == pytest.approx(0.0036076285698285315, rel=1e-6)
    lo, hi = rc.clopper_pearson(5, 10)
    assert (lo, hi) == pytest.approx((0.18708602844739855, 0.8129139715526015), rel=1e-6)
    assert rc.clopper_pearson(0, 0) == (0.0, 1.0)


def test_rule_of_three_identity_for_299_draws():
    ub = rc.clopper_pearson_upper(0, 299, 0.05)
    assert ub == pytest.approx(1 - 0.05 ** (1 / 299), rel=1e-6)
    assert 0.0099 < ub <= 0.01


# ------------------------------------------------------- densities --------------
def test_diag_truncnormal_density_integrates_to_one_and_samples_in_box():
    rng = np.random.default_rng(0)
    lo, hi = np.array([0.1, 0.1, 0.1]), np.array([0.9, 0.9, 0.9])
    tn = rc.DiagTruncNormal([0.15, 0.5, 0.88], [0.05, 0.3, 0.1], lo, hi)
    u = lo + rng.random((400_000, 3)) * (hi - lo)
    integral = np.exp(tn.logpdf(u)).mean() * np.prod(hi - lo)
    assert integral == pytest.approx(1.0, abs=0.01)
    s = tn.sample(5000, rng)
    assert np.all((s >= lo) & (s <= hi))
    assert np.all(np.isneginf(tn.logpdf(np.array([[0.0, 0.5, 0.5]]))))


def test_operational_dist_defaults_to_plausible_band_and_yaml_overrides():
    sp = _space(2)
    p = rc.OperationalDist.from_space(sp)
    assert p.kind == "uniform" and p.source == "plausible"
    assert np.allclose(p.lo, 0.1) and np.allclose(p.hi, 0.9)
    q = rc.OperationalDist.from_space(sp, {"x0": {"range": [0.2, 1.5]}})
    assert q.lo[0] == 0.2 and q.hi[0] == 1.0          # clipped to search bounds
    t = rc.OperationalDist.from_space(sp, {"x1": {"mean": 0.5, "sd": 0.1}})
    assert t.kind == "truncnorm"
    x = t.sample(1000, np.random.default_rng(0))
    assert np.all(np.isfinite(t.logpdf(x)))


# ----------------------------------------------------------- weights ------------
@pytest.mark.parametrize("d", [6, 12])
def test_defensive_mixture_weight_never_exceeds_one_over_alpha(d):
    rng = np.random.default_rng(1)
    p = rc.OperationalDist.from_space(_space(d))
    centers = rng.uniform(0.1, 0.9, size=(8, d))
    for alpha in (0.1, 0.3):
        q = rc.build_proposal(p, centers, alpha=alpha, sd_frac=0.02, rng=rng)
        x, _ = q.sample(20_000, rng)
        w = q.weight(x)
        assert w.max() <= 1.0 / alpha + 1e-12
        # also on p-draws (where q is thinnest relative to p)
        assert q.weight(p.sample(20_000, rng)).max() <= 1.0 / alpha + 1e-12
        assert w.mean() == pytest.approx(1.0, abs=0.05)   # E_q[p/q] = 1


def test_build_proposal_weights_follow_surrogate_failure_mass():
    rng = np.random.default_rng(2)
    p = rc.OperationalDist.from_space(_space(3))
    centers = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])
    q = rc.build_proposal(p, centers, g_fn=lambda x: (x[:, 0] > 0.6).astype(float),
                          alpha=0.3, rng=rng)
    # Voronoi cells split on x0+x1+x2 = 1.5, so cell 0 still holds some x0 > 0.6 mass
    assert q.pis[1] > 2 * q.pis[0] > 0


def test_no_centers_degrades_to_plain_mc():
    p = rc.OperationalDist.from_space(_space(2))
    q = rc.build_proposal(p, [])
    assert q.alpha == 1.0
    x, fp = q.sample(10, np.random.default_rng(0))
    assert fp.all() and np.allclose(q.weight(x), 1.0)


# ------------------------------------------------------- betting CS -------------
def test_betting_cs_covers_a_bernoulli_mean():
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(40):
        x = (rng.random(1000) < 0.2).astype(float)
        lo, hi = rc.betting_cs(x, 0.05)
        hits += lo <= 0.2 <= hi
    assert hits >= 38


def test_one_sided_upper_bound_on_all_zeros():
    lo, hi = rc.betting_cs(np.zeros(2000), 0.05, side="upper")
    assert lo == 0.0 and 0.0 < hi < 0.005


def test_dr_floor_shrinks_with_budget_and_alpha():
    f1 = rc.dr_floor_halfwidth(2000, 0.3)
    assert rc.dr_floor_halfwidth(20000, 0.3) < f1 < rc.dr_floor_halfwidth(2000, 0.1)
    # the floor at the originally proposed budget is WIDER than plain-MC CP at 1e-3
    lo, hi = rc.clopper_pearson(2, 2000)
    assert f1 > 0.5 * (hi - lo)


def test_crossover_is_beyond_the_cpu_toy_budget():
    n = rc.crossover_n(1e-2, 0.5, n_grid=[2000, 4000, 8000, 16000, 32000])
    assert n is not None and n > 2000


def test_failure_prob_link():
    g = rc.failure_prob_link(np.array([0.0, 0.45, 0.5, 0.55, 2.0]), 0.5)
    assert np.allclose(g, [0.0, 0.0, 0.5, 1.0, 1.0])


# ------------------------------------------------------- estimators -------------
def _toy(d=3):
    """Failure = ball of radius r around c in p = U[0.1,0.9]^d; exact P known."""
    import math
    c, r = np.array([0.3] * d), 0.12
    vol = math.pi ** (d / 2) / math.gamma(d / 2 + 1) * r ** d
    truth = vol / 0.8 ** d

    def fails(x):
        return np.linalg.norm(np.asarray(x) - c, axis=1) <= r
    return c, r, truth, fails


@pytest.mark.parametrize("which", ["good", "wrong", "none"])
def test_dr_estimate_covers_truth_whatever_the_surrogate(which):
    c, r, truth, fails = _toy()
    p = rc.OperationalDist.from_space(_space(3))
    g = {"good": lambda x: fails(x).astype(float),
         "wrong": lambda x: 1.0 - fails(x).astype(float),     # adversarially wrong
         "none": None}[which]
    hits = 0
    for s in range(25):
        rng = np.random.default_rng(100 + s)
        q = rc.build_proposal(p, [c], alpha=0.3, rng=rng)
        x, _ = q.sample(1500, rng)
        f, w = fails(x), q.weight(x)
        if g is None:
            e = rc.estimate_from_samples(f, w, alpha=0.3, method="is")
        else:
            ch, cr = rc.surrogate_mean_bound(g(p.sample(200_000, rng)), 0.025)
            e = rc.estimate_from_samples(f, w, g(x), ch, cr, alpha=0.3, method="dr")
        hits += e["lo"] <= truth <= e["hi"]
    assert hits >= 23


def test_operational_risk_block_and_sequential_stop():
    c, r, truth, fails = _toy()
    p = rc.OperationalDist.from_space(_space(3))
    rng = np.random.default_rng(7)
    q = rc.build_proposal(p, [c], alpha=0.3, rng=rng)
    calls = []

    def ev(x):
        calls.append(len(x))
        return fails(x)
    o = rc.operational_risk(ev, p, q, g_fn=lambda x: fails(x).astype(float), n_max=6000,
                            batch=500, target_halfwidth=0.02, n_quad=100_000, rng=rng)
    assert o["stopped"] == "target_halfwidth" and o["n_rollouts"] < 6000
    assert all(n == 500 for n in calls)
    assert o["lo"] <= truth <= o["hi"]
    assert o["w_max_observed"] <= o["w_max_bound"] + 1e-12
    for key in ("dist", "estimate", "method", "alpha", "ess", "plain_mc_baseline", "caveat"):
        assert key in o
    assert o["plain_mc_baseline"]["n"] > 0


# ----------------------------------------------------------- certify ------------
def test_certify_box_certifies_a_safe_box_and_refuses_a_box_with_a_region():
    c, r, truth, fails = _toy()
    safe = rc.certify_box(fails, [0.6] * 3, [0.9] * 3, 299, eps=0.01)
    assert safe["failures"] == 0 and safe["certified"]
    assert safe["upper_bound"] == pytest.approx(0.00997, abs=1e-4)
    bad = rc.certify_box(fails, [0.2] * 3, [0.4] * 3, 299, eps=0.01)
    assert bad["failures"] > 0 and not bad["certified"]


def test_certify_envelope_uses_the_simulator_not_the_surrogate():
    from breakpoint_mjx.envelope import certify_envelope
    sp = _space(3)
    env = {"envelope": {f"x{i}": [0.2, 0.4] for i in range(3)}}
    c, r, truth, fails = _toy()
    out = certify_envelope(env, sp, fails, n=299)
    assert out["failures"] > 0 and not out["certified"] and not out["box_is_point"]


def test_run_stage_end_to_end_on_a_toy_simulator():
    from breakpoint_mjx.schemas import FailureRecord
    sp = _space(3)
    c, r, truth, fails = _toy()
    recs = [FailureRecord(theta_vector=list(c + 0.01), theta_named={}, failure_score=0.9,
                          component_signals={}, rollout_signature=[], seed=0,
                          transfer_score=1.0, severity="high", id="cmaes-0")]

    class Sur:
        def predict_score(self, x):
            return np.where(fails(x), 1.0, 0.0)
    out = rc.run_stage(fails, sp, [], recs, 0.5, surrogate=Sur(),
                       envelope={"envelope": {f"x{i}": [0.6, 0.9] for i in range(3)}},
                       n_max=2000, batch=500, target_halfwidth=None, n_quad=50_000)
    o = out["operational_risk"]
    assert o["lo"] <= truth <= o["hi"] and o["n_rollouts"] == 2000
    assert o["proposal"]["n_components"] == 1
    assert out["envelope_certificate"]["certified"]


# ------------------------------------------------------- relabels ---------------
def test_report_renders_operational_risk_and_relabels_search_rate():
    from breakpoint_mjx.report import render_html
    from breakpoint_mjx.stats import compute_report_stats
    from tests.test_report import _make_report
    rep = _make_report()
    stats = compute_report_stats(rep.families, n_total=100, n_failures=5, threshold=0.5)
    c, r, truth, fails = _toy()
    p = rc.OperationalDist.from_space(_space(3))
    q = rc.build_proposal(p, [c], alpha=0.3)
    o = rc.operational_risk(fails, p, q, n_max=500, batch=500)
    html = render_html(rep, stats=stats, operational_risk=o)
    assert "search-evaluated worlds" in html and "NOT a deployment failure probability" in html
    assert "Operational failure probability" in html and "anytime-valid" in html


def test_families_json_labels_the_search_selected_rate():
    from breakpoint_mjx.webpush import build_families_json

    class T:
        name = "toy"
    out = build_families_json(target=T(), space=_space(2), families=[], n_total=10,
                              n_failures=5, failure_rate_ci=(0.5, 0.2, 0.8),
                              search_mode="cmaes", horizon=10,
                              operational_risk={"estimate": 0.01})
    assert out["stats"]["failure_rate_basis"] == "search_selected"
    assert out["operational_risk"]["estimate"] == 0.01


def test_surrogate_predict_proba_docstring_no_longer_claims_calibration():
    from breakpoint_mjx.surrogate import Surrogate
    doc = Surrogate.predict_proba.__doc__
    assert "NOT calibrated" in doc and not doc.strip().startswith("Calibrated")


def test_cli_run_exposes_the_risk_stage():
    from breakpoint_mjx.cli import build_parser
    a = build_parser().parse_args(["run", "--robot", "go1", "--risk", "8192",
                                   "--risk-dist", "ops.yaml"])
    assert a.risk == 8192 and a.risk_dist == "ops.yaml"
    assert build_parser().parse_args(["run", "--robot", "go1"]).risk == 0


def test_moment_fit_beats_fixed_on_a_failure_set_wide_in_inert_dims():
    """SYNTHETIC. The failure set is a slab in x0 only (x1..x3 inert). A fixed, narrow
    component leaves most failing p-mass at w ~ 1/alpha (variance factor > 1, i.e. worse
    than plain MC); the moment-matched one spreads along the inert dims."""
    rng = np.random.default_rng(11)
    p = rc.OperationalDist.from_space(_space(4))

    def fails(x):
        return np.abs(np.asarray(x)[:, 0] - 0.3) < 0.02
    g = lambda x: fails(x).astype(float)
    center = [[0.3, 0.5, 0.5, 0.5]]
    xv = p.sample(400_000, rng)
    fv = fails(xv)
    vf = {}
    for fit in ("fixed", "moments"):
        q = rc.build_proposal(p, center, g_fn=g, alpha=0.3, fit=fit, rng=rng)
        vf[fit] = q.weight(xv[fv]).sum() / fv.sum()
    assert vf["fixed"] > 1.0 > vf["moments"]
    with pytest.raises(ValueError):
        rc.build_proposal(p, center, g_fn=g, fit="bogus")
