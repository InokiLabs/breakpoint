"""Certified Minimal Cause (breakpoint_mjx/mincause.py): pure-core tests on CPU with
analytic failure functions (no jax, no GPU)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from breakpoint_mjx import mincause as mc
from breakpoint_mjx.config import ThetaDim, ThetaSpace
from breakpoint_mjx.naming import label_from_cause
from breakpoint_mjx.schemas import BreakPointReport, FailureFamily, FailureRecord

D = 8
NOM = np.zeros(D)
LO = -np.ones(D)
HI = np.ones(D)


def _space():
    return ThetaSpace(dims={
        f"d{i}": ThetaDim(f"d{i}", "model", (-1.0, 1.0), 0.0, (-0.5, 0.5)) for i in range(D)
    })


def _and_fn(p=0.0):
    """Fails iff d2 > 0.3 AND d5 < -0.2, with outcome flip probability p."""
    def f(X, seeds):
        t = (X[:, 2] > 0.3) & (X[:, 5] < -0.2)
        if p > 0:
            t = t ^ (np.random.default_rng(int(seeds[0])).random(len(X)) < p)
        return t
    return f


def _rep():
    th = np.random.default_rng(1).uniform(-1, 1, D)
    th[2], th[5] = 0.8, -0.9
    return th


def test_clopper_pearson_bounds():
    assert mc.cp_lower(8, 8) == pytest.approx(0.1 ** (1 / 8))      # 0.75
    assert mc.cp_lower(7, 8) >= 0.5 > mc.cp_lower(6, 8)
    assert mc.cp_upper(0, 8) < 0.5
    assert mc.decide(8, 8, 8) == "fail"
    assert mc.decide(0, 8, 8) == "pass"
    assert mc.decide(4, 8, 8) == "undecided"
    assert mc.decide(1, 1, 1) == "fail"           # K=1 ablation: no certification


@pytest.mark.parametrize("p", [0.0, 0.1])
def test_recovers_planted_and_set(p):
    r = mc.certified_minimal_cause(_rep(), NOM, LO, HI, _and_fn(p), k_seeds=8, seed0=3)
    assert r.certified
    assert sorted(r.cause) == [2, 5]
    assert dict(zip(r.cause, r.directions)) == {2: "high", 5: "low"}
    assert r.necessary is True
    # magnitude shrink lands near the planted thresholds (6 bisection steps)
    assert r.theta_min[2] == pytest.approx(0.3, abs=0.8 / 64 + 1e-9)
    assert r.theta_min[5] == pytest.approx(-0.2, abs=0.9 / 64 + 1e-9)
    assert r.min_distance == pytest.approx(max(abs(r.theta_min[2]), abs(r.theta_min[5])) / 2)
    others = [i for i in range(D) if i not in (2, 5)]
    assert np.all(r.theta_min[others] == 0.0)


def test_necessity_false_for_or_structure():
    """d2 > 0.3 OR d5 < -0.2 with both present: either alone suffices, so reverting the
    chosen cause while keeping the other dim still fails -> necessary is False."""
    def f(X, seeds):
        return (X[:, 2] > 0.3) | (X[:, 5] < -0.2)
    r = mc.certified_minimal_cause(_rep(), NOM, LO, HI, f, k_seeds=4)
    assert r.certified and len(r.cause) == 1 and r.cause[0] in (2, 5)
    assert r.necessary is False


def test_uncertified_representative_keeps_nothing():
    def f(X, seeds):  # never fails
        return np.zeros(len(X), bool)
    r = mc.certified_minimal_cause(_rep(), NOM, LO, HI, f, k_seeds=4)
    assert not r.certified and r.cause == [] and r.theta_min is None
    assert "not certified" in mc.margin_string(r, _space().names())


def test_batched_calls_are_few():
    r = mc.certified_minimal_cause(_rep(), NOM, LO, HI, _and_fn(), k_seeds=8)
    # one call per ddmin round + shrink steps + final certification, not one per config
    assert r.calls <= r.rounds + mc.SHRINK_STEPS + 3
    assert r.sims % 4 == 0            # sequential rounds of K/2 = 4 seeds


def test_label_and_margin_consistency():
    names = _space().names()
    r = mc.certified_minimal_cause(_rep(), NOM, LO, HI, _and_fn(), k_seeds=8)
    rep = _rep()
    mc.check_label_consistency(r, names, NOM, dict(zip(names, rep)))
    s = mc.margin_string(r, names)
    assert "d2 >= 0.3" in s and "d5 <= -0.2" in s and "necessary: yes" in s
    # noiseless: the first sequential look (4 seeds, alpha 0.07) is already decisive
    assert "certified 4/4 seeds (CP lower 0.56)" in s
    # a displayed value on the wrong side of nominal must trip the assertion
    bad = dict(zip(names, rep))
    bad["d2"] = -0.5
    with pytest.raises(AssertionError):
        mc.check_label_consistency(r, names, NOM, bad)


def test_label_from_cause():
    assert label_from_cause([("friction_floor", "low")]) == "low-friction slip"
    assert label_from_cause([("latency_steps", "high"), ("friction_floor", "low")]) == \
        "latency gait collapse + low-friction slip"
    assert label_from_cause([]) == "fails at nominal"


def _fam(fid, rank, members, theta):
    return FailureFamily(id=fid, name=f"old {fid}", member_ids=members,
                         representative_id=members[0], triggering_parameters=theta,
                         mean_failure_score=1.0, mean_transfer_score=0.5, priority_rank=rank)


def _rec(rid, theta_vec, names):
    return FailureRecord(theta_vector=list(theta_vec), theta_named=dict(zip(names, theta_vec)),
                         failure_score=1.0, component_signals={}, rollout_signature=[],
                         seed=0, id=rid)


def test_apply_merges_families_with_same_cause_and_renames():
    space = _space()
    names = space.names()
    a, b, c = _rep(), _rep().copy(), np.random.default_rng(7).uniform(-1, 1, D)
    b[0] = -b[0]                                   # different noise dims, same cause
    c[2], c[5] = 0.1, 0.1                          # does not fail: stays uncertified
    recs = {"r1": _rec("r1", a, names), "r2": _rec("r2", b, names),
            "r3": _rec("r3", c, names), "r4": _rec("r4", a, names)}
    fams = [_fam("fam-0", 1, ["r1", "r4"], recs["r1"].theta_named),
            _fam("fam-1", 3, ["r2"], recs["r2"].theta_named),
            _fam("fam-2", 2, ["r3"], recs["r3"].theta_named)]
    out = mc.apply_mincause(fams, recs, space, _and_fn(), k_seeds=4, log=None)
    assert [f.id for f in out] == ["fam-0", "fam-2"]
    f0 = out[0]
    assert f0.merged_from == ["fam-0", "fam-1"]
    assert f0.member_ids == ["r1", "r4", "r2"] and recs["r2"].family_id == "fam-0"
    assert f0.cause_certified is True and f0.cause_necessary is True
    assert {d["dim"]: d["direction"] for d in f0.minimal_cause} == {"d2": "high", "d5": "low"}
    assert f0.name == "high d2 + low d5" or f0.name == "low d5 + high d2"
    assert f0.margin.startswith("fails when ")
    assert out[1].cause_certified is False and out[1].name == "old fam-2"


def test_schema_back_compat_and_roundtrip():
    old = {"id": "fam-0", "name": "x", "member_ids": ["a"], "representative_id": "a",
           "triggering_parameters": {"d0": 0.1}, "mean_failure_score": 1.0,
           "mean_transfer_score": 0.2, "priority_rank": 1}
    rep = BreakPointReport.from_json(json.dumps(
        {"run_id": "r", "policy_descriptor": "p", "comparison": {}, "families": [old],
         "mp4_paths": {}}))
    f = rep.families[0]
    assert f.minimal_cause is None and f.min_distance is None
    assert f.cause_certified is None and f.cause_necessary is None
    f.minimal_cause = [{"dim": "d0", "value": 0.3, "direction": "high"}]
    f.min_distance = 0.15
    again = BreakPointReport.from_json(rep.to_json()).families[0]
    assert again.minimal_cause == f.minimal_cause and again.min_distance == 0.15


def test_run_stage_disabled_path_never_builds_fail_fn():
    fams = [_fam("fam-0", 1, ["r1"], {})]

    def boom():
        raise AssertionError("fail_fn must not be built when BP_MINCAUSE=0")
    out, ran = mc.run_stage(fams, {}, _space(), boom, environ={"BP_MINCAUSE": "0"}, log=None)
    assert out is fams and ran is False


def test_run_stage_default_on_and_error_safe():
    space = _space()
    names = space.names()
    recs = {"r1": _rec("r1", _rep(), names)}
    fams = [_fam("fam-0", 1, ["r1"], recs["r1"].theta_named)]
    out, ran = mc.run_stage(fams, recs, space, lambda: _and_fn(), environ={}, log=None)
    assert ran and out[0].minimal_cause is not None and fams[0].minimal_cause is None

    def broken(X, seeds):
        raise RuntimeError("gpu gone")
    out, ran = mc.run_stage(fams, recs, space, lambda: broken, environ={}, log=None)
    assert ran is False and out is fams and fams[0].name == "old fam-0"


def test_shrink_never_reports_an_uncertified_point_under_interaction():
    """Fails iff d2 - d5 > 0.9 (a coupled budget): each dim can be bisected down on its own
    while the other is still large, but applying both moves together breaks the failure.
    The rolled-back speculative combine must still return a point that really fails."""
    def f(X, seeds):
        return (X[:, 2] - X[:, 5]) > 0.9
    r = mc.certified_minimal_cause(_rep(), NOM, LO, HI, f, k_seeds=4)
    assert sorted(r.cause) == [2, 5] and r.necessary is True
    assert f(r.theta_min[None, :], None)[0]
    assert r.shrunk and r.theta_min[2] - r.theta_min[5] < 0.8 + 0.9 - 0.05


# --------------------------------------------------------------------------- #
# Round 2: sequential certification, budget presets, progress, cap, snapping
# --------------------------------------------------------------------------- #
def test_look_schedule_exact_error_at_most_alpha_and_first_look_can_decide():
    for k, mult in ((8, 4), (8, 3), (8, 2), (4, 2), (16, 2)):
        looks, cs, ds = mc.look_schedule(k, k * mult)
        assert looks == sorted(looks) and looks[-1] == k * mult
        assert cs[0] <= looks[0] and ds[0] >= 0      # all-fail / all-pass decides at once
        for q in (0.5, 0.45, 0.3):                     # true rate below the bar
            pf, _, _, _ = mc.operating_characteristics(looks, cs, ds, q)
            assert pf <= mc.ALPHA + 1e-9
        for q in (0.5, 0.55, 0.7):                     # true rate at/above the bar
            _, pp, _, _ = mc.operating_characteristics(looks, cs, ds, q)
            assert pp <= mc.ALPHA + 1e-9
    looks, _, _ = mc.look_schedule(8, 32)
    assert looks[:3] == [4, 8, 12]                   # rounds of K/2
    assert mc.look_schedule(4, 8)[0] == [4, 8]        # n=2 and n=6 can never decide
    assert mc.look_schedule(1, 1) == ([1], [1], [0])


def test_sequential_uses_fewer_expected_seeds_than_round1_at_equal_power():
    looks, cs, ds = mc.look_schedule(8, 32)
    pf, _, _, en = mc.operating_characteristics(looks, cs, ds, 0.9)
    assert pf > 0.99 and en < 0.7 * 9.6            # round-1 fixed K=8 needs ~9.6 at q=0.9


def test_sequential_certifier_stops_early_and_is_one_call_per_round():
    calls = []

    def f(X, seeds):
        calls.append(len(X))
        return X[:, 0] > 0                           # deterministic: decisive at 4 seeds
    c = mc._Certifier(f, k_seeds=8, max_seed_mult=4)
    dec, k, n = c.certify(np.array([[1.0], [-1.0]]))
    assert dec == ["fail", "pass"] and list(n) == [4, 4] and calls == [8]
    fixed = mc._Certifier(f, k_seeds=8, max_seed_mult=4, sequential=False)
    _, _, n2 = fixed.certify(np.array([[1.0], [-1.0]]))
    assert list(n2) == [8, 8]


def test_sequential_false_positive_rate_bounded():
    """A config failing exactly half the time must rarely certify as failing."""
    rng = np.random.default_rng(0)

    def f(X, seeds):
        return rng.random(len(X)) < 0.5
    c = mc._Certifier(f, k_seeds=8, max_seed_mult=4)
    dec, _, _ = c.certify(np.zeros((2000, 1)))
    wrong = sum(d == "fail" for d in dec) / len(dec)
    assert wrong <= mc.ALPHA                          # union bound over looks


def test_latency_threshold_prints_as_integer():
    names = [f"d{i}" for i in range(D)]
    names[2] = "latency_steps"
    r = mc.certified_minimal_cause(_rep(), NOM, LO, HI, _and_fn(), k_seeds=4)
    r.theta_min = r.theta_min.copy()
    r.theta_min[2] = 0.516
    s = mc.margin_string(r, names)
    assert "latency_steps >= 1 " in s or "latency_steps >= 1;" in s
    assert "0.516" not in s
    assert mc.effective_threshold("latency_steps", 1.5) == 2      # half to even, as jnp
    assert mc.effective_threshold("latency_steps", 2.5) == 2
    assert mc.effective_threshold("friction_floor", 0.37) == 0.37


def test_stage_settings_presets_and_overrides():
    assert mc.stage_settings({})[1] == mc.CMC_PRESETS["full"]
    name, st = mc.stage_settings({"BP_BUDGET": "smoke"})
    assert name == "smoke" and st["k"] == 4 and st["mult"] == 2 and st["top"] == 2
    assert st["cap"] is not None
    assert mc.stage_settings({"BP_BUDGET": "standard"})[1] == {
        "k": 8, "mult": 3, "top": 4, "cap": None, "shrink": mc.SHRINK_STEPS}
    assert mc.stage_settings({"BP_BUDGET": "smoke"})[1]["shrink"] == 3
    _, st = mc.stage_settings({"BP_BUDGET": "smoke", "BP_MINCAUSE_K": "8",
                               "BP_MINCAUSE_MAXMULT": "3", "BP_MINCAUSE_CAP": "all"})
    assert st["k"] == 8 and st["mult"] == 3 and st["cap"] is None and st["top"] == 2


def _four_families():
    space = _space()
    names = space.names()
    recs, fams = {}, []
    for j in range(4):
        rid = f"r{j}"
        recs[rid] = _rec(rid, _rep(), names)
        fams.append(_fam(f"fam-{j}", 4 - j, [rid], recs[rid].theta_named))  # reversed rank
    return space, recs, fams


def test_smoke_budget_certifies_only_top_families_and_labels_the_rest():
    space, recs, fams = _four_families()
    lines = []
    out, ran = mc.run_stage(fams, recs, space, lambda: _and_fn(),
                            environ={"BP_BUDGET": "smoke", "BP_MINCAUSE_CAP": "all"},
                            log=lines.append)
    assert ran
    by_id = {f.id: f for f in out}
    # top 2 by priority_rank are fam-3 (rank 1) and fam-2 (rank 2); they share a cause
    assert by_id["fam-3"].cause_certified is True
    assert by_id["fam-3"].merged_from == ["fam-3", "fam-2"]
    for fid in ("fam-0", "fam-1"):
        assert by_id[fid].cause_certified is None
        assert "CMC skipped at smoke budget" in by_id[fid].margin
        assert by_id[fid].name == f"old {fid}"
    assert any("budget=smoke" in ln and "K=4" in ln for ln in lines)


def test_rollout_cap_is_hard_and_family_keeps_clustering_name():
    space, recs, fams = _four_families()
    seen = []

    def f(X, seeds):
        seen.append(len(X))
        return _and_fn()(X, seeds)
    out = mc.apply_mincause(fams, recs, space, f, k_seeds=4, max_seed_mult=2, log=None,
                            top_n=2, rollout_cap=60, budget_label="smoke")
    assert sum(seen) <= 60
    capped = [x for x in out if "rollout cap 60 reached" in (x.margin or "")]
    assert capped and all(x.name.startswith("old ") for x in capped)


def test_progress_line_per_batched_call_with_eta():
    space, recs, fams = _four_families()
    lines, calls = [], []
    t = iter(range(10_000))

    def f(X, seeds):
        calls.append(len(X))
        return _and_fn()(X, seeds)
    mc.apply_mincause(fams[:1], recs, space, f, k_seeds=4, log=lines.append,
                      clock=lambda: float(next(t)))
    prog = [ln for ln in lines if ln.startswith("  CMC family ")]
    assert len(prog) == len(calls) > 0
    first = prog[0]
    for piece in ("family 1/1", "representative", "rows", "rollouts", "elapsed",
                  "ETA pending"):
        assert piece in first
    assert "ETA ~" in prog[1]
    assert any("ddmin round" in ln for ln in prog) and any("shrink step" in ln for ln in prog)


def test_bucket_sizes_are_a_fixed_small_set_and_max_batch_default():
    assert mc.bucket_sizes(256) == [64, 128, 256]
    assert mc.bucket_sizes(1024) == [256, 512, 1024]
    assert mc.default_max_batch({}, backend="cpu") == 256
    assert mc.default_max_batch({"BP_BUDGET": "smoke"}, backend="cpu") == 64
    assert mc.default_max_batch({"BP_BUDGET": "smoke"}, backend="gpu") == 1024
    assert mc.bucket_sizes(64) == [64]
    assert mc.default_max_batch({}, backend="gpu") == 1024
    assert mc.default_max_batch({"BP_MINCAUSE_MAXBATCH": "96"}, backend="gpu") == 96
    with pytest.raises(ValueError):
        mc.default_max_batch({"BP_MINCAUSE_MAXBATCH": "0"})


def test_evaluator_fail_fn_uses_only_bucket_shapes(monkeypatch):
    """CPU stand-in for the jitted evaluate: records the batch shapes it is called with."""
    jax = pytest.importorskip("jax")
    import breakpoint_mjx.search as search

    shapes = []

    def evaluate(batch, key):
        shapes.append(int(batch.shape[0]))
        return np.asarray(batch)[:, 0]

    monkeypatch.setattr(search, "scores_from_trajectories",
                        lambda traj, th, h, obj: (np.asarray(traj) > 0, None, None))
    lines = []
    ff = mc.make_evaluator_fail_fn(evaluate, _space(), None, 10, 0.5, jax.random.PRNGKey(0),
                                   max_batch=16, log=lines.append)
    for n in (3, 5, 9, 16, 40, 7):
        X = np.zeros((n, D))
        X[::2, 0] = 1.0
        out = ff(X, np.arange(n))
        assert out.tolist() == [bool(i % 2 == 0) for i in range(n)]
    assert set(shapes) == {16}                        # max 16 < 64: one shape
    assert len(lines) == len(set(shapes))


def test_margin_says_when_final_recheck_did_not_recertify():
    r = mc.certified_minimal_cause(_rep(), NOM, LO, HI, _and_fn(), k_seeds=4)
    r.final_certified, r.k, r.n = False, 4, 8
    s = mc.margin_string(r, _space().names())
    assert "NOT re-certified" in s and "certified 4/8" not in s
