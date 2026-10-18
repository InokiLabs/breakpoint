"""CPU tests for the equal-budget baseline (breakpoint_mjx/equal_budget.py).

Metric functions run on hand-built inputs (numpy only). The runner is exercised against
a synthetic evaluator (latency drives a fall; no physics), which is enough to check the
budget accounting: every arm, including the directed CMA-ES arm, spends exactly B.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from breakpoint_mjx import equal_budget as eb
from breakpoint_mjx.config import ThetaDim, ThetaSpace, load_search_config, load_theta_space
from breakpoint_mjx.paths import config_path


def _space2():
    """Two dims: a in [0, 10] nominal 5 band [4, 6]; b in [0, 1] nominal 0 band [0, .5]."""
    return ThetaSpace(dims={
        "a": ThetaDim("a", "model", (0.0, 10.0), 5.0, (4.0, 6.0)),
        "b": ThetaDim("b", "rollout", (0.0, 1.0), 0.0, (0.0, 0.5)),
    })


@pytest.fixture(scope="module")
def go1_space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


# ---------------------------------------------------------------------------
# pure metrics
# ---------------------------------------------------------------------------

def test_nominal_distances_are_box_normalized():
    sp = _space2()
    l2, linf = eb.nominal_distances([[5.0, 0.0], [10.0, 0.0], [7.0, 0.4]], sp)
    np.testing.assert_allclose(l2, [0.0, 0.5, np.hypot(0.2, 0.4)])
    np.testing.assert_allclose(linf, [0.0, 0.5, 0.4])


def test_closest_failure_none_when_no_failures_and_independent_minima():
    sp = _space2()
    assert eb.closest_failure(np.zeros((0, 2)), sp) == {"l2": None, "linf": None,
                                                         "index": None}
    # row0: l2 0.5, linf 0.5 ; row1: l2 ~0.532, linf 0.4 -> minima come from different rows
    c = eb.closest_failure([[10.0, 0.0], [8.5, 0.4]], sp)
    assert c["index"] == 0 and c["l2"] == pytest.approx(0.5)
    assert c["linf"] == pytest.approx(0.4)


def test_arm_metrics_counts_on_hand_built_input():
    sp = _space2()
    X = [[5.0, 0.0], [9.9, 0.0], [1.0, 0.0], [5.5, 0.95], [5.0, 0.1]]
    scores = [0.1, 0.9, 0.8, 0.6, 0.49]           # threshold 0.5 -> rows 1, 2, 3 fail
    m = eb.arm_metrics(X, scores, sp, 0.5, horizon=100)
    assert m["evals"] == 5 and m["n_failures"] == 3
    assert m["failure_rate"] == pytest.approx(0.6)
    assert m["modes"] == ["high a", "high b", "low a"]
    assert m["n_modes"] == 3
    # every row passes (2 dims -> max_outside 4, at most 2 extreme dims allowed)
    assert m["n_in_band"] == 3
    assert m["closest_l2"] == pytest.approx(0.4)     # row 2: |1-5|/10 (row 1: 0.49)
    assert m["closest_linf"] == pytest.approx(0.4)


def test_arm_metrics_no_failures():
    m = eb.arm_metrics([[5.0, 0.0]], [0.0], _space2(), 0.5)
    assert m["n_failures"] == 0 and m["closest_l2"] is None and m["n_modes"] == 0


def test_arm_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        eb.arm_metrics([[5.0, 0.0]], [0.1, 0.2], _space2(), 0.5)


def test_in_band_uses_rules_not_transfer(go1_space):
    nom = {n: d.nominal for n, d in go1_space.dims.items()}
    assert eb.in_band(nom, go1_space, horizon=150)
    # three dims pinned at their hard extreme -> unsurvivable corner (MAX_EXTREME = 2)
    corner = dict(nom, friction_floor=0.15, latency_steps=4.0, push_mag=2.0)
    assert not eb.in_band(corner, go1_space, horizon=150)
    # two pinned is allowed even though transfer_likelihood drops
    two = dict(nom, friction_floor=0.15, latency_steps=4.0)
    assert eb.in_band(two, go1_space, horizon=150)
    # impossible: outside the hard bounds
    assert not eb.in_band(dict(nom, friction_floor=-0.1), go1_space, horizon=150)


def test_in_band_does_not_call_the_transfer_heuristic(go1_space, monkeypatch):
    import breakpoint_mjx.plausibility as pl
    import breakpoint_mjx.transfer as tr

    def boom(*a, **k):
        raise AssertionError("transfer_likelihood must not be used by the metrics")

    monkeypatch.setattr(tr, "transfer_likelihood", boom)
    monkeypatch.setattr(pl, "transfer_likelihood", boom)
    X = np.asarray([go1_space.nominal_vector()] * 3)
    m = eb.arm_metrics(X, [0.9, 0.9, 0.1], go1_space, 0.5, horizon=150)
    assert m["n_in_band"] == 2


def test_dominant_modes_direction_aware():
    sp = _space2()
    assert eb.dominant_modes([[0.0, 0.1], [5.0, 0.9], [6.0, 0.0]], sp) == [
        "low a", "high b", "high a"]
    assert eb.dominant_modes(np.zeros((0, 2)), sp) == []


def test_k_closest_failures_orders_by_l2():
    sp = _space2()
    X = [[10.0, 0.0], [6.0, 0.0], [5.0, 0.0], [0.0, 1.0]]
    got = eb.k_closest_failures(X, [0.9, 0.9, 0.1, 0.9], sp, 0.5, 2)
    np.testing.assert_allclose(got, [[6.0, 0.0], [10.0, 0.0]])
    assert eb.k_closest_failures(X, [0.0] * 4, sp, 0.5, 2).shape == (0, 2)


def test_aggregate_mean_min_max_and_missing():
    agg = eb.aggregate([{"n_failures": 2, "closest_l2": None},
                        {"n_failures": 4, "closest_l2": 0.3}],
                       keys=("n_failures", "closest_l2"))
    assert agg["n_failures"] == {"mean": 3.0, "min": 2.0, "max": 4.0, "values": [2, 4],
                                 "n_missing": 0}
    assert agg["closest_l2"]["mean"] == pytest.approx(0.3)
    assert agg["closest_l2"]["n_missing"] == 1


# ---------------------------------------------------------------------------
# budget accounting
# ---------------------------------------------------------------------------

def test_plan_directed_exact_split_and_refusal():
    p = eb.plan_directed(192, 2, 32)
    assert (p.restarts, p.popsize, p.generations, p.batch) == (2, 32, 3, 64)
    assert p.restarts * p.popsize * p.generations == 192
    with pytest.raises(ValueError, match="use 128 or 192"):
        eb.plan_directed(150, 2, 32)


HORIZON = 6


def _synthetic_evaluate(space, calls):
    import jax.numpy as jnp

    j = space.names().index("latency_steps")
    lo = np.array(space.lower())
    hi = np.array(space.upper())

    def evaluate(thetas, key):
        th = np.asarray(thetas)
        calls.append(th.shape[0])
        drive = (th[:, j] - lo[j]) / (hi[j] - lo[j])
        n = th.shape[0]
        t = np.arange(HORIZON)[None, :]
        return {
            "trunk_height": jnp.asarray(0.30 - 0.25 * drive[:, None] * (t / HORIZON)),
            "up_proj": jnp.asarray(1.0 - 0.9 * drive[:, None] * (t / HORIZON)),
            "lin_vel": jnp.zeros((n, HORIZON, 3)),
            "ang_vel": jnp.zeros((n, HORIZON, 3)),
            "command": jnp.zeros((n, HORIZON, 3)),
        }

    return evaluate


@pytest.fixture(scope="module")
def small_cfg():
    c = load_search_config(config_path("search.yaml"))
    return replace(c, popsize=4, horizon=HORIZON, generations=1)


def test_every_arm_spends_exactly_the_budget(go1_space, small_cfg):
    pytest.importorskip("jax")
    pytest.importorskip("evosax")
    from breakpoint_mjx.riskcert import OperationalDist

    plan = eb.plan_directed(24, 2, 4)              # 2 x 4 x 3 generations
    calls = []
    evaluate = _synthetic_evaluate(go1_space, calls)
    dist = OperationalDist.from_space(go1_space, {"latency_steps": {"range": [0.0, 4.0]}})
    res = eb.run_benchmark(evaluate, go1_space, small_cfg, plan, [0, 1],
                           train_dist=dist, log=None)
    for arm in eb.ARMS:
        for m in res["arms"][arm]["per_seed"]:
            assert m["evals"] == 24, arm
    # every call has the directed batch shape (one compile), and the total adds up:
    # 4 arms x 2 seeds x 24 rollouts, each in batches of 8 (lhs_boundary splits its
    # budget on batch boundaries, so it pads nothing)
    assert set(calls) == {8}
    assert sum(calls) == len(eb.ARMS) * 2 * 24


def test_directed_arm_real_cmaes_evals_equal_budget(go1_space, small_cfg):
    pytest.importorskip("jax")
    pytest.importorskip("evosax")
    calls = []
    plan = eb.plan_directed(16, 2, 4)
    X, sc, evals = eb.run_arm("directed", _synthetic_evaluate(go1_space, calls),
                              go1_space, small_cfg, plan, 0, log=None)
    assert evals == 16 and X.shape == (16, len(go1_space.dims)) and sc.shape == (16,)
    assert calls == [8, 8]                          # 2 generations x (2 restarts x 4)


def test_uniform_arm_pads_last_batch_but_counts_real_rows(go1_space, small_cfg):
    pytest.importorskip("jax")
    import jax

    calls = []
    sc, n = eb._eval_rows(_synthetic_evaluate(go1_space, calls),
                          np.tile(go1_space.nominal_vector(), (10, 1)), 8,
                          jax.random.PRNGKey(0), small_cfg, None)
    assert n == 10 and sc.shape == (10,) and calls == [8, 8]


def test_train_dr_skips_cleanly_without_a_declared_distribution(go1_space, small_cfg,
                                                                 tmp_path):
    p, reason = eb.resolve_train_dr(go1_space, environ={})
    assert p is None and "skipped" in reason
    res = eb.run_benchmark(lambda *a: None, go1_space, small_cfg, eb.plan_directed(8, 2, 4),
                           [0], arms=("train_dr",), train_dist=None, train_reason=reason,
                           log=None)
    assert res["arms"]["train_dr"] == {"skipped": reason}
    md = eb.to_markdown({"basis": _basis(), **res})
    assert "train_dr skipped" in md and "simulation only" in md

    y = tmp_path / "odd.yaml"
    y.write_text("operational:\n  latency_steps: {range: [0.0, 1.0]}\n")
    p2, src = eb.resolve_train_dr(go1_space, environ={"BP_RISK_DIST": str(y)})
    assert p2 is not None and "BP_RISK_DIST" in src


def _basis(**kw):
    b = {"label": "unit", "target": "go1", "budget": 8, "horizon": HORIZON, "seeds": [0],
         "directed_plan": "2 x 4 x 1", "policy": "synthetic", "failure_threshold": 0.5,
         "replay": False}
    b.update(kw)
    return b


def test_markdown_and_json_carry_the_basis(go1_space, small_cfg, tmp_path):
    per = [eb.arm_metrics([go1_space.nominal_vector()], [0.9], go1_space, 0.5)]
    res = {"basis": _basis(), "arms": {"uniform": {"per_seed": per,
                                                   "summary": eb.aggregate(per)}}}
    jp, mp = eb.write_outputs(res, str(tmp_path))
    md = open(mp, encoding="utf-8").read()
    assert "B=8" in md and "horizon 6" in md and "simulation only" in md
    assert "| uniform |" in md and "replay" not in md.split("|---|")[0].split("\n")[-2]
    assert '"budget": 8' in open(jp, encoding="utf-8").read()


# ---------------------------------------------------------------------------
# replay wiring (fake backend)
# ---------------------------------------------------------------------------

TH = {"trunk_height_min": 0.18, "up_proj_min": 0.5}


class _FakeBackend:
    """Falls at step 3 unless the theta is at nominal on dim 'a' (a == 5)."""

    def __init__(self):
        self.early_stop = None
        self.seen = []

    def simulate(self, rec, collect_states=False):
        self.seen.append(dict(rec.theta_named))
        fell = rec.theta_named["a"] != 5.0
        h = np.full(20, 0.28)
        if fell:
            h[3:] = 0.05
        out = {"floor_height": h, "trunk_height": h, "up_proj": np.ones(20)}
        return (out, []) if collect_states else out


def _score(s):
    return 0.9 if bool((np.asarray(s["floor_height"]) < 0.18).any()) else 0.1


def test_replay_confirmation_rebuilds_the_exact_theta_and_counts():
    sp = _space2()
    be = _FakeBackend()
    X = np.asarray([[9.0, 0.7], [5.0, 0.9]])        # first reproduces, second never falls
    out = eb.replay_confirmation(X, be, sp, k_seeds=4, score_fn=_score, threshold=0.5,
                                 thresholds=TH, batch_key=[0, 1], n_after=2)
    assert out["n"] == 2 and out["confirmed"] == 1
    assert out["rows"][0]["verdict"] == "reproduced"
    assert out["rows"][1]["reproduces"] is False
    # the replayed counterexample is the full failure theta, not a nominal-patched one
    assert {"a": 9.0, "b": 0.7} in be.seen


def test_runner_feeds_k_closest_failures_to_replay(go1_space, small_cfg):
    pytest.importorskip("jax")
    got = []

    def fake_replay(Xc):
        got.append(np.asarray(Xc))
        return {"n": len(Xc), "confirmed": len(Xc) - 1}

    res = eb.run_benchmark(_synthetic_evaluate(go1_space, []), go1_space, small_cfg,
                           eb.plan_directed(32, 2, 4), [0], arms=("uniform",),
                           replay_fn=fake_replay, replay_k=3, log=None)
    m = res["arms"]["uniform"]["per_seed"][0]
    if m["n_failures"] >= 3:
        assert got and got[0].shape[0] == 3
        assert m["replay_rate"] == pytest.approx(2 / 3)
    else:  # fewer failures than k: every failure is replayed
        assert m["replay"]["n"] == min(3, m["n_failures"])


def test_cli_exposes_baseline_subcommand():
    from breakpoint_mjx import cli

    a = cli.build_parser().parse_args(
        ["baseline", "--target", "go1", "--budget", "64", "--seeds", "2", "--preset",
         "smoke", "--out", "x"])
    assert a.func is cli.cmd_baseline
    assert (a.budget, a.seeds, a.preset, a.replay_k) == (64, 2, "smoke", 4)


def test_lhs_boundary_spends_half_on_lhs_then_refines_near_the_threshold(go1_space,
                                                                          small_cfg):
    pytest.importorskip("jax")
    pytest.importorskip("scipy")
    calls = []
    plan = eb.plan_directed(32, 2, 4)                     # batch 8
    X, sc, evals = eb.run_arm("lhs_boundary", _synthetic_evaluate(go1_space, calls),
                              go1_space, small_cfg, plan, 0, log=None)
    assert evals == 32 and X.shape == (32, len(go1_space.names()))
    lo, hi = np.asarray(go1_space.lower()), np.asarray(go1_space.upper())
    assert np.all(X >= lo - 1e-9) and np.all(X <= hi + 1e-9)
    # stage 1 is a Latin hypercube: each dim's 16 values fall in 16 distinct strata
    u = (X[:16] - lo) / (hi - lo)
    for d in range(u.shape[1]):
        assert len(set(np.floor(u[:, d] * 16).astype(int))) == 16
    # deterministic per seed
    X2, _, _ = eb.run_arm("lhs_boundary", _synthetic_evaluate(go1_space, []), go1_space,
                          small_cfg, plan, 0, log=None)
    np.testing.assert_array_equal(X, X2)
