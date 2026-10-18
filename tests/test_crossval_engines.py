"""CPU-pure tests for the cross-engine agreement layer (fake engines, no sim).

Everything here is aggregation: no engine is imported and no physics runs, so this file
is part of the DEFAULT suite. The engine-specific measurements live in
``tests/test_mujoco_c_parity.py`` (``BP_RUN_CBACKEND=1``) and
``tests/test_genesis_backend.py`` (``BP_RUN_GENESIS=1``).
"""
import types

import pytest

from breakpoint_mjx.crossval_engines import (
    EngineResult, agreement_report, available_engines, replay_matrix,
)
from breakpoint_mjx.schemas import FailureRecord


def _rec(rid, fam=None):
    return FailureRecord(
        theta_vector=[0.0], theta_named={"x": 0.0}, failure_score=0.9,
        component_signals={}, rollout_signature=[0.0], seed=0, id=rid, family_id=fam,
    )


class FakeEngine:
    def __init__(self, name, scores):
        self.name = name
        self._scores = dict(scores)  # record id -> score, or "boom" to raise

    def simulate(self, record):
        s = self._scores[record.id]
        if s == "boom":
            raise RuntimeError("engine exploded")
        return s  # score_fn below is identity


IDENT = float


def test_replay_matrix_and_agreement_basic():
    recs = [_rec("a", "fam1"), _rec("b", "fam2")]
    e1 = FakeEngine("mjx", {"a": 0.9, "b": 0.9})
    e2 = FakeEngine("other", {"a": 0.9, "b": 0.1})  # disagrees on b
    m = replay_matrix(recs, [e1, e2], threshold=0.5, score_fn=IDENT)
    rep = agreement_report(m, 0.5, recs)

    assert rep.n_records == 2
    assert rep.reproduction_rate["mjx"] == 1.0
    assert rep.reproduction_rate["other"] == 0.5
    assert rep.pairwise_agreement["mjx|other"] == 0.5
    assert len(rep.disagreements) == 1
    d = rep.disagreements[0]
    assert d["id"] == "b" and d["family"] == "fam2"
    assert d["verdicts"] == {"mjx": True, "other": False}


def test_engine_error_is_captured_not_raised():
    recs = [_rec("a"), _rec("b")]
    e1 = FakeEngine("mjx", {"a": 0.9, "b": 0.9})
    e2 = FakeEngine("flaky", {"a": "boom", "b": 0.9})
    m = replay_matrix(recs, [e1, e2], threshold=0.5, score_fn=IDENT)

    cell = m["a"][1]
    assert isinstance(cell, EngineResult) and cell.error and cell.score is None

    rep = agreement_report(m, 0.5, recs)
    assert rep.error_rate["flaky"] == 0.5
    # errored cell excluded from the pair denominator: only record b counts
    assert rep.pairwise_agreement["mjx|flaky"] == 1.0
    # an errored engine never lands in disagreements
    assert all("flaky" in d["verdicts"] or d["id"] != "a" for d in rep.disagreements)


def test_agreement_empty_matrix():
    rep = agreement_report({}, 0.5)
    assert rep.n_records == 0 and rep.engines == []


def test_report_round_trips_json(tmp_path):
    recs = [_rec("a")]
    e1 = FakeEngine("mjx", {"a": 0.9})
    m = replay_matrix(recs, [e1], threshold=0.5, score_fn=IDENT)
    out = tmp_path / "report.json"
    d = agreement_report(m, 0.5, recs).to_json(str(out))
    assert out.exists() and d["n_records"] == 1


def test_genesis_backend_is_scope_gated():
    """Genesis is parity-ported for URDF targets ONLY, and refuses everything else.

    ``READY`` is now True (the measurement is in ``tests/test_genesis_backend.py``), so
    the guard that matters here is the SCOPE one: an env that is not a ``URDFEnv`` --
    every mujoco_playground target -- must raise rather than produce a hand-ported
    rollout that would silently count as a cross-engine reproduction.
    """
    from breakpoint_mjx.backends.genesis_backend import READY, GenesisBackend

    assert READY is True
    b = GenesisBackend(object(), None, target=None, space=None, consts=None, horizon=10)
    with pytest.raises(NotImplementedError):
        b.simulate(_rec("a"))


# --------------------------------------------------------------------------- #
# A 1-engine report must never read as cross-engine evidence
# --------------------------------------------------------------------------- #
def test_skipped_engines_and_reasons_are_carried_on_the_report():
    recs = [_rec("a")]
    m = replay_matrix(recs, [FakeEngine("mjx", {"a": 0.9})], 0.5, IDENT)
    rep = agreement_report(m, 0.5, recs, skipped=[("genesis", "not importable")])
    assert rep.skipped == [["genesis", "not importable"]]
    assert rep.to_json()["skipped"] == [["genesis", "not importable"]]


def test_is_cross_engine_is_false_for_one_engine():
    """The trap this flag closes: a one-engine report has an EMPTY pairwise_agreement,
    which reads exactly like 'every engine agreed'."""
    recs = [_rec("a"), _rec("b")]
    m = replay_matrix(recs, [FakeEngine("mjx", {"a": 0.9, "b": 0.1})], 0.5, IDENT)
    rep = agreement_report(m, 0.5, recs)
    assert rep.pairwise_agreement == {}       # nothing to compare...
    assert rep.is_cross_engine is False       # ...and the report SAYS so
    assert rep.to_json()["is_cross_engine"] is False


def test_is_cross_engine_needs_two_engines_that_actually_ran():
    """An engine that errored on EVERY record contributed no verdict, so it does not
    make a report cross-engine."""
    recs = [_rec("a"), _rec("b")]
    engines = [FakeEngine("mjx", {"a": 0.9, "b": 0.9}),
               FakeEngine("broken", {"a": "boom", "b": "boom"})]
    rep = agreement_report(replay_matrix(recs, engines, 0.5, IDENT), 0.5, recs)
    assert rep.error_rate["broken"] == 1.0
    assert rep.is_cross_engine is False

    engines[1] = FakeEngine("other", {"a": "boom", "b": 0.9})
    rep2 = agreement_report(replay_matrix(recs, engines, 0.5, IDENT), 0.5, recs)
    assert rep2.is_cross_engine is True


# --------------------------------------------------------------------------- #
# available_engines: every engine is either instantiated or skipped WITH A REASON
# --------------------------------------------------------------------------- #
class _FakePlaygroundEnv:
    """Stands in for a mujoco_playground env: it has the ``_get_obs`` hook the C backend
    needs, and it is not a URDFEnv, so Genesis must skip it."""

    def _get_obs(self, data, info):     # noqa: D401 - presence is the whole point
        raise AssertionError("never called in this test")


def test_available_engines_skips_genesis_for_a_playground_env(monkeypatch):
    from breakpoint_mjx.backends import mjx_backend
    from breakpoint_mjx.backends.genesis_backend import GenesisBackend
    from breakpoint_mjx.backends.mujoco_c import MujocoCBackend

    monkeypatch.setattr(GenesisBackend, "available", staticmethod(lambda: True))
    monkeypatch.setattr(MujocoCBackend, "available", staticmethod(lambda: False))
    monkeypatch.setattr(mjx_backend.MjxBackend, "__init__",
                        lambda self, *a, **k: None)

    engines, skipped = available_engines(
        _FakePlaygroundEnv(), None, None, None, None, None, 10
    )
    assert [e.name for e in engines] == ["mjx"]
    reasons = dict(skipped)
    assert "URDF" in reasons["genesis"]
    assert reasons["mujoco_c"] == "mujoco not importable"


def test_available_engines_skips_the_c_backend_for_a_urdf_env(monkeypatch):
    """URDFEnv has no ``_get_obs`` and does not route physics through ``mjx_env.step``,
    so the C backend's two parity hooks never fire. Skipping with a reason is honest;
    running it would produce one error per record and an error_rate that says nothing."""
    from breakpoint_mjx.backends import mjx_backend
    from breakpoint_mjx.backends.genesis_backend import GenesisBackend
    from breakpoint_mjx.backends.mujoco_c import MujocoCBackend

    monkeypatch.setattr(MujocoCBackend, "available", staticmethod(lambda: True))
    monkeypatch.setattr(GenesisBackend, "available", staticmethod(lambda: True))
    monkeypatch.setattr(GenesisBackend, "supports_env", staticmethod(lambda env: True))
    monkeypatch.setattr(mjx_backend.MjxBackend, "__init__",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(GenesisBackend, "__init__", lambda self, *a, **k: None)

    env = types.SimpleNamespace()          # no _get_obs -> not a Playground env
    engines, skipped = available_engines(env, None, None, None, None, None, 10)
    assert sorted(e.name for e in engines) == ["genesis", "mjx"]
    assert "mujoco_playground hooks" in dict(skipped)["mujoco_c"]


# --------------------------------------------------------------------------- #
# Independent full-collision replay of certified minimal counterexamples
# (fake backend: the aggregation, early stop and demotion logic, no physics)
# --------------------------------------------------------------------------- #
import numpy as np  # noqa: E402

from breakpoint_mjx.crossval_engines import (  # noqa: E402
    minimal_counterexample, replay_family, replay_stage, stop_after_fall,
)
from breakpoint_mjx.schemas import FailureFamily, family_status  # noqa: E402

TH = {"trunk_height_min": 0.18, "up_proj_min": 0.5}


class _Dim:
    def __init__(self, kind="model"):
        self.kind = kind


class _Space:
    dims = {"a": _Dim(), "b": _Dim()}

    def names(self):
        return ["a", "b"]

    def nominal_vector(self):
        return [1.0, 0.0]


def _fam(cause, certified=True, fid="fam-0"):
    return FailureFamily(id=fid, name="n", member_ids=["r"], representative_id="r",
                         triggering_parameters={"a": 3.0, "b": 2.0}, mean_failure_score=1.0,
                         mean_transfer_score=1.0, priority_rank=1,
                         cause_certified=certified, minimal_cause=cause)


def _summary(fall_at, T=40):
    h = np.full(T, 0.28)
    if fall_at is not None:
        h[fall_at:] = 0.05
    return {"floor_height": h, "trunk_height": h, "up_proj": np.ones(T)}


class _FakeBackend:
    """Falls at step 5 on every counterexample seed in ``fall_seeds``; never at nominal.
    Honors early_stop exactly like MujocoCBackend (truncates the summary)."""

    def __init__(self, fall_seeds):
        self.fall_seeds = set(fall_seeds)
        self.early_stop = None
        self.calls = []

    def simulate(self, rec, collect_states=False):
        nominal = rec.theta_named["a"] == 1.0 and rec.theta_named["b"] == 0.0
        fell = (not nominal) and rec.env_index in self.fall_seeds
        full = _summary(5 if fell else None)
        T = len(full["up_proj"])
        n = T
        for t in range(T):
            if self.early_stop and self.early_stop(t, {k: v[t] for k, v in full.items()}):
                n = t + 1
                break
        self.calls.append((rec.id, n))
        out = {k: v[:n] for k, v in full.items()}
        return (out, list(range(n + 1))) if collect_states else out


def _score(s):
    return 0.9 if bool((np.asarray(s["floor_height"]) < 0.18).any()) else 0.1


def test_minimal_counterexample_moves_only_certified_dims():
    ce = minimal_counterexample(_fam([{"dim": "b", "value": 0.7, "direction": "high"}]),
                                _Space())
    assert ce == {"a": 1.0, "b": 0.7}                    # a stays nominal, not 3.0
    assert minimal_counterexample(_fam(None, certified=False), _Space()) is None


def test_replay_family_reproduces_and_clips_after_fall():
    be = _FakeBackend(fall_seeds=range(8))
    rp, ce_states, twin_states = replay_family(
        be, _Space(), _fam([{"dim": "b", "value": 0.7, "direction": "high"}]), k=8,
        score_fn=_score, threshold=0.5, thresholds=TH, batch_key=[0, 1], n_after=3,
        collect=True)
    assert (rp["k"], rp["n"], rp["reproduces"]) == (8, 8, True)
    assert rp["fall_steps"] == [5] * 8
    # stopped 3 steps after the fall at step 5 -> 9 steps, not the 40-step horizon
    assert all(n == 9 for rid, n in be.calls if "nominal" not in rid)
    assert len(ce_states) == 10                          # reset state + 9 steps
    assert rp["nominal_twin"]["fell"] is False
    assert len(twin_states) == 10                        # twin cut at the same length
    assert be.early_stop is None


def test_replay_family_demotes_and_stops_once_bar_is_unreachable():
    be = _FakeBackend(fall_seeds=[0, 2])                 # seeds 1 and 3 do not fall
    rp, _, _ = replay_family(
        be, _Space(), _fam([{"dim": "b", "value": 0.7, "direction": "high"}]), k=8,
        score_fn=_score, threshold=0.5, thresholds=TH, batch_key=[0, 1], n_after=3)
    assert rp["reproduces"] is False
    assert rp["n"] < 8                                   # early exit, not all 8 seeds
    fam = _fam([{"dim": "b", "value": 0.7, "direction": "high"}])
    fam.replay = rp
    assert rp["verdict"] == "inconclusive"                # 2/4: CP interval straddles 0.5
    assert family_status(fam) == "not_confirmed_outside_mjx"


def test_stop_after_fall_honours_cap_without_a_fall():
    stop = stop_after_fall(TH, n_after=3, max_steps=4)
    s = {"floor_height": 0.28, "up_proj": 1.0}
    assert [stop(t, s) for t in range(4)] == [False, False, False, True]


def test_replay_stage_escape_hatch_and_nothing_certified():
    fams = [_fam([{"dim": "b", "value": 0.7, "direction": "high"}])]
    off = replay_stage(fams, env=None, policy_fn=None, target=None, space=_Space(),
                       consts=None, horizon=10, thresholds=TH, threshold=0.5,
                       environ={"BP_REPLAY": "0"})
    assert off == {"status": "disabled", "reason": "BP_REPLAY=0"}
    none = replay_stage([_fam(None, certified=False)], env=None, policy_fn=None,
                        target=None, space=_Space(), consts=None, horizon=10,
                        thresholds=TH, threshold=0.5, environ={})
    assert none["status"] == "skipped"


def test_full_collision_builder_enables_contacts_and_checks_layout():
    mujoco = pytest.importorskip("mujoco")
    from breakpoint_mjx.backends.mujoco_c import (
        build_full_collision_model, enable_body_floor_contacts, layout_mismatches,
    )

    xml = """<mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 .1" contype="1" conaffinity="0"/>
      <body name="trunk" pos="0 0 .3"><freejoint/>
        <geom name="torso" type="box" size=".1 .05 .05" contype="0" conaffinity="0"/>
        <geom name="foot" type="sphere" size=".02" pos=".3 0 0" contype="0" conaffinity="1"/>
        <site name="imu"/>
      </body></worldbody>
      <sensor><framepos name="p" objtype="site" objname="imu"/></sensor></mujoco>"""
    base = mujoco.MjModel.from_xml_string(xml)
    m, n = enable_body_floor_contacts(base)
    assert n == 1 and int(m.geom_conaffinity[base.geom("torso").id]) == 1
    assert int(base.geom_conaffinity[base.geom("torso").id]) == 0   # original untouched
    assert layout_mismatches(base, m) == []

    class _Env:
        mj_model = base

    full, info = build_full_collision_model(_Env(), target=None)
    assert info["source"] == "enabled_contacts" and info["n_enabled"] == 1
    # the torso box now stops on the floor instead of passing through it
    d = mujoco.MjData(full)
    for _ in range(600):
        mujoco.mj_step(full, d)
    assert d.xpos[full.body("trunk").id][2] > 0.0
    d0 = mujoco.MjData(base)
    for _ in range(600):
        mujoco.mj_step(base, d0)
    # without the torso contact only the offset foot touches, so the trunk tips and its
    # box ends up (partly) below the floor plane, the feet-only artifact
    assert d0.xpos[base.body("trunk").id][2] < d.xpos[full.body("trunk").id][2]


def test_replay_stage_runs_on_the_pristine_model_and_restores_it(monkeypatch):
    """rollout_one leaves a vmapped tracer bound to env._mjx_model after the search; the
    C replay runs the env's own reset, so the stage must swap the pristine model in (the
    measured failure: UnexpectedTracerError on every family) and put it back after."""
    import breakpoint_mjx.crossval_engines as ce

    env = types.SimpleNamespace(_mjx_model="leaked-tracer")
    seen = {}

    def fake(families, **kw):
        seen["model"] = kw["env"]._mjx_model
        return {"status": "ran"}

    monkeypatch.setattr(ce, "_replay_stage", fake)
    out = ce.replay_stage([], env=env, policy_fn=None, target=None, space=_Space(),
                          consts=None, horizon=10, thresholds=TH, threshold=0.5,
                          base_model="pristine")
    assert out == {"status": "ran"} and seen["model"] == "pristine"
    assert env._mjx_model == "leaked-tracer"


def test_replay_pools_like_cmc_when_inconclusive():
    """Two misses in 8 make 7/8 unreachable: inconclusive at K=8 (early stop). With
    BP_REPLAY_MAXMULT-style pooling the replay tops up and decides on the exact
    group-sequential boundaries (looks every K/2 = 4 seeds): reproduced at 10/12. The old
    full-alpha CP test at each wave decided later (14/16) AND over-spent alpha."""
    fam = _fam([{"dim": "b", "value": 0.7, "direction": "high"}])
    kw = dict(score_fn=_score, threshold=0.5, thresholds=TH, batch_key=[0, 1], n_after=3)
    flaky = set(range(32)) - {1, 5}
    rp1, _, _ = replay_family(_FakeBackend(flaky), _Space(), fam, k=8, **kw)
    assert (rp1["k"], rp1["n"], rp1["verdict"]) == (4, 6, "inconclusive")  # early stop
    rp4, _, _ = replay_family(_FakeBackend(flaky), _Space(), fam, k=8, max_mult=4, **kw)
    assert (rp4["k"], rp4["n"], rp4["verdict"]) == (10, 12, "reproduced")
    assert rp4["reproduces"] is True
    assert rp4["decision_rule"] == "group-sequential"
    assert rp1["decision_rule"] == "clopper-pearson"
