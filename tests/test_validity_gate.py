"""Physical-validity gate + floor-relative fall metric (CPU, synthetic trajectories).

Every trajectory here is SYNTHETIC (hand-built numpy arrays); the measured CPU numbers
live in docs/validity.md / docs/validity_probe.json.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from breakpoint_mjx import failure
from breakpoint_mjx.objectives.legged import (
    LEGGED_CONTRACT_KEYS,
    LEGGED_CORE_KEYS,
    NO_CONTACT_DIST,
    legged_summarize,
)
from breakpoint_mjx.report import envelope_is_zero_volume, render_html
from breakpoint_mjx.stats import compute_report_stats, sim_validity_counts

THR = {"trunk_height_min": 0.18, "up_proj_min": 0.5, "tracking_err_max": 0.5,
       "ang_vel_max": 8.0}
T = 60


def _traj(h=0.28, up=1.0, validity=True, fh=None, pen=-0.015, qacc=2000.0, seed=0):
    rng = np.random.default_rng(seed)
    cmd = np.tile([0.5, 0.0, 0.1], (T, 1))
    tr = {
        "trunk_height": np.broadcast_to(np.asarray(h, float), (T,)).copy(),
        "up_proj": np.broadcast_to(np.asarray(up, float), (T,)).copy(),
        "lin_vel": cmd + 0.05 * rng.standard_normal((T, 3)),
        "ang_vel": 0.2 * rng.standard_normal((T, 3)),
        "command": cmd,
    }
    if validity:
        tr["floor_height"] = (tr["trunk_height"].copy() if fh is None
                              else np.broadcast_to(np.asarray(fh, float), (T,)).copy())
        tr["penetration"] = np.broadcast_to(np.asarray(pen, float), (T,)).copy()
        tr["qacc_max"] = np.broadcast_to(np.asarray(qacc, float), (T,)).copy()
    return tr


def _score(tr, thr=THR):
    sig = failure.component_signals(tr, thr)
    return sig, failure.failure_score(sig, T)


# ---------------------------------------------------------------- the gate itself ----
def test_nominal_traj_is_valid_and_not_a_failure():
    sig, s = _score(_traj())
    assert sig["sim_invalid"] == 0.0 and sig["sim_invalid_reason"] == 0.0
    assert sig["fall"] == 0.0 and s < 0.5


def test_below_floor_while_upright_is_sim_invalid_and_scores_zero():
    h = np.full(T, 0.28)
    h[30:] = -0.2                     # torso sinks through the floor while still upright
    sig, s = _score(_traj(h=h))
    assert failure.is_sim_invalid(sig)
    assert "below_floor" in failure.sim_invalid_reasons(sig)
    assert s == 0.0


def test_real_fall_then_sink_stays_a_failure_by_default():
    """Measured Go1 pattern (docs/validity.md): tip over first, THEN the collision-less
    trunk sinks below the floor. That is a real fall, not an artifact."""
    up = np.ones(T); up[20:] = -0.6     # tip-over at step 20
    h = np.full(T, 0.28); h[22:] = 0.1; h[27:] = -0.3   # sink afterwards
    sig, s = _score(_traj(h=h, up=up))
    assert not failure.is_sim_invalid(sig)
    assert sig["fall"] == 1.0 and s > 0.5
    # The literal whole-rollout rule discards it (opt-in only).
    wsig, ws = _score(_traj(h=h, up=up), {**THR, "validity_whole_rollout": 1.0})
    assert failure.is_sim_invalid(wsig) and ws == 0.0
    assert "below_floor" in failure.sim_invalid_reasons(wsig)


def test_tilted_floor_upright_walk_is_not_a_fall():
    """Walking downhill on floor_slope_rad=+0.1: world z drops below trunk_height_min,
    but the height along the floor normal stays at standing height."""
    z = np.linspace(0.28, -0.10, T)    # world-frame qpos[2] drifts downhill
    tr = _traj(h=z, fh=0.27)
    sig, s = _score(tr)
    assert sig["fall"] == 0.0 and not failure.is_sim_invalid(sig) and s < 0.5
    # Pre-gate behaviour (no floor_height): the same walk read as a fall.
    legacy = {k: v for k, v in tr.items() if k not in failure.VALIDITY_KEYS}
    lsig, ls = _score(legacy)
    assert lsig["fall"] == 1.0 and ls > 0.5


def test_nan_is_sim_invalid():
    h = np.full(T, 0.28); h[10] = np.nan
    sig, s = _score(_traj(h=h))
    assert failure.is_sim_invalid(sig) and "nonfinite" in failure.sim_invalid_reasons(sig)
    assert s == 0.0
    qa = np.full(T, 2000.0); qa[5] = np.inf
    sig2, s2 = _score(_traj(qacc=qa))
    assert failure.is_sim_invalid(sig2) and s2 == 0.0


def test_penetration_gate_skips_the_spawn_settle_window():
    pen = np.full(T, -0.015); pen[2:4] = -0.06          # tilted-floor spawn transient
    sig, _ = _score(_traj(pen=pen))
    assert not failure.is_sim_invalid(sig)
    pen2 = np.full(T, -0.015); pen2[30] = -0.06          # mid-rollout deep penetration
    sig2, s2 = _score(_traj(pen=pen2))
    assert failure.sim_invalid_reasons(sig2) == ["penetration"] and s2 == 0.0
    # Threshold is configurable.
    sig3, _ = _score(_traj(pen=pen2), {**THR, "penetration_max": 0.1})
    assert not failure.is_sim_invalid(sig3)


def test_qacc_blowup_is_sim_invalid():
    qa = np.full(T, 2000.0); qa[40] = 5e5
    sig, s = _score(_traj(qacc=qa))
    assert failure.sim_invalid_reasons(sig) == ["blowup"] and s == 0.0


def test_artifact_after_the_fall_does_not_invalidate():
    up = np.ones(T); up[15:] = -0.9
    qa = np.full(T, 2000.0); qa[50] = 1e6               # blow-up long after the tip-over
    sig, s = _score(_traj(up=up, qacc=qa))
    assert not failure.is_sim_invalid(sig) and s > 0.5


# ---------------------------------------------------- backward-compat regression ----
def _old_component_signals(traj, thresholds):
    """Verbatim copy of the pre-gate failure.component_signals (commit 3768662)."""
    raw_h = np.asarray(traj["trunk_height"]); raw_up = np.asarray(traj["up_proj"])
    h = np.nan_to_num(raw_h, nan=-1.0, posinf=-1.0, neginf=-1.0)
    up = np.nan_to_num(raw_up, nan=-1.0, posinf=-1.0, neginf=-1.0)
    ang = np.nan_to_num(np.asarray(traj["ang_vel"]), nan=0.0, posinf=0.0, neginf=0.0)
    n = len(h)
    nonfinite = ~np.isfinite(raw_h) | ~np.isfinite(raw_up)
    fallen = (h < thresholds["trunk_height_min"]) | (up < thresholds["up_proj_min"]) | nonfinite
    fall = float(fallen.any())
    ttf = float(np.argmax(fallen)) if fall else float(n)
    tracking_error = float(failure._tracking_err_series(traj).mean())
    ang_norm = np.linalg.norm(ang, axis=1)
    instability = float((ang_norm > thresholds["ang_vel_max"]).mean())
    return {"fall": fall, "time_to_failure": ttf, "tracking_error": tracking_error,
            "instability": instability, "min_height": float(h.min())}


def _old_failure_score(sig, horizon=500):
    score = 0.7 * sig["fall"]
    score += 0.3 * min(sig["tracking_error"] / 2.0, 1.0)
    score += 0.2 * sig["instability"]
    if sig["fall"]:
        score += 0.2 * (1.0 - sig["time_to_failure"] / horizon)
    return float(score)


@pytest.mark.parametrize("seed", range(12))
def test_missing_validity_keys_score_byte_identically(seed):
    rng = np.random.default_rng(seed)
    h = 0.28 + 0.1 * rng.standard_normal(T)
    up = np.clip(1.0 - 0.6 * rng.random(T), -1, 1)
    if seed % 3 == 0:
        h[rng.integers(T)] = np.nan
    if seed % 4 == 1:
        h[:] = -0.4                                   # would be "below floor" if gated
    tr = _traj(h=h, up=up, validity=False, seed=seed)
    tr["ang_vel"] = 5.0 * rng.standard_normal((T, 3))
    new = failure.component_signals(tr, THR)
    old = _old_component_signals(tr, THR)
    for k, v in old.items():
        assert new[k] == v
    assert new["sim_invalid"] == 0.0
    assert failure.failure_score(new, T) == _old_failure_score(old, T)
    assert failure.behavior_descriptor(tr, THR, T) == failure.behavior_descriptor(tr, THR, T, sig=new)
    assert failure.rollout_signature(tr) == failure.rollout_signature(dict(tr))


def test_signals_stay_float_for_record_serialization():
    sig = failure.component_signals(_traj(h=-0.3), THR)
    assert all(isinstance(float(v), float) for v in sig.values())


# ------------------------------------------------------------ legged_summarize ----
def _fake_state(tilt=0.0, contacts=(-0.01, 0.2), qacc=(1.0, -300.0, 7.0), impl=True):
    torso, floor = 1, 0
    xmat = np.zeros((2, 9)); xmat[torso] = np.eye(3).reshape(9)
    c, s = np.cos(tilt), np.sin(tilt)
    Rf = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])          # pitch about y
    geom_xmat = np.zeros((1, 9)); geom_xmat[floor] = Rf.reshape(9)
    xpos = np.array([[0, 0, 0], [1.0, 0.0, 0.3]])
    con = types.SimpleNamespace(dist=np.asarray(contacts, float))
    data = types.SimpleNamespace(
        xmat=xmat, qpos=np.array([1.0, 0, 0.3, 1, 0, 0, 0]), qvel=np.zeros(6),
        xpos=xpos, geom_xpos=np.zeros((1, 3)), geom_xmat=geom_xmat,
        qacc=np.asarray(qacc, float),
    )
    if impl:
        data._impl = types.SimpleNamespace(contact=con)
    else:
        data.contact = con
    return types.SimpleNamespace(data=data, info={"command": np.zeros(3)}), (floor, torso)


def test_legged_summarize_adds_floor_relative_validity_keys():
    st, consts = _fake_state(tilt=0.0)
    out = legged_summarize(st, consts)
    assert tuple(out) == LEGGED_CONTRACT_KEYS
    assert out["trunk_height"] == pytest.approx(0.3)
    assert out["floor_height"] == pytest.approx(0.3)
    assert out["penetration"] == pytest.approx(-0.01)
    assert out["qacc_max"] == pytest.approx(300.0)
    # Tilted floor: height along the floor normal, not world z.
    st2, _ = _fake_state(tilt=0.1)
    n = np.array([np.sin(0.1), 0.0, np.cos(0.1)])
    assert legged_summarize(st2, consts)["floor_height"] == pytest.approx(
        float(np.dot([1.0, 0.0, 0.3], n)))
    # Old-style data.contact also works; empty contacts -> the no-contact fill.
    st3, _ = _fake_state(contacts=(), impl=False)
    assert legged_summarize(st3, consts)["penetration"] == pytest.approx(NO_CONTACT_DIST)


def test_legged_summarize_duck_state_keeps_the_five_core_keys():
    torso = 1
    xmat = np.zeros((2, 9)); xmat[torso] = np.eye(3).reshape(9)
    st = types.SimpleNamespace(
        data=types.SimpleNamespace(xmat=xmat, qpos=np.zeros(7), qvel=np.zeros(6)),
        info={"command": np.zeros(3)})
    assert tuple(legged_summarize(st, (0, torso))) == LEGGED_CORE_KEYS


# ---------------------------------------------------------------- stats + report ----
def _rec(sig):
    return {"component_signals": sig}


def test_stats_exclude_sim_invalid_from_denominators():
    from tests.test_report import _make_report

    rep = _make_report()                                       # 5 family members total
    ok = {"fall": 1.0, "sim_invalid": 0.0, "sim_invalid_reason": 0.0}
    bad_floor = {"fall": 0.0, "sim_invalid": 1.0, "sim_invalid_reason": 2.0}
    bad_both = {"fall": 0.0, "sim_invalid": 1.0, "sim_invalid_reason": 12.0}
    records = [_rec(ok)] * 90 + [_rec(bad_floor)] * 7 + [_rec(bad_both)] * 3
    v = sim_validity_counts(records)
    assert v == {"n_evaluated": 100, "n_sim_invalid": 10, "n_valid": 90,
                 "reasons": {"nonfinite": 0, "below_floor": 7, "penetration": 3,
                             "blowup": 3},
                 "gate_active": True}
    stats = compute_report_stats(rep.families, n_total=100, n_failures=9,
                                 threshold=0.5, records=records)
    assert stats["n_sim_invalid"] == 10 and stats["n_valid"] == 90
    assert stats["overall"]["n"] == 90
    assert stats["overall"]["rate"] == pytest.approx(9 / 90)
    by_id = {d["family_id"]: d for d in stats["ranking"]}
    assert by_id["f0"]["probability"] == pytest.approx(3 / 90)
    # Without records: unchanged historical math, no validity block.
    plain = compute_report_stats(rep.families, n_total=100, n_failures=9)
    assert plain["overall"]["n"] == 100 and "validity" not in plain
    # Records from a run without the gate: nothing excluded, gate reported inactive.
    old = compute_report_stats(rep.families, n_total=3, n_failures=1,
                               records=[_rec({"fall": 1.0})] * 3)
    assert old["validity"]["gate_active"] is False and old["overall"]["n"] == 3


def test_report_renders_validity_line_relabel_and_zero_envelope():
    from tests.test_report import _make_report

    rep = _make_report()
    records = ([_rec({"sim_invalid": 0.0, "sim_invalid_reason": 0.0})] * 95
               + [_rec({"sim_invalid": 1.0, "sim_invalid_reason": 2.0})] * 4
               + [_rec({"sim_invalid": 1.0, "sim_invalid_reason": 8.0})])
    stats = compute_report_stats(rep.families, n_total=100, n_failures=5,
                                 threshold=0.5, records=records)
    zero_env = {"envelope": {"friction_floor": [0.8, 0.8], "kp_scale": [1.0, 1.0]},
                "summary": "safe while friction_floor in [0.8, 0.8]",
                "safe_volume_fraction": 0.0}
    assert envelope_is_zero_volume(zero_env)
    html = render_html(rep, stats=stats, envelope=zero_env)
    assert ("5 of 100 evaluated worlds were excluded as simulator artifacts "
            "(penetration / below-floor / blow-up)") in html
    assert "torso below the floor plane 4" in html and "acceleration blow-up 1" in html
    assert "over 95 physically valid evaluated envs" in html
    assert "Share of search-selected failures:</b> 0.600 (3 of 5)" in html
    assert "Failure rate (probability)" not in html
    assert "No failure-free box found around nominal" in html
    assert "[0.8, 0.8]" not in html
    # Physical-validity line sits at the top (before the families).
    assert html.index("Physical validity") < html.index("f1.mp4")
    # A real envelope is printed.
    env = {"envelope": {"friction_floor": [0.5, 1.2]}, "summary": "safe while friction_floor in [0.5, 1.2]",
           "safe_volume_fraction": 0.4}
    assert not envelope_is_zero_volume(env)
    assert "safe while friction_floor in [0.5, 1.2]" in render_html(rep, stats=stats, envelope=env)


def test_report_says_when_the_gate_was_not_active():
    from tests.test_report import _make_report

    rep = _make_report()
    stats = compute_report_stats(rep.families, n_total=3, n_failures=1,
                                 records=[_rec({"fall": 1.0})] * 3)
    assert "gate not active for this run" in render_html(rep, stats=stats)


def test_envelope_summary_names_a_zero_volume_box():
    from breakpoint_mjx.envelope import _summarize

    lo = np.array([0.1, 0.5]); hi = np.array([1.5, 1.5])
    box = np.array([0.8, 1.0])
    assert _summarize(["a", "b"], box, box, lo, hi) == "no failure-free box found around nominal"
    assert _summarize(["a", "b"], np.array([0.5, 0.5]), np.array([1.2, 1.5]), lo, hi) \
        == "safe while a in [0.5, 1.2]"


def test_write_report_picks_up_validity_from_the_runs_failures_json(tmp_path):
    """run_pipeline writes all evaluated records to failures.json beside report.html; the
    report attaches the validity counts from it when the record count matches."""
    import json

    from breakpoint_mjx.report import write_report
    from tests.test_report import _make_report

    rep = _make_report()
    recs = ([{"component_signals": {"sim_invalid": 0.0, "sim_invalid_reason": 0.0}}] * 18
            + [{"component_signals": {"sim_invalid": 1.0, "sim_invalid_reason": 4.0}}] * 2)
    (tmp_path / "failures.json").write_text(json.dumps(recs))
    stats = compute_report_stats(rep.families, n_total=20, n_failures=5, threshold=0.5)
    out = write_report(rep, str(tmp_path / "report.html"), stats=stats, embed=False)
    html = open(out, encoding="utf-8").read()
    assert "2 of 20 evaluated worlds were excluded as simulator artifacts" in html
    assert "contact penetration 2" in html and "over 18 physically valid" in html
    # Mismatched count (a different run's file): left alone.
    stats2 = compute_report_stats(rep.families, n_total=21, n_failures=5)
    out2 = write_report(rep, str(tmp_path / "report2.html"), stats=stats2, embed=False)
    assert "Physical validity" not in open(out2, encoding="utf-8").read()
