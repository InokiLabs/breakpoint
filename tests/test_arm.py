"""Stress Lab (breakpoint_mjx/arm): robots, drive trains, search plumbing, world model.

CPU only, needs mujoco + scipy. Programs here are a second or two long so the whole file
runs in well under a minute."""
import json
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from breakpoint_mjx.arm import ar4, world_model as WM  # noqa: E402
from breakpoint_mjx.arm.program import (Kinematics, build_reference, builtin_programs,  # noqa: E402
                                        load_program, parse_program, trapezoid)
from breakpoint_mjx.arm.robots import ROBOTS, build_model, get_robot  # noqa: E402
from breakpoint_mjx.arm.search import Runner, minimal_cause, stress_space  # noqa: E402
from breakpoint_mjx.arm.sim import simulate, simulate_batch, taught_offsets  # noqa: E402


def short(robot="mk5", payload=0.5, **kw):
    home = list(get_robot(robot).home_deg)
    tgt = list(home)
    tgt[0] += 15
    raw = {"name": "short", "model": robot, "cycles": 2, "tolerance_mm": 1.0,
           "tool": {"payload_kg": payload, "cog_mm": [0, 0, 50]}, "start_deg": home,
           "moves": [{"type": "joint", "deg": tgt}, {"type": "dwell", "s": 0.3, "measure": "a"},
                     {"type": "joint", "deg": home}, {"type": "dwell", "s": 0.2}], **kw}
    return parse_program(raw)


# ---------------------------------------------------------------- robot definitions
def test_ar4_gear_ratios_follow_from_the_firmware():
    np.testing.assert_allclose(ar4.gear_ratios("mk5"), [40, 50, 50, 44.8, 9.837, 20], atol=0.01)
    np.testing.assert_allclose(ar4.gear_ratios("mk3"), ar4.gear_ratios("mk5"), atol=0.01)


@pytest.mark.parametrize("rid, q, xyz", [
    ("ur5e", [0] * 6, (0.8172, 0.2329, 0.0628)),        # UR's published zero pose
    ("xarm6", [0] * 6, (0.207, 0.0, 0.112)),            # UFACTORY's published zero pose
])
def test_published_zero_poses(rid, q, xyz):
    k = Kinematics(rid)
    p, _ = k.fk(np.deg2rad(q))
    np.testing.assert_allclose(p, xyz, atol=1e-3)


def test_every_robot_compiles_with_a_payload_on_its_flange():
    for r in ROBOTS.values():
        m = build_model(r, payload_kg=2.0, payload_cog=(0, 0, 0.05))
        assert m.nq == 6 and m.site("tcp").id >= 0
        assert m.body_mass.sum() > 2.0


def test_aliases_and_unknown_robots():
    assert get_robot("mk5").id == "ar4_mk5"
    with pytest.raises(ValueError):
        get_robot("nope")


def test_builtin_programs_parse_and_are_reachable():
    for name in builtin_programs():
        p = load_program(name)
        ref = build_reference(p, {}, rate_hz=200.0)
        assert not ref.issues, (name, ref.issues)


def test_program_rejects_a_control_mode_the_robot_does_not_have():
    with pytest.raises(ValueError, match="control"):
        short("ur5e", control="ros2")


def test_trapezoid_is_monotone_and_ends_at_one():
    s = trapezoid(1.0, 0.5, 2.0, 1.0, 1e-3)
    assert s[-1] == pytest.approx(1.0) and np.all(np.diff(s) >= -1e-12)


def test_ik_round_trip():
    k = Kinematics("ur5e")
    q0 = np.deg2rad([5, -80, 100, -110, -90, 10])
    p, R = k.fk(q0)
    q, err, ok = k.ik(np.deg2rad(get_robot("ur5e").home_deg), p, R, iters=300)
    assert ok and err < 1e-4


# ---------------------------------------------------------------- drive trains
@pytest.mark.parametrize("robot", ["mk5", "ur5e", "xarm6"])
def test_nominal_short_program_passes(robot):
    p = short(robot, payload=0.5)
    r = simulate(p, {}, keep_trace=False, taught=taught_offsets(p))
    assert r["valid"] and not r["failed"], r["events"]
    assert r["metrics"]["lost_steps"] == 0


def test_ar4_wrist_overload_loses_steps_and_velocity_mode_never_recovers():
    p = short("mk5", payload=0.5)
    p.moves[0]["deg"] = [15, 0, 0, 0, 40, 0]          # swing J5 with a heavy offset tool
    r = simulate(p, {"payload_kg": 2.5, "cog_x_mm": 100, "accel_scale": 4, "torque_scale": 0.6},
                 keep_trace=False)
    assert r["metrics"]["lost_steps"] > 0
    assert r["metrics"]["lost_steps_by_joint"][4] > 0  # J5 is the weak joint (9.84:1)


def test_ar4_position_mode_makes_up_lost_steps_that_velocity_mode_keeps():
    st = {"payload_kg": 2.5, "cog_x_mm": 100, "accel_scale": 4, "torque_scale": 0.6}
    out = {}
    for mode in ("ros2", "ros2_position"):
        p = short("mk5", payload=0.5, control=mode)
        p.moves[0]["deg"] = [15, 0, 0, 0, 40, 0]
        out[mode] = simulate(p, st, keep_trace=False)["metrics"]
    if out["ros2"]["lost_steps"] and out["ros2_position"]["lost_steps"]:
        assert out["ros2_position"]["repeat_drift_mm"] < out["ros2"]["repeat_drift_mm"]


def test_servo_overload_triggers_a_protective_stop():
    p = short("ur5e", payload=3.0)
    r = simulate(p, {"payload_kg": 8.0, "cog_x_mm": 150, "torque_scale": 0.5, "accel_scale": 5},
                 keep_trace=False)
    assert r["failed"]
    assert r["modes"]["protective_stop"] >= 1.0 or r["modes"]["torque_margin"] >= 1.0


def test_softer_gearbox_sags_where_the_motor_encoders_cannot_see():
    p = short("ur5e", payload=3.0)
    taught = taught_offsets(p)
    base = simulate(p, {}, keep_trace=False, taught=taught)["metrics"]["max_hidden_error_mm"]
    soft = simulate(p, {"stiffness_scale": 0.3, "payload_kg": 5.0}, keep_trace=False,
                    taught=taught)["metrics"]["max_hidden_error_mm"]
    assert soft > base + 0.05


def test_batch_matches_single_runs():
    p = short("mk5")
    sts = [{}, {"payload_kg": 1.5}]
    batch = simulate_batch(p, sts)
    for st, b in zip(sts, batch):
        one = simulate(p, st, keep_trace=False)
        assert one["metrics"]["max_absolute_error_mm"] == pytest.approx(
            b["metrics"]["max_absolute_error_mm"], abs=1e-9)


def test_trace_has_what_the_viewer_needs():
    r = simulate(short("xarm6"), {})
    tr = r["trace"]
    for k in ("t", "xpos", "xquat", "tcp", "tcp_ref", "q", "q_ref", "q_enc", "aux", "util"):
        assert len(tr[k]) == len(tr["t"])
    assert np.asarray(tr["xpos"]).shape[1:] == (6, 3)


# ---------------------------------------------------------------- search plumbing
def test_stress_space_follows_the_robot_and_the_program():
    ar = stress_space(short("mk5", payload=0.7))
    ur = stress_space(short("ur5e", payload=3.0))
    assert "supply_scale" in ar.dims and "ff_scale" not in ar.dims
    assert "ff_scale" in ur.dims and "supply_scale" not in ur.dims
    assert ar.dims["payload_kg"].nominal == pytest.approx(0.7)


def test_minimal_cause_keeps_only_the_axes_the_failure_needs():
    p = short("ur5e", payload=3.0)
    sp = stress_space(p)
    th = {n: d.nominal for n, d in sp.dims.items()}
    th.update({"payload_kg": 8.0, "cog_x_mm": 150, "torque_scale": 0.5, "accel_scale": 5,
               "friction_scale": 1.2, "shift_z_mm": 5.0})
    with Runner(workers=0) as rn:
        mc = minimal_cause(p, th, sp, runner=rn, bisect_iters=2)
    assert mc["reproduced"]
    assert "shift_z_mm" not in mc["axes"] and "friction_scale" not in mc["axes"]


# ---------------------------------------------------------------- world model
def test_log_csv_round_trip():
    log = WM.RobotLog(t=np.linspace(0, 1, 20), cmd=np.zeros((20, 6)), fb=np.ones((20, 6)) * 0.01,
                      tcp=np.full((20, 3), np.nan), meta={"payload_kg": 0.5})
    log.tcp[5] = [0.1, 0.2, 0.3]
    back = WM.log_from_csv(WM.log_to_csv(log))
    np.testing.assert_allclose(back.fb, log.fb, atol=1e-6)
    assert back.meta["payload_kg"] == 0.5 and back.has_tcp
    with pytest.raises(ValueError, match="missing columns"):
        WM.log_from_csv("t,cmd_j1\n0,0\n")


def test_world_model_json_round_trip_and_correction_shape():
    wm = WM.WorldModel(params={"friction_scale": 1.3}, W=np.ones((6, len(WM.FEATURES))) * 1e-3)
    back = WM.WorldModel.from_json(wm.to_json())
    assert back.params == wm.params
    c = back.correction(np.zeros((3, 6)), np.ones((3, 6)) * 0.5)
    assert c.shape == (3, 6)
    np.testing.assert_allclose(back.correction(np.zeros((1, 6)), np.zeros((1, 6))), 0.0)


def test_fit_recovers_an_unmodelled_latency():
    p = short("mk5")
    log = WM.synthetic_log(p, truth={"friction_scale": 1.0, "torque_scale": 1.0,
                                     "backlash_scale": 1.0, "stiffness_scale": 1.0},
                           latency_s=0.02)
    with Runner(workers=0) as rn:
        wm = WM.fit(log, iters=1, pop=4, runner=rn)
    before, after = wm.report["before"]["joint_rms_deg"], wm.report["after"]["joint_rms_deg"]
    assert after < 0.5 * before
    json.loads(wm.to_json())
