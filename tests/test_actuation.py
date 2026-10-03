"""Actuation-faithful BYO: the contract's PD gains land on the Go1 env's mj AND MJX models,
the unhonored-mismatch gate, and the gap-check verdict. The env tests load the real
Playground Go1 on CPU (no rollouts)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from breakpoint_mjx.adapters import actuation as act
from breakpoint_mjx import check_policy as cp


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("BP_OBS_SPEC", "BP_HONOR_ACTUATION", "BP_ALLOW_GAIN_MISMATCH"):
        monkeypatch.delenv(k, raising=False)


def _contract(tmp_path, kp=20.0, kd=0.7, honor=True, name="c.yaml", extra=""):
    p = tmp_path / name
    kp_s = kp if not isinstance(kp, list) else "[" + ", ".join(map(str, kp)) + "]"
    p.write_text(f"version: 1\nrobot: go1\n{extra}obs: [base_lin_vel, base_ang_vel, "
                 f"projected_gravity, joint_pos_rel, joint_vel, last_action, commands]\n"
                 f"actuation: {{kp: {kp_s}, kd: {kd}, source: test, honor: {str(honor).lower()}}}\n",
                 encoding="utf-8")
    return p


# ---- pure helpers -------------------------------------------------------------------
def test_block_collapses_uniform_lists_and_keeps_per_joint():
    b = act.actuation_block([20.0] * 12, [0.5] * 12, 0.25, "deploy")
    assert b == {"kp": 20.0, "kd": 0.5, "action_scale": 0.25, "source": "deploy",
                 "honor": True}
    b = act.actuation_block([20.0] * 6 + [25.0] * 6, 0.5)
    assert isinstance(b["kp"], list) and len(b["kp"]) == 12


def test_parse_validates():
    assert act.parse_actuation({"obs": []}) is None
    with pytest.raises(ValueError):
        act.parse_actuation({"actuation": {"kd": 0.5}})
    with pytest.raises(ValueError):
        act.parse_actuation({"actuation": {"kp": -1.0}})
    assert act.parse_actuation({"actuation": {"kp": 20}})["honor"] is True


def test_resolve_gains_permutes_policy_order_to_actuator_order():
    kp_pol = np.arange(12, dtype=float)
    perm = list(range(11, -1, -1))           # policy joint j drives actuator 11 - j
    kp, kd = act.resolve_gains({"kp": kp_pol.tolist(), "kd": 0.5}, 12, perm=perm)
    np.testing.assert_array_equal(kp, kp_pol[::-1])
    np.testing.assert_array_equal(kd, np.full(12, 0.5))


def test_gain_gate_levels():
    honored = {"kp": 20.0, "kd": 0.5, "honor": True}
    assert act.gain_gate_row(honored, 35.0, 0.5)[0] == "info"
    unhonored = dict(honored, honor=False)
    assert act.gain_gate_row(unhonored, 35.0, 0.5)[0] == "fatal"          # 43% apart
    assert act.gain_gate_row(unhonored, 35.0, 0.5, allow=True)[0] == "warn"
    assert act.gain_gate_row({"kp": 33.0, "kd": 0.5, "honor": False}, 35.0, 0.5)[0] == "warn"
    assert act.gain_gate_row({"kp": 35.0, "kd": 0.5, "honor": False}, 35.0, 0.5) is None


def test_check_policy_refuses_unhonored_mismatch(tmp_path, monkeypatch, capsys):
    c = _contract(tmp_path, honor=False)
    ran = []
    monkeypatch.setattr(cp, "run_check", lambda *a, **k: ran.append(1) or {})
    args = SimpleNamespace(robot="go1", policy="params:x.npz", n=4, horizon=10, seed=0,
                           obs_spec=str(c), infer_contract=False)
    assert cp.check(args) == 2 and not ran
    assert "FATAL" in capsys.readouterr().out


def test_check_policy_override_lets_it_through(tmp_path, monkeypatch):
    c = _contract(tmp_path, honor=False)
    d = cp.diagnose([{"fall": 0.0, "time_to_failure": 10}] * 2, np.zeros((2, 10, 3)),
                    np.full((2, 10, 2), 0.3), {}, 10)
    d["seconds"] = 0.0
    monkeypatch.setattr(cp, "run_check", lambda *a, **k: d)
    args = SimpleNamespace(robot="go1", policy="params:x.npz", n=2, horizon=10, seed=0,
                           obs_spec=str(c), infer_contract=False, allow_gain_mismatch=True)
    assert cp.check(args) == 0


def _d(fall_rate, track, passed, n=32):
    return {"fall_rate": fall_rate, "tracking_error": track, "passed": passed, "n": n}


def test_gap_verdicts():
    g = cp.gap_verdict(_d(0.0, 0.20, True), _d(0.0, 0.21, True))
    assert g["verdict"] == "no nominal shift" and not g["refuse"]
    g = cp.gap_verdict(_d(0.0, 0.20, True), _d(0.0, 0.40, True))
    assert g["verdict"] == "NOMINAL SHIFT" and not g["refuse"]
    g = cp.gap_verdict(_d(0.0, 0.20, True), _d(0.9, 0.20, False))
    assert g["refuse"] and "walks at its contract's" in g["message"]
    g = cp.gap_verdict(_d(0.9, 0.2, False), _d(0.8, 0.2, False))
    assert g["verdict"] == "WALKS AT NEITHER" and g["refuse"]
    text = cp.format_gap(cp.gap_verdict(_d(0.0, 0.2, True), _d(0.5, 0.2, False)),
                         "kp 25 kd 0.5", "kp 35 kd 0.5")
    assert "survival 1.000 (32/32)" in text and "survival 0.500 (16/32)" in text


def test_carry_extras_into_an_inferred_contract(tmp_path):
    prior = _contract(tmp_path, kp=25.0, kd=0.5)
    out = tmp_path / "inferred.yaml"
    out.write_text("version: 1\nrobot: go1\nobs: [joint_vel]\n", encoding="utf-8")
    cp._carry_extras(str(prior), out)
    assert act.parse_actuation(out)["kp"] == 25.0


# ---- the real Go1 env (CPU; loads the model, runs no rollout) -------------------------
def _go1_env():
    pytest.importorskip("mujoco_playground")
    from breakpoint_mjx.targets.go1 import load_env

    return load_env()


def _gains(env):
    mj, mx = env.mj_model, env.mjx_model
    dofs = act._actuator_dofs(mj)
    return {"mj_gain": np.asarray(mj.actuator_gainprm[:, 0]),
            "mj_bias": np.asarray(mj.actuator_biasprm[:, :3]),
            "mj_damp": np.asarray(mj.dof_damping)[dofs],
            "mx_gain": np.asarray(mx.actuator_gainprm)[:, 0],
            "mx_bias": np.asarray(mx.actuator_biasprm)[:, :3],
            "mx_damp": np.asarray(mx.dof_damping)[dofs]}


def test_honor_true_sets_mj_and_mjx_gains(tmp_path, monkeypatch):
    c = _contract(tmp_path, kp=20.0, kd=0.7)
    monkeypatch.setenv("BP_OBS_SPEC", str(c))
    env = _go1_env()
    g = _gains(env)
    for side in ("mj", "mx"):
        np.testing.assert_allclose(g[f"{side}_gain"], 20.0, rtol=1e-6)
        np.testing.assert_allclose(g[f"{side}_bias"][:, 1], -20.0, rtol=1e-6)
        # Playground Go1 keeps Kd in dof_damping (biasprm[:, 2] == 0): kd goes there only
        np.testing.assert_allclose(g[f"{side}_bias"][:, 2], 0.0)
        np.testing.assert_allclose(g[f"{side}_damp"], 0.7, rtol=1e-6)
    assert env._bp_actuation["native_kp"] == [35.0] * 12
    assert env._bp_actuation["kd_site"] == "dof_damping"


def test_per_joint_gains_follow_the_contract_joint_order(tmp_path, monkeypatch):
    from breakpoint_mjx.adapters.obs_spec import GO1_INTERFACE

    pol_names = list(reversed(GO1_INTERFACE.joint_names))
    kp = [float(10 + i) for i in range(12)]              # policy order
    c = _contract(tmp_path, kp=kp, kd=0.5,
                  extra=f"joint_names: [{', '.join(pol_names)}]\n")
    monkeypatch.setenv("BP_OBS_SPEC", str(c))
    g = _gains(_go1_env())
    np.testing.assert_allclose(g["mx_gain"], kp[::-1])    # actuator order


def test_honor_false_and_global_off_keep_native(tmp_path, monkeypatch):
    monkeypatch.setenv("BP_OBS_SPEC", str(_contract(tmp_path, honor=False)))
    g = _gains(_go1_env())
    np.testing.assert_allclose(g["mx_gain"], 35.0)
    np.testing.assert_allclose(g["mx_damp"], 0.5)
    monkeypatch.setenv("BP_OBS_SPEC", str(_contract(tmp_path, honor=True, name="d.yaml")))
    monkeypatch.setenv("BP_HONOR_ACTUATION", "0")
    np.testing.assert_allclose(_gains(_go1_env())["mx_gain"], 35.0)


def test_check_policy_load_env_modes(tmp_path):
    pytest.importorskip("mujoco_playground")
    from breakpoint_mjx.targets import get_target

    c = str(_contract(tmp_path, kp=25.0, kd=0.5, honor=False))
    t = get_target("go1")
    np.testing.assert_allclose(_gains(cp._load_env(t, c))["mx_gain"], 35.0)   # honor false
    np.testing.assert_allclose(_gains(cp._load_env(t, c, mode="force"))["mx_gain"], 25.0)
    np.testing.assert_allclose(_gains(cp._load_env(t, c, mode="native"))["mx_gain"], 35.0)


def test_kp_scale_multiplies_the_honored_gains(tmp_path, monkeypatch):
    """The ThetaSpace kp_scale multiplier now scales the POLICY's gains."""
    jnp = pytest.importorskip("jax").numpy
    from breakpoint_mjx.targets._legged import build_legged_fields
    from breakpoint_mjx.config import load_theta_space
    from breakpoint_mjx.paths import config_path

    monkeypatch.setenv("BP_OBS_SPEC", str(_contract(tmp_path, kp=20.0, kd=0.5)))
    env = _go1_env()
    from breakpoint_mjx.targets.go1 import resolve_constants

    consts = resolve_constants(env)
    space = load_theta_space(config_path("configs/go1/theta_space.yaml"))
    scalars = {n: jnp.asarray(space.dims[n].nominal) for n in space.names()}
    scalars["kp_scale"] = jnp.asarray(1.5)
    f = build_legged_fields(scalars, env.mjx_model, consts)
    np.testing.assert_allclose(np.asarray(f["actuator_gainprm"])[:, 0], 30.0, rtol=1e-6)
