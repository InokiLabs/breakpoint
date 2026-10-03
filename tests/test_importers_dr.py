"""init writes the policy's training gains (actuation:) and its training domain
randomization as a riskcert operational ODD (operational:). CPU-only, no jax."""
from pathlib import Path

import numpy as np
import pytest

from breakpoint_mjx.adapters.importers import (ODD_DIMS, _literal_blocks, from_isaaclab_cfg,
                                               from_rsl_rl_run, from_unitree_rl_gym)
from breakpoint_mjx.adapters.obs_spec import build_contract, load_contract
from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.paths import config_path

FIX = Path(__file__).parent / "fixtures" / "contract"
TREE = FIX / "unitree_rl_gym_tree"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("BP_HONOR_GAINS", "BP_ALLOW_GAIN_MISMATCH", "BP_TRAIN_CFG", "BP_OBS_SPEC"):
        monkeypatch.delenv(k, raising=False)


def _space():
    return load_theta_space(config_path("configs/go1/theta_space.yaml"))


def test_odd_dims_exist_in_the_go1_theta_space():
    names = set(_space().names())
    assert set(ODD_DIMS.values()) <= names


# ---- acceptance (a): both fixtures ------------------------------------------------
def test_unitree_go2_fixture_writes_actuation_and_operational():
    res = from_unitree_rl_gym(FIX / "unitree_go2_deploy.yaml")
    a = res.contract["actuation"]
    assert a == {"kp": 20.0, "kd": 0.5, "action_scale": 0.25, "source": "deploy",
                 "honor": True}
    op = res.contract["operational"]
    assert op["friction_floor"] == {"range": [0.5, 1.25]}
    assert op["torso_mass_add_kg"] == {"range": [-1.0, 2.0]}
    assert op["push_mag"] == {"range": [0.0, 1.0]}
    rows = {r["field"]: r for r in res.rows}
    assert rows["deploy.stiffness"]["level"] == "info"       # honored, not just warned
    assert "HONORED" in rows["deploy.stiffness"]["detail"]
    assert not res.fatal


def test_rsl_rl_fixture_writes_actuation_and_operational():
    res = from_rsl_rl_run(FIX / "rsl_rl_go1_flat")
    assert res.contract["actuation"]["kp"] == 25.0
    assert res.contract["actuation"]["kd"] == 0.5
    assert res.contract["actuation"]["honor"] is True
    op = res.contract["operational"]
    assert op["friction_floor"] == {"range": [0.6, 1.0]}
    assert op["torso_mass_add_kg"] == {"range": [-1.0, 3.0]}
    assert op["push_mag"]["range"] == [0.0, pytest.approx(0.7071, abs=1e-4)]
    warn = {r["field"] for r in res.rows if r["level"] == "warn"}
    assert "events.reset_base" in warn                      # unmapped key -> WARN row
    assert not res.fatal


def test_bundled_fixture_writes_native_gains_without_rows():
    res = from_rsl_rl_run(FIX / "bundled_go1_rsl_rl")
    assert res.rows == []
    assert res.contract["actuation"]["kp"] == 35.0 and res.contract["actuation"]["kd"] == 0.5


def test_operational_block_is_the_riskcert_schema():
    """The block goes straight into riskcert.OperationalDist.from_space (BP_RISK_DIST)."""
    from breakpoint_mjx.riskcert import OperationalDist

    res = from_unitree_rl_gym(FIX / "unitree_go2_deploy.yaml")
    space = _space()
    p = OperationalDist.from_space(space, res.contract["operational"])
    i = space.names().index("torso_mass_add_kg")
    assert (p.lo[i], p.hi[i]) == (-1.0, 2.0)
    j = space.names().index("friction_floor")
    assert p.hi[j] == pytest.approx(1.20)                   # clipped to the search bound
    assert p.source == "yaml"


def test_contract_round_trips_actuation_and_operational(tmp_path):
    res = from_rsl_rl_run(FIX / "rsl_rl_go1_flat")
    out = tmp_path / "c.yaml"
    res.to_yaml(out)
    c = load_contract(out)
    assert c.extras["actuation"]["kp"] == 25.0
    assert c.extras["operational"]["friction_floor"] == {"range": [0.6, 1.0]}
    again = build_contract(c.to_dict())
    assert again.extras == c.extras


# ---- legged_gym python configs: parsed with ast, never executed -------------------
def test_literal_blocks_merge_base_then_robot_and_skip_non_literals():
    base = TREE / "legged_gym" / "envs" / "base" / "legged_robot_config.py"
    go2 = TREE / "legged_gym" / "envs" / "go2" / "go2_config.py"
    b = _literal_blocks([base, go2])
    assert b["control"]["stiffness"] == {"joint": 20.0}
    assert b["domain_rand"]["randomize_base_mass"] is True
    assert b["domain_rand"]["friction_range"] == [0.5, 1.25]
    assert "not_a_literal" not in b["domain_rand"]


def test_unitree_checkout_discovery_reads_gains_and_dr_from_python():
    res = from_unitree_rl_gym(TREE / "deploy" / "deploy_mujoco" / "configs" / "go2.yaml")
    assert res.contract["actuation"]["kp"] == 20.0
    assert res.contract["actuation"]["source"] == "control"
    op = res.contract["operational"]
    assert op["torso_mass_add_kg"] == {"range": [-1.0, 3.0]}
    assert op["friction_floor"] == {"range": [0.5, 1.25]}
    warn = {r["field"] for r in res.rows if r["level"] == "warn"}
    assert "domain_rand.randomize_motor_strength" in warn   # unmapped -> WARN
    assert "legged_robot_config.py" in res.source


def test_unrandomized_mass_and_point_ranges_are_warn_rows_not_ranges():
    res = from_unitree_rl_gym({"num_actions": 12, "domain_rand": {
        "randomize_friction": True, "friction_range": [0.8, 0.8],
        "randomize_base_mass": False, "added_mass_range": [-1.0, 1.0]}})
    assert "operational" not in res.contract
    warn = {r["field"] for r in res.rows if r["level"] == "warn"}
    assert {"domain_rand.friction_range", "domain_rand.randomize_base_mass"} <= warn


def test_isaaclab_learned_actuator_and_unmapped_event_rows():
    cfg = {"scene": {"robot": {"actuators": {"legs": {
               "class_type": "isaaclab.actuators.actuator_net:ActuatorNetMLP"}}}},
           "events": {"physics_material": {
               "func": "isaaclab.envs.mdp.events:randomize_rigid_body_material",
               "params": {"static_friction_range": [0.8, 0.8]}},
               "base_external_force_torque": {
               "func": "isaaclab.envs.mdp.events:apply_external_force_torque"}},
           "observations": {"policy": {"base_ang_vel": {
               "func": "isaaclab.envs.mdp.observations:base_ang_vel"}}}}
    res = from_isaaclab_cfg(cfg)
    fields = {r["field"]: r["level"] for r in res.rows}
    assert fields["actuators.legs"] == "warn"
    assert fields["events.physics_material.static_friction_range"] == "warn"  # a point
    assert fields["events.base_external_force_torque"] == "warn"
    assert "actuation" not in res.contract                  # nothing to honor


# ---- honor:false with a >20% mismatch is FATAL, overridable -----------------------
def test_unhonored_gain_mismatch_is_fatal_unless_overridden(monkeypatch):
    monkeypatch.setenv("BP_HONOR_GAINS", "0")
    res = from_unitree_rl_gym(FIX / "unitree_go2_deploy.yaml")
    assert res.contract["actuation"]["honor"] is False
    assert [r["field"] for r in res.fatal] == ["deploy.stiffness"]
    monkeypatch.setenv("BP_ALLOW_GAIN_MISMATCH", "1")
    res = from_unitree_rl_gym(FIX / "unitree_go2_deploy.yaml")
    assert not res.fatal
    assert {r["field"]: r["level"] for r in res.rows}["deploy.stiffness"] == "warn"


def test_cli_init_writes_both_blocks(tmp_path, capsys):
    from breakpoint_mjx import cli

    out = tmp_path / "contract.yaml"
    rc = cli.main(["init", "--from", "unitree_rl_gym", str(FIX / "unitree_go2_deploy.yaml"),
                   "--out", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "HONORED" in text and "operational.torso_mass_add_kg" in text
    import yaml

    d = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert d["actuation"]["kp"] == 20.0
    assert set(d["operational"]) >= {"friction_floor", "torso_mass_add_kg"}
    assert np.isfinite(d["operational"]["push_mag"]["range"][1])
