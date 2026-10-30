"""Evaluation envs carry no Playground-internal pushes/kicks (BreakPoint's push is a theta
dim). Regression: G1 failed the calm walk check (fall rate 0.06-0.12) only because the
default G1 config shoves the base up to 2 m/s every 5-10 s; with pushes off, 0 falls."""
from __future__ import annotations

import pytest

pytest.importorskip("mujoco_playground")

from breakpoint_mjx.targets._legged import eval_config_overrides  # noqa: E402


@pytest.mark.parametrize("env_name,key,value", [
    ("G1JoystickFlatTerrain", "push_config.enable", False),
    ("T1JoystickFlatTerrain", "push_config.enable", False),
    ("BerkeleyHumanoidJoystickFlatTerrain", "push_config.enable", False),
    ("Go1JoystickFlatTerrain", "pert_config.enable", False),
    ("BarkourJoystick", "velocity_kick", [0.0, 0.0]),
])
def test_eval_overrides_switch_off_env_perturbations(env_name, key, value):
    out = eval_config_overrides(env_name, "jax")
    assert out["impl"] == "jax"
    assert out[key] == value


def test_overrides_apply_to_the_loaded_env():
    from mujoco_playground import registry
    env = registry.load("G1JoystickFlatTerrain",
                        config_overrides=eval_config_overrides("G1JoystickFlatTerrain", "jax"))
    assert env._config.push_config.enable is False
