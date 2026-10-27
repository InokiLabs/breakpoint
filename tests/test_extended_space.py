"""Extended theta axes: opt-in via BP_THETA_SPACE=extended, no-ops when undeclared."""
import copy

import numpy as np
import pytest
import yaml

from breakpoint_mjx.backends.mujoco_c import _apply_extended_numpy
from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.paths import config_path
from breakpoint_mjx.targets._legged import (
    EXTENDED_MODEL_DIMS,
    actuator_strength,
    fault_index,
)

ROBOTS = ("go1", "barkour", "g1", "legged_default")


@pytest.mark.parametrize("robot", ROBOTS)
def test_extended_space_is_the_default_plus_new_axes(robot, monkeypatch):
    base = load_theta_space(config_path(f"configs/{robot}/theta_space.yaml"))
    monkeypatch.setenv("BP_THETA_SPACE", "extended")
    ext = load_theta_space(config_path(f"configs/{robot}/theta_space.yaml"))
    assert ext.names()[:len(base.names())] == base.names()
    assert set(EXTENDED_MODEL_DIMS) <= set(ext.names())
    assert {"action_noise_scale", "wind_force_n"} <= set(ext.names())
    for d in ext.dims.values():
        lo, hi = d.bounds
        assert lo <= d.plausible[0] <= d.nominal <= d.plausible[1] <= hi


def test_unknown_variant_is_an_error_not_a_fallback(monkeypatch):
    monkeypatch.setenv("BP_THETA_SPACE", "nope")
    with pytest.raises(FileNotFoundError):
        config_path("configs/go1/theta_space.yaml")
    assert config_path("search.yaml").name == "search.yaml"


def test_fault_index_spans_every_motor():
    assert fault_index(np, {"fault_motor_frac": 0.0}, 12) == 0
    assert fault_index(np, {"fault_motor_frac": 1.0}, 29) == 28
    s = actuator_strength(np, {"motor_strength_scale": 0.8, "fault_motor_frac": 0.5,
                               "fault_strength": 0.5}, 12)
    np.testing.assert_allclose(s, np.where(np.arange(12) == 6, 0.4, 0.8))


class _M:
    def __init__(self, nu=4):
        self.actuator_gainprm = np.ones((nu, 3)) * 30
        self.actuator_biasprm = np.tile([0.0, -30.0, -1.0], (nu, 1))
        self.actuator_forcerange = np.tile([-20.0, 20.0], (nu, 1))
        self.actuator_trnid = np.stack([np.arange(1, nu + 1), np.zeros(nu, int)], 1)
        self.jnt_actfrcrange = np.tile([-50.0, 50.0], (nu + 1, 1))
        self.dof_damping = np.ones(6 + nu)
        self.geom_solref = np.tile([0.02, 1.0], (3, 1))
        self.pair_solref = np.tile([0.02, 1.0], (2, 1))


def test_numpy_mirror_no_op_without_dims_and_scales_with_them():
    m = _M()
    ref = copy.deepcopy(m)
    _apply_extended_numpy(m, {"friction_floor": 0.5}, 0, np.array([1]))
    for f in vars(ref):
        np.testing.assert_array_equal(getattr(m, f), getattr(ref, f))
    _apply_extended_numpy(m, {"motor_strength_scale": 0.5, "fault_motor_frac": 1.0,
                              "fault_strength": 0.5, "joint_damping_scale": 2.0,
                              "floor_solref_timeconst": 0.05}, 0, np.array([1]))
    np.testing.assert_allclose(m.actuator_forcerange[:, 1], [10, 10, 10, 5])
    np.testing.assert_allclose(m.jnt_actfrcrange[:, 1], [50, 25, 25, 25, 12.5])
    np.testing.assert_allclose(m.actuator_gainprm[:, 0], [30, 30, 30, 15])
    np.testing.assert_allclose(m.dof_damping, [1] * 6 + [2] * 4)
    assert m.geom_solref[0, 0] == 0.05 and m.geom_solref[1, 0] == 0.02
    assert m.pair_solref[1, 0] == 0.05 and m.pair_solref[0, 0] == 0.02


def test_extended_files_ship_as_package_data():
    import tomllib
    from pathlib import Path

    pp = tomllib.loads(Path(__file__).parents[1].joinpath("pyproject.toml").read_text())
    globs = pp["tool"]["setuptools"]["package-data"]["breakpoint_mjx"]
    assert "configs/**/*.yaml" in globs
    yaml.safe_load(config_path("configs/go1/theta_space_extended.yaml").read_text())
