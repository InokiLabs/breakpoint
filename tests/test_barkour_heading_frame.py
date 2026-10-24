"""Barkour's command frame (the "imu_frame" site) is yawed 180 deg from its torso body.

Regression: the engine read velocities in the torso body frame, so a Barkour policy
walking forward at the commanded speed scored tracking_error ~1.5 (every Barkour zoo
policy failed the walk check while tracking to ~0.14 in the env's own sensors).
"""
from __future__ import annotations

import types

import numpy as np

from breakpoint_mjx import failure
from breakpoint_mjx.objectives.legged import legged_summarize
from breakpoint_mjx.targets.barkour import BARKOUR


def _state(yaw, v_world, w_body, cmd):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    data = types.SimpleNamespace(xmat=np.stack([np.eye(3), R]).reshape(2, 9),
                                 qpos=np.array([0, 0, 0.3, 1, 0, 0, 0.0]),
                                 qvel=np.concatenate([v_world, w_body]))
    return types.SimpleNamespace(data=data, info={"command": np.asarray(cmd, float)})


def test_barkour_reads_velocity_in_its_imu_frame():
    # Torso body yawed 180 deg in the world, so the IMU (robot forward) points along +x.
    st = _state(np.pi, np.array([0.8, 0.0, 0.0]), np.array([0.0, 0.0, 0.3]), [0.8, 0.0, 0.3])
    out = BARKOUR.objective.summarize(st, (0, 1))
    np.testing.assert_allclose(out["lin_vel"], [0.8, 0.0, 0.0], atol=1e-6)
    traj = {k: np.asarray(out[k])[None] for k in ("lin_vel", "ang_vel", "command")}
    assert failure._tracking_err_series(traj)[0] < 1e-5
    # The torso-body reading (the old behavior) reverses the direction of travel.
    np.testing.assert_allclose(legged_summarize(st, (0, 1))["lin_vel"], [-0.8, 0.0, 0.0], atol=1e-6)


def test_barkour_objective_keeps_the_legged_scoring():
    from breakpoint_mjx.objectives import LEGGED_OBJECTIVE
    assert BARKOUR.objective.name == "legged"
    assert BARKOUR.objective.component_signals is LEGGED_OBJECTIVE.component_signals
    assert BARKOUR.objective.failure_score is LEGGED_OBJECTIVE.failure_score
