"""Unit tests for breakpoint_mjx.search.scores_from_trajectories -- pure numpy, no JAX/GPU."""
from pathlib import Path

import numpy as np
import pytest

from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.search import _records, scores_from_trajectories
from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]

THRESHOLDS = dict(
    trunk_height_min=0.18,
    up_proj_min=0.5,
    tracking_err_max=0.5,
    ang_vel_max=8.0,
)

HORIZON = 50
N = 2
T = 50


def _batched_traj():
    """Build a synthetic batched trajectory with N=2, T=50.

    Env 0: nominal (height 0.28, up_proj 1.0) -- should score LOW.
    Env 1: fallen  (height 0.05, up_proj 0.1) -- should score HIGH.

    Shapes:
      trunk_height : (2, 50)
      up_proj      : (2, 50)
      lin_vel      : (2, 50, 3)
      ang_vel      : (2, 50, 3)
      command      : (2, 50, 3)
    """
    trunk_height = np.stack([
        np.full(T, 0.28),   # env 0: nominal standing height
        np.full(T, 0.05),   # env 1: fallen (below trunk_height_min=0.18)
    ])  # (2, T)

    up_proj = np.stack([
        np.full(T, 1.0),    # env 0: upright
        np.full(T, 0.1),    # env 1: tipped (below up_proj_min=0.5)
    ])  # (2, T)

    # command: forward at 1 m/s for both
    command = np.zeros((N, T, 3))
    command[:, :, 0] = 1.0

    # lin_vel: env 0 tracks well (small error), env 1 has no forward velocity
    lin_vel = np.zeros((N, T, 3))
    lin_vel[0, :, 0] = 0.95    # env 0: small tracking error
    lin_vel[1, :, 0] = 0.0     # env 1: stationary (large tracking error)

    # ang_vel: both stable (well below ang_vel_max=8.0)
    ang_vel = np.zeros((N, T, 3))

    return {
        "trunk_height": trunk_height,
        "up_proj": up_proj,
        "lin_vel": lin_vel,
        "ang_vel": ang_vel,
        "command": command,
    }


def test_scores_shape():
    traj = _batched_traj()
    scores, sigs, comps = scores_from_trajectories(traj, THRESHOLDS, HORIZON)
    assert scores.shape == (N,), f"Expected shape ({N},), got {scores.shape}"


def test_fallen_scores_higher_than_nominal():
    traj = _batched_traj()
    scores, sigs, comps = scores_from_trajectories(traj, THRESHOLDS, HORIZON)
    assert scores[1] > scores[0], (
        f"Fallen env (score={scores[1]:.4f}) should score higher than "
        f"nominal env (score={scores[0]:.4f})"
    )


def test_signatures_count():
    traj = _batched_traj()
    scores, sigs, comps = scores_from_trajectories(traj, THRESHOLDS, HORIZON)
    assert len(sigs) == N, f"Expected {N} signatures, got {len(sigs)}"


def test_component_signals_count():
    traj = _batched_traj()
    scores, sigs, comps = scores_from_trajectories(traj, THRESHOLDS, HORIZON)
    assert len(comps) == N, f"Expected {N} component dicts, got {len(comps)}"


def test_fallen_env_detected_as_fall():
    traj = _batched_traj()
    scores, sigs, comps = scores_from_trajectories(traj, THRESHOLDS, HORIZON)
    assert comps[1]["fall"] == 1.0, "Env 1 (fallen) should have fall=1.0"
    assert comps[0]["fall"] == 0.0, "Env 0 (nominal) should have fall=0.0"


# ---------------------------------------------------------------------------
# _records: Upgrade #6 batch_key + env_index wiring (pure, no JAX)
# ---------------------------------------------------------------------------

def _space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


def test_records_set_batch_key_and_env_index():
    """_records stores the batch key (as two uint32 ints) on every row and sets
    env_index to the row's index WITHIN the batch (not the global id offset)."""
    space = _space()
    D = len(space.names())
    n = 3
    thetas = np.tile(np.array(space.nominal_vector(), dtype=float), (n, 1))
    scores = np.array([0.1, 0.6, 0.95])
    sigs = [[0.0, 0.0, 0.0] for _ in range(n)]
    comps = [{"fall": 0.0} for _ in range(n)]
    # A legacy PRNGKey is a uint32[2]; _records must accept any array-like of >=2 ints.
    batch_key = np.array([7, 11], dtype=np.uint32)

    recs = _records(space, thetas, scores, sigs, comps, seed=5, tag="random",
                    start_id=10, batch_key=batch_key)

    assert len(recs) == n
    for i, r in enumerate(recs):
        assert r.batch_key == [7, 11]          # stored as plain Python ints
        assert r.env_index == i                # row index within the batch
        assert r.id == f"random-{10 + i}"      # global id still uses start_id
        assert r.seed == 5
        assert len(r.theta_vector) == D


def test_records_env_index_independent_of_start_id():
    """env_index must be the within-batch row index even when start_id > 0, so the
    render key fold_in(batch_key, env_index) reconstructs the eval row exactly."""
    space = _space()
    thetas = np.tile(np.array(space.nominal_vector(), dtype=float), (2, 1))
    recs = _records(space, thetas, np.array([0.5, 0.5]), [[0.0]] * 2, [{"fall": 0.0}] * 2,
                    seed=0, tag="cmaes", start_id=256, batch_key=[1, 2])
    assert [r.env_index for r in recs] == [0, 1]
    assert [r.id for r in recs] == ["cmaes-256", "cmaes-257"]
