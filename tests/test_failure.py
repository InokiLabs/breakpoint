"""Unit tests for breakpoint_mjx.failure — pure numpy, no JAX/GPU required."""
import numpy as np
import pytest

from breakpoint_mjx.failure import (
    component_signals,
    failure_score,
    rollout_signature,
    behavior_descriptor,
    FAILURE_MODES,
    SIGNATURE_LEN,
)

THRESHOLDS = dict(
    trunk_height_min=0.18,
    up_proj_min=0.5,
    tracking_err_max=0.5,
    ang_vel_max=8.0,
)


def _traj(T=500, height=0.28, up=1.0, track_err=0.05, ang=0.0):
    """Build a synthetic trajectory dict with the 5 required keys.

    command is [1, 0, 0] (forward at 1 m/s).
    lin_vel is set so the L2 error in the xy plane equals `track_err`:
      lin_vel[:, 0] = 1 - track_err  (under-tracks by track_err in x)
    ang_vel is filled so ||ang_vel|| == `ang` (all in z component).
    """
    command = np.zeros((T, 3)); command[:, 0] = 1.0
    lin_vel = np.zeros((T, 3)); lin_vel[:, 0] = 1.0 - track_err
    ang_vel = np.zeros((T, 3)); ang_vel[:, 2] = ang
    trunk_height = np.full(T, height, dtype=float)
    up_proj = np.full(T, up, dtype=float)
    return dict(
        trunk_height=trunk_height,
        up_proj=up_proj,
        lin_vel=lin_vel,
        ang_vel=ang_vel,
        command=command,
    )


# ---------------------------------------------------------------------------
# Nominal trajectory
# ---------------------------------------------------------------------------

def test_nominal_no_fall():
    traj = _traj(height=0.28, up=1.0, track_err=0.05, ang=0.0)
    sig = component_signals(traj, THRESHOLDS)
    assert sig["fall"] == 0.0
    assert sig["tracking_error"] < THRESHOLDS["tracking_err_max"]
    score = failure_score(sig, horizon=500)
    assert score < 0.5, f"nominal score too high: {score}"


# ---------------------------------------------------------------------------
# Fall by height
# ---------------------------------------------------------------------------

def test_fallen_by_height():
    traj = _traj(height=0.05, up=1.0)
    sig = component_signals(traj, THRESHOLDS)
    assert sig["fall"] == 1.0
    assert sig["time_to_failure"] == 0.0
    score = failure_score(sig, horizon=500)
    assert score > 0.5, f"fallen-by-height score too low: {score}"


# ---------------------------------------------------------------------------
# Fall by tilt (up_proj below threshold, height ok)
# ---------------------------------------------------------------------------

def test_fallen_by_tilt():
    traj = _traj(height=0.28, up=0.1)
    sig = component_signals(traj, THRESHOLDS)
    assert sig["fall"] == 1.0


# ---------------------------------------------------------------------------
# Tracking loss (command forward, robot stationary)
# ---------------------------------------------------------------------------

def test_tracking_loss():
    traj = _traj(height=0.28, up=1.0, track_err=1.0, ang=0.5)
    sig = component_signals(traj, THRESHOLDS)
    assert sig["fall"] == 0.0
    assert sig["tracking_error"] > THRESHOLDS["tracking_err_max"]
    score = failure_score(sig, horizon=500)
    assert score > 0.0


# ---------------------------------------------------------------------------
# NaN trajectory (sim blow-up)
# ---------------------------------------------------------------------------

def test_nan_trajectory_is_failure():
    traj = _traj()
    traj["trunk_height"] = np.full(500, np.nan)
    sig = component_signals(traj, THRESHOLDS)
    assert sig["fall"] == 1.0
    score = failure_score(sig, horizon=500)
    assert np.isfinite(score), f"failure_score is not finite: {score}"


# ---------------------------------------------------------------------------
# Signature shape and distinctness
# ---------------------------------------------------------------------------

def test_signature_length():
    traj = _traj()
    sig = rollout_signature(traj)
    assert len(sig) == SIGNATURE_LEN


def test_nominal_vs_fallen_signatures_differ():
    nom = rollout_signature(_traj(height=0.28, up=1.0))
    fallen = rollout_signature(_traj(height=0.05, up=0.1))
    assert nom != fallen, "Nominal and fallen signatures are identical — check downsampling"


# ---------------------------------------------------------------------------
# behavior_descriptor (MAP-Elites axes) — pure numpy
# ---------------------------------------------------------------------------

def _mode(name):
    return FAILURE_MODES.index(name)


def test_descriptor_height_loss_immediate():
    """Fall by height from step 0 -> mode height_loss, ttf_norm 0.0."""
    traj = _traj(T=500, height=0.05, up=1.0)
    mode, ttf = behavior_descriptor(traj, THRESHOLDS, horizon=500)
    assert mode == _mode("height_loss"), FAILURE_MODES[mode]
    assert ttf == 0.0


def test_descriptor_orientation():
    """Tilt (up_proj below threshold, height ok) -> mode orientation."""
    traj = _traj(T=500, height=0.28, up=0.1)
    mode, ttf = behavior_descriptor(traj, THRESHOLDS, horizon=500)
    assert mode == _mode("orientation"), FAILURE_MODES[mode]
    assert ttf == 0.0


def test_descriptor_tracking_no_fall():
    """Large tracking error but never falls -> mode tracking_error, ttf_norm 1.0."""
    traj = _traj(T=500, height=0.28, up=1.0, track_err=1.0)
    mode, ttf = behavior_descriptor(traj, THRESHOLDS, horizon=500)
    assert mode == _mode("tracking_error"), FAILURE_MODES[mode]
    assert ttf == 1.0


def test_descriptor_nan_takes_precedence():
    """Any non-finite step -> mode nan, even though tracking-error fill would tie."""
    traj = _traj(T=500)
    traj["trunk_height"] = np.full(500, np.nan)
    mode, ttf = behavior_descriptor(traj, THRESHOLDS, horizon=500)
    assert mode == _mode("nan"), FAILURE_MODES[mode]


def test_descriptor_midrollout_ttf():
    """Nominal for first half, then falls by height -> ttf_norm ~0.5, mode height_loss."""
    T = 500
    height = np.concatenate([np.full(250, 0.28), np.full(250, 0.05)])
    up = np.full(T, 1.0)
    command = np.zeros((T, 3)); command[:, 0] = 1.0
    lin_vel = np.zeros((T, 3)); lin_vel[:, 0] = 0.95
    ang_vel = np.zeros((T, 3))
    traj = dict(trunk_height=height, up_proj=up, lin_vel=lin_vel, ang_vel=ang_vel, command=command)
    mode, ttf = behavior_descriptor(traj, THRESHOLDS, horizon=T)
    assert mode == _mode("height_loss"), FAILURE_MODES[mode]
    assert ttf == pytest.approx(0.5, abs=1e-6)


def test_descriptor_ttf_in_unit_interval_and_mode_in_range():
    """Descriptor always returns a valid (mode_index, ttf_norm in [0,1])."""
    for traj in (_traj(height=0.28, up=1.0), _traj(height=0.05), _traj(up=0.1)):
        mode, ttf = behavior_descriptor(traj, THRESHOLDS, horizon=500)
        assert 0 <= mode < len(FAILURE_MODES)
        assert 0.0 <= ttf <= 1.0


def test_descriptor_precomputed_sig_matches():
    """Passing a precomputed component_signals (the atlas optimization) yields the
    identical descriptor as recomputing it internally."""
    for traj in (_traj(height=0.05), _traj(up=0.1), _traj(track_err=1.0)):
        sig = component_signals(traj, THRESHOLDS)
        assert behavior_descriptor(traj, THRESHOLDS, 500, sig=sig) == \
               behavior_descriptor(traj, THRESHOLDS, 500)
