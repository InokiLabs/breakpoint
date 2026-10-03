"""Pure-numpy failure detection over a single env's trajectory dict.
Keys: trunk_height (T,), up_proj (T,), lin_vel (T,3), ang_vel (T,3), command (T,3),
plus the optional physical-validity keys floor_height (T,), penetration (T,),
qacc_max (T,) (see the validity gate below; absent -> historical behaviour).
Non-finite values (sim blow-ups) are treated as failures, never NaN-poison the score."""
from __future__ import annotations
import numpy as np

SIGNATURE_POINTS = 16
SIGNATURE_LEN = SIGNATURE_POINTS * 3  # height, up_proj, tracking-error channels


def _downsample(x, n):
    x = np.asarray(x)
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx]


def _tracking_err_series(traj):
    lin = np.asarray(traj["lin_vel"]); ang = np.asarray(traj["ang_vel"]); cmd = np.asarray(traj["command"])
    lin_err = np.linalg.norm(cmd[:, :2] - lin[:, :2], axis=1)
    yaw_err = np.abs(cmd[:, 2] - ang[:, 2])
    return np.nan_to_num(lin_err + yaw_err, nan=5.0, posinf=5.0, neginf=5.0)


# --------------------------------------------------------------------------- #
# Physical-validity gate (default ON whenever the trajectory carries the validity keys
# that objectives.legged.legged_summarize emits: floor_height / penetration / qacc_max).
#
# A rollout in which the SIMULATOR misbehaves before the policy has failed (a NaN state,
# the torso passing below the floor plane while the robot is still upright, contacts
# interpenetrating far deeper than nominal walking ever produces, or an acceleration
# blow-up) is a simulator artifact, not a policy failure. It is flagged ``sim_invalid``
# and scores 0, so the search never optimizes toward it and families / stats / riskcert
# never count it as a failure. Trajectories WITHOUT the validity keys (older runs, the
# Genesis duck backend, synthetic tests) get ``sim_invalid = 0`` and score
# byte-identically to before.
#
# CAUSAL WINDOW (measured, docs/validity.md): the Go1 Playground model gives ONLY the four
# foot spheres floor collision, so after a genuine tip-over the collision-less trunk sinks
# through the floor (CPU, horizon 150, obs_dropout_prob=0.2: 18/32 rollouts end below the
# floor and in all 18 the tip-over, up_proj < 0.5, came 6 to 12 steps BEFORE the sink).
# Those are real falls with a post-failure rendering artifact, so by default the gate only
# inspects steps up to and including the first fall step, and "below floor" only counts
# while the robot has not yet tipped over. Set thresholds["validity_whole_rollout"] = 1 to
# inspect every step instead (this also discards genuine falls that end in a sink).
#
# SETTLE WINDOW: the env's reset pose is computed for a flat floor, so on a tilted floor
# (floor_slope_rad = +/-0.1) the feet spawn up to ~6 cm inside the plane for the first
# few steps (measured: every step with penetration < -3 cm on the slope sweep fell in
# steps 0-9, runs of <= 2 steps). Penetration is therefore not checked during the first
# ``validity_settle_steps`` (default 10 = 0.2 s at 50 Hz).
#
# CALIBRATION (CPU, demo checkpoint media/dl/frozen_go1_policy.pkl, horizon 150,
# 32 nominal-theta rollouts, scripts/validity_probe.py): nominal worst foot penetration
# -0.0266 m (the MJX soft contact lets a 0.023 m foot sphere sink ~2 cm while walking),
# nominal max |qacc| 2466. Defaults sit at ~1.5x and ~4x those: 0 of 32 nominal flagged.
# --------------------------------------------------------------------------- #
DEFAULT_PENETRATION_MAX = 0.04    # metres of interpenetration tolerated (nominal: 0.0266)
DEFAULT_QACC_MAX = 1.0e4          # max |qacc| tolerated (nominal: 2466)
DEFAULT_SETTLE_STEPS = 10         # spawn transient skipped for the penetration check

# Reason bits for ``sim_invalid_reason`` (a float bitmask, so every signal stays a float:
# search._records stores component_signals as {k: float(v)}).
SIM_INVALID_REASONS = (
    (1, "nonfinite"),        # NaN / inf in the state (solver blow-up)
    (2, "below_floor"),      # torso below the floor plane while still upright
    (4, "penetration"),      # contact interpenetration deeper than penetration_max
    (8, "blowup"),           # max |qacc| above qacc_max
)
VALIDITY_KEYS = ("floor_height", "penetration", "qacc_max")


def _height_series(traj):
    """The fall-height channel: ``floor_height`` (along the floor normal) when present,
    else the historical world-frame ``trunk_height``."""
    return traj["floor_height"] if "floor_height" in traj else traj["trunk_height"]


def has_validity_keys(traj) -> bool:
    """True iff the trajectory carries the physical-validity keys (gate active)."""
    return all(k in traj for k in VALIDITY_KEYS)


def _validity(traj, thresholds, nonfinite, fallen, up):
    """(sim_invalid: float, reason_bits: float) for one trajectory. Pure numpy.

    ``fallen`` is the per-step fall mask ``component_signals`` computed and ``up`` the
    NaN-cleaned up_proj; together they define the causal window (see the block above)."""
    if not has_validity_keys(traj):
        return 0.0, 0.0
    fh = np.asarray(traj["floor_height"], dtype=float)
    pen = np.asarray(traj["penetration"], dtype=float)
    qa = np.asarray(traj["qacc_max"], dtype=float)
    T = len(fh)
    whole = bool(float(thresholds.get("validity_whole_rollout", 0.0)))
    settle = int(thresholds.get("validity_settle_steps", DEFAULT_SETTLE_STEPS))
    pen_max = float(thresholds.get("penetration_max", DEFAULT_PENETRATION_MAX))
    qacc_lim = float(thresholds.get("qacc_max", DEFAULT_QACC_MAX))
    steps = np.arange(T)
    if whole:
        window = np.ones(T, dtype=bool)
        upright = np.ones(T, dtype=bool)
    else:
        t_fail = int(np.argmax(fallen)) if bool(np.any(fallen)) else T - 1
        window = steps <= t_fail
        tipped = up < float(thresholds["up_proj_min"])
        t_tip = int(np.argmax(tipped)) if bool(tipped.any()) else T
        upright = steps < t_tip
    bad = ~np.isfinite(fh) | ~np.isfinite(pen) | ~np.isfinite(qa) | np.asarray(nonfinite)
    bits = 0
    if bool((bad & window).any()):
        bits |= 1
    if bool(((np.where(np.isfinite(fh), fh, np.inf) < 0.0) & upright).any()):
        bits |= 2
    pen_win = window & (steps >= settle)
    if bool(((np.where(np.isfinite(pen), pen, np.inf) < -pen_max) & pen_win).any()):
        bits |= 4
    if bool(((np.where(np.isfinite(qa), qa, -np.inf) > qacc_lim) & window).any()):
        bits |= 8
    return float(bits != 0), float(bits)


def is_sim_invalid(sig) -> bool:
    """True iff a ``component_signals`` dict was flagged as a simulator artifact."""
    return bool(sig.get("sim_invalid", 0.0))


def sim_invalid_reasons(sig) -> list:
    """Human reason names for a flagged signals dict (empty when valid)."""
    bits = int(sig.get("sim_invalid_reason", 0.0) or 0)
    return [name for bit, name in SIM_INVALID_REASONS if bits & bit]


def component_signals(traj, thresholds):
    raw_up = np.asarray(traj["up_proj"])
    # Fall height: floor-relative when the trajectory carries it (a tilted floor no longer
    # reads as a fall while the robot walks downhill), else the historical world-frame
    # trunk_height (backward compatible for older trajectories / backends).
    raw_h = np.asarray(_height_series(traj))
    h = np.nan_to_num(raw_h, nan=-1.0, posinf=-1.0, neginf=-1.0)
    up = np.nan_to_num(raw_up, nan=-1.0, posinf=-1.0, neginf=-1.0)
    ang = np.nan_to_num(np.asarray(traj["ang_vel"]), nan=0.0, posinf=0.0, neginf=0.0)
    T = len(h)
    nonfinite = ~np.isfinite(raw_h) | ~np.isfinite(raw_up)
    fallen = (h < thresholds["trunk_height_min"]) | (up < thresholds["up_proj_min"]) | nonfinite
    fall = float(fallen.any())
    ttf = float(np.argmax(fallen)) if fall else float(T)
    tracking_error = float(_tracking_err_series(traj).mean())
    ang_norm = np.linalg.norm(ang, axis=1)
    instability = float((ang_norm > thresholds["ang_vel_max"]).mean())
    sim_invalid, reason = _validity(traj, thresholds, nonfinite, fallen, up)
    return {
        "fall": fall, "time_to_failure": ttf, "tracking_error": tracking_error,
        "instability": instability, "min_height": float(h.min()),
        "sim_invalid": sim_invalid, "sim_invalid_reason": reason,
    }


def failure_score(sig, horizon=500):
    """Higher = more broken. Fall dominates; tracking + instability add; earlier failure scores higher.

    A rollout flagged ``sim_invalid`` (a simulator artifact, see the validity gate above)
    scores 0.0: it is excluded from failures, so families and riskcert never count it."""
    if sig.get("sim_invalid", 0.0):
        return 0.0
    score = 0.7 * sig["fall"]
    score += 0.3 * min(sig["tracking_error"] / 2.0, 1.0)
    score += 0.2 * sig["instability"]
    if sig["fall"]:
        score += 0.2 * (1.0 - sig["time_to_failure"] / horizon)
    return float(score)


# Ordered, fixed set of failure modes for the MAP-Elites atlas behavior axis 1.
# Index into this tuple is stable/interpretable; do not reorder without re-running.
FAILURE_MODES = ("height_loss", "orientation", "tracking_error", "nan")


def behavior_descriptor(one_traj, thresholds, horizon, sig=None):
    """Pure-numpy 2D behavior descriptor for ONE env's rollout (the MAP-Elites axes).

    Returns ``(mode_index: int, ttf_norm: float)``:

    * **Axis 1 -- dominant failure mode.** Index into ``FAILURE_MODES``. Precedence:
      a sim blow-up is categorically the most severe and most distinct mode, so if ANY
      step is non-finite the mode is ``"nan"`` outright. Otherwise it is the argmax over
      three per-mode severities, derived from the SAME ``thresholds`` ``component_signals``
      uses (no new thresholds re-derived) on the all-finite trajectory:
        - ``height_loss``    = fraction of steps with ``trunk_height < trunk_height_min``
        - ``orientation``    = fraction of steps with ``up_proj      < up_proj_min``
        - ``tracking_error`` = mean tracking error / ``tracking_err_max``, clipped to 1
      Ties break by ``FAILURE_MODES`` order (``height_loss`` first, so a clean fall where
      both height and orientation cross reads as ``height_loss``). A non-failing rollout
      has near-zero severities; its low fitness keeps it out of the elite set in practice.
    * **Axis 2 -- time-to-failure.** First failing step / ``horizon`` using the SAME
      fall semantics as ``component_signals['time_to_failure']`` (trunk_height/up_proj
      crossing or non-finite), in ``[0, 1]``; ``1.0`` if it never fails.

    ``sig`` is an optional precomputed ``component_signals(one_traj, thresholds)`` dict.
    When supplied (e.g. by the atlas, which already computed it for the failure score),
    the internal ``component_signals`` call is skipped -- identical output, one fewer pass.
    """
    if sig is None:
        sig = component_signals(one_traj, thresholds)
    raw_h = np.asarray(_height_series(one_traj)); raw_up = np.asarray(one_traj["up_proj"])
    nonfinite = ~np.isfinite(raw_h) | ~np.isfinite(raw_up)

    if nonfinite.any():
        mode_index = FAILURE_MODES.index("nan")
    else:
        # All steps are finite in this branch, so the raw arrays need no NaN handling.
        height_sev = float((raw_h < thresholds["trunk_height_min"]).mean())
        orient_sev = float((raw_up < thresholds["up_proj_min"]).mean())
        track_sev = float(min(sig["tracking_error"] / max(thresholds["tracking_err_max"], 1e-9), 1.0))
        # Order matches FAILURE_MODES; nan severity is 0 here (the any-nonfinite case
        # is handled above). np.argmax breaks ties toward the lower (earlier) index.
        severities = np.array([height_sev, orient_sev, track_sev, 0.0])
        mode_index = int(np.argmax(severities))

    if sig["fall"]:
        ttf_norm = float(min(sig["time_to_failure"] / max(float(horizon), 1.0), 1.0))
    else:
        ttf_norm = 1.0
    return mode_index, ttf_norm


def rollout_signature(traj):
    h = _downsample(np.nan_to_num(np.asarray(_height_series(traj)), nan=-1.0, posinf=-1.0, neginf=-1.0), SIGNATURE_POINTS)
    up = _downsample(np.nan_to_num(np.asarray(traj["up_proj"]), nan=-1.0, posinf=-1.0, neginf=-1.0), SIGNATURE_POINTS)
    terr = _downsample(_tracking_err_series(traj), SIGNATURE_POINTS)
    return np.concatenate([h, up, terr]).astype(float).tolist()
