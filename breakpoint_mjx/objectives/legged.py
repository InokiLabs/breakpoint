"""The LEGGED (stay-upright) failure objective, packaged as an ``Objective``.

This is a WRAPPER, not a rewrite: the callables it exposes ARE the proven
``breakpoint_mjx.failure`` functions (``component_signals`` / ``failure_score`` /
``behavior_descriptor`` / ``rollout_signature`` / ``FAILURE_MODES``), and
``legged_summarize`` is the former inline ``rollout.rollout_one`` summary dict, moved
here VERBATIM (same ops, same key order). So go1/barkour/g1 -- which route through this
objective (it is the ``Target.objective`` default) -- produce byte-identical scores,
signals, descriptors, and summaries to the pre-T7 engine.

``legged_summarize`` reads the floating-base pose/velocity positionally
(``qpos[2]`` / ``qvel[:3]`` / ``qvel[3:6]``) and the torso orientation
(``xmat[torso_body_id]``), exactly as before; ``consts`` is the legged
``(floor_geom_id, torso_body_id)`` tuple. It uses only array operators / indexing (no
jax import), so this module imports on a CPU-only box.
"""
from __future__ import annotations

import numpy as np

from breakpoint_mjx import failure
from breakpoint_mjx.objectives import Objective


# Fill value for ``penetration`` when the backend exposes no contact arrays (or zero
# contacts): large and positive, i.e. "nothing is interpenetrating". A constant, so it is
# jit-safe and never trips the penetration gate.
NO_CONTACT_DIST = 1.0e3


def _contact_dist(data):
    """The per-contact signed distance array (negative = penetration), or None.

    MJX >= 3.3 keeps contacts on the implementation-private ``data._impl.contact``;
    older MJX (and C-MuJoCo-shaped fakes) expose ``data.contact``. The lookup is
    structural (static Python on attributes, never on traced values), so it is
    jit-safe. Returns None when no contact array is reachable.
    """
    # Prefer _impl (reading ``data.contact`` on MJX >= 3.3 raises a DeprecationWarning).
    impl = getattr(data, "_impl", None)
    con = getattr(impl, "contact", None) if impl is not None else None
    if con is None:
        con = getattr(data, "contact", None)
    dist = getattr(con, "dist", None) if con is not None else None
    return dist


def legged_summarize(state, consts):
    """Per-step legged summary: the former inline ``rollout_one`` body, verbatim, plus
    three physical-validity keys.

    ``consts`` == ``(floor_geom_id, torso_body_id)`` (the legged convention). The first
    five keys are unchanged (same ops, same order): ``trunk_height`` is still the
    world-frame ``qpos[2]`` that autolytic / crossval read.

    Physical-validity keys (added only when the backend's data carries the fields, so a
    duck-typed state with just ``xmat/qpos/qvel`` still returns the original 5 keys):

    * ``floor_height`` -- torso position above the floor PLANE, measured along the
      floor's own normal: ``dot(xpos[torso] - geom_xpos[floor], geom_xmat[floor][:, 2])``.
      On a tilted floor (``floor_slope_rad``) world z drifts as the robot walks up or
      down the slope; this does not. Negative = the torso is below the floor.
    * ``penetration``  -- ``min(contact.dist)`` this step (negative = interpenetration
      depth, metres); ``NO_CONTACT_DIST`` when there are no contact arrays / contacts.
    * ``qacc_max``     -- ``max |qacc|`` this step, a solver blow-up detector.

    Array operators / ``.min()`` / ``.max()`` only (no jax import), jit-safe.
    """
    floor_geom_id, torso_body_id = consts
    data = state.data
    R = data.xmat[torso_body_id].reshape(3, 3)
    out = {
        "trunk_height": data.qpos[2],
        "up_proj": R[2, 2],
        "lin_vel": R.T @ data.qvel[:3],
        "ang_vel": data.qvel[3:6],
        "command": state.info["command"],
    }
    geom_xpos = getattr(data, "geom_xpos", None)
    geom_xmat = getattr(data, "geom_xmat", None)
    xpos = getattr(data, "xpos", None)
    qacc = getattr(data, "qacc", None)
    if geom_xpos is None or geom_xmat is None or xpos is None or qacc is None:
        return out
    normal = geom_xmat[floor_geom_id].reshape(3, 3)[:, 2]
    out["floor_height"] = ((xpos[torso_body_id] - geom_xpos[floor_geom_id]) * normal).sum()
    dist = _contact_dist(data)
    if dist is None or dist.shape[0] == 0:          # static shape check, jit-safe
        # ``qacc[0] * 0 + C`` keeps the dtype/array type of the backend (jnp or numpy).
        out["penetration"] = qacc[0] * 0.0 + NO_CONTACT_DIST
    else:
        out["penetration"] = dist.min()
    out["qacc_max"] = abs(qacc).max()
    return out


def legged_summarize_in_frame(R_heading):
    """``legged_summarize`` with velocities expressed in a fixed frame mounted on the torso.

    ``R_heading`` (3x3, constant) is that frame's orientation relative to the torso body
    frame -- the robot's IMU / heading site, the frame its joystick command is defined in.
    ``legged_summarize`` reads velocities in the torso BODY frame, which is right only when
    the two coincide (true for every Playground robot except Barkour, whose ``imu_frame``
    site is yawed 180 deg: its body x axis points backwards, so forward walking read as
    reverse and tracking_error was ~1.5 on policies that track to ~0.14).
    ``(R.T @ v) @ R_heading`` is ``R_heading.T @ R.T @ v``; the traced array stays the
    left operand, so this is jit-safe with a numpy ``R_heading``.
    """
    Rh = np.asarray(R_heading, dtype=np.float32)

    def summarize(state, consts):
        out = legged_summarize(state, consts)
        out["lin_vel"] = out["lin_vel"] @ Rh
        out["ang_vel"] = out["ang_vel"] @ Rh
        return out

    return summarize


# The legged trajectory/summary contract, for docs + a light test check. The first five
# keys are the historical stay-upright contract; the last three are the physical-validity
# keys (floor-relative height, worst contact penetration, max |qacc|) that
# ``failure.component_signals`` uses to exclude simulator artifacts. Backends whose data
# lacks those fields (e.g. the Genesis duck state) emit only the first five, and
# ``failure`` then scores byte-identically to before.
LEGGED_CONTRACT_KEYS = ("trunk_height", "up_proj", "lin_vel", "ang_vel", "command",
                        "floor_height", "penetration", "qacc_max")
LEGGED_CORE_KEYS = LEGGED_CONTRACT_KEYS[:5]


def legged_nominal_performance(signals, thresholds):
    """Nominal-config performance aggregate for the hardening Pareto (HIGHER better).

    This is the historical nominal-reward math, moved here VERBATIM so
    routing through the objective hook is byte-identical for legged targets:
    ``tracking_reward = mean(1 - clip(tracking_error / tracking_err_max, 0, 1))`` over
    rollouts, ``nominal_reward = tracking_reward * (1 - fall_rate)`` (mean-then-multiply,
    NOT mean of per-rollout products -- preserved exactly).

    Edge decisions (edge-case-enumeration): empty ``signals`` -> all-zero dict (HANDLE;
    a 0-env eval is degenerate but must not NaN the Pareto); non-finite tracking errors
    are treated as worst-case ``tmax`` (HANDLE; a blown-up nominal rollout must not
    poison the mean with NaN -- finite inputs are byte-identical to the historical
    math). Signals from a different objective are OUT-OF-CONTRACT (see Objective docs).
    """
    if not signals:
        return {"nominal_reward": 0.0, "tracking_reward": 0.0, "fall_rate": 0.0,
                "mean_tracking_error": 0.0}
    tmax = float(thresholds.get("tracking_err_max", 0.5))
    terrs = np.asarray([s["tracking_error"] for s in signals], dtype=float)
    terrs = np.nan_to_num(terrs, nan=tmax, posinf=tmax, neginf=tmax)
    falls = np.asarray([s["fall"] for s in signals], dtype=float)
    tracking_reward = float(np.mean(1.0 - np.clip(terrs / max(tmax, 1e-9), 0.0, 1.0)))
    fall_rate = float(np.mean(falls))
    return {
        "nominal_reward": float(tracking_reward * (1.0 - fall_rate)),
        "tracking_reward": tracking_reward,
        "fall_rate": fall_rate,
        "mean_tracking_error": float(terrs.mean()),
    }

# The single legged objective instance. Its hooks ARE the existing failure.py functions
# (identity references), so routing through it is byte-identical to calling them directly.
LEGGED_OBJECTIVE = Objective(
    name="legged",
    summarize=legged_summarize,
    component_signals=failure.component_signals,
    failure_score=failure.failure_score,
    behavior_descriptor=failure.behavior_descriptor,
    failure_modes=failure.FAILURE_MODES,
    rollout_signature=failure.rollout_signature,
    contract_keys=LEGGED_CONTRACT_KEYS,
    nominal_performance=legged_nominal_performance,
)
