"""The MANIPULATION (fixed-base robot-arm) failure objective, as an ``Objective``.

The legged objective asks "did the robot stay upright / track its command?". A
manipulation policy has a completely different failure contract: the FAILURE is the
TASK not being achieved -- the object never reaches its target -- plus the arm-specific
catastrophes a locomotion objective has no notion of: the grasped object being DROPPED,
an arm JOINT driven into its hard LIMIT, or a sim blow-up. This module implements that
contract over a MuJoCo Playground manipulation env (Franka ``PandaPickCube``; see
``breakpoint_mjx.targets.panda`` for the confirmed env facts + sources).

Per-step SUMMARY (``manipulation_summarize``, run under jax.vmap in ``rollout_one``):
  * ``obj_to_target`` -- distance from the manipulated object to its target position.
  * ``ee_to_obj``     -- distance from the end-effector (gripper site) to the object.
  * ``obj_height``    -- object height (z); an informative summary + clustering-signature
                         channel (drop is DETECTED from ``ee_to_obj``, not this).
  * ``joint_margin``  -- min normalized distance of the arm joints to their nearest
                         limit (1 == mid-range, 0 == at a limit), for joint-limit
                         violation detection.
  * ``reached``       -- the env's ``info["reached_box"]`` success/reward proxy.

FAILURE (``component_signals`` / ``failure_score``): the task is NOT achieved (the
object never enters the ``success_dist`` ball of the target), the object is DROPPED
after being grasped (``ee_to_obj`` grows past ``drop_dist`` once grasped), a joint hits
its limit (``joint_margin < joint_margin_min``), or the rollout goes non-finite. Each
catastrophe alone is strong enough to flag a run as a failure.

Thresholds are read from the merged ``cfg.thresholds`` (the panda target supplies the
manipulation keys via ``Target.thresholds``; every key is read with a ``.get`` default
so the objective is robust to a missing key and never depends on the legged keys).

Import-safe: only numpy/stdlib at top. ``manipulation_summarize`` imports ``jax.numpy``
lazily (it is only ever called under a jax trace on Colab); the pure signal/score/
descriptor/signature functions below are exercised on CPU with synthetic trajectories.
"""
from __future__ import annotations

import numpy as np

from breakpoint_mjx.objectives import Objective

# Ordered, fixed set of manipulation failure modes for the MAP-Elites atlas behavior
# axis 1. Index into this tuple is stable/interpretable; do not reorder without
# re-running the atlas. A non-finite blow-up folds into "dropped_object" (a catastrophic
# loss of control of the object), documented in ``manipulation_behavior_descriptor``.
FAILURE_MODES = ("missed_target", "dropped_object", "joint_limit", "timeout")

SIGNATURE_POINTS = 16
SIGNATURE_CHANNELS = 4  # obj_to_target, ee_to_obj, obj_height, joint_margin
SIGNATURE_LEN = SIGNATURE_POINTS * SIGNATURE_CHANNELS

# Single source of truth for the graded-miss normalizer (metres): the object-to-target
# distance at which the "how far did it miss" penalty saturates. Used by BOTH
# failure_score AND behavior_descriptor so they can never disagree. It is a fixed part of
# the objective's scoring definition, NOT a Target.thresholds override -- the
# Objective.failure_score(sig, horizon) contract has no thresholds argument, so a
# per-target override could change the descriptor but not the score. Kept a module
# constant to keep the two consistent.
OBJ_REACH_NORM = 0.30


# Threshold defaults (metres / unitless). The panda target overrides these via
# Target.thresholds; ``.get`` here keeps the objective robust if a key is absent.
_DEFAULTS = {
    "success_dist": 0.05,     # object within this of target => task achieved that step
    "grasp_dist": 0.06,       # gripper within this of object => grasped
    "drop_dist": 0.15,        # once grasped, ee-to-object beyond this => dropped
    "joint_margin_min": 0.02, # normalized joint-to-limit distance below this => violation
}


def _thr(thresholds, key):
    return float(thresholds.get(key, _DEFAULTS[key]))


def manipulation_summarize(state, consts):
    """Per-step manipulation summary (run under jax.vmap in ``rollout_one``).

    ``consts`` is the panda target's resolve_constants() dict (object/gripper/floor ids
    + the arm joint qpos addresses & limits). Returns the 5-key summary the pure
    functions below consume. ``jax.numpy`` is imported here (lazily) so the module stays
    CPU-importable; this runs only under a trace on Colab.
    """
    import jax.numpy as jnp

    d = state.data
    obj_pos = d.xpos[consts["obj_body_id"]]
    gripper_pos = d.site_xpos[consts["gripper_site_id"]]
    target_pos = state.info["target_pos"]

    obj_to_target = jnp.linalg.norm(target_pos - obj_pos)
    ee_to_obj = jnp.linalg.norm(obj_pos - gripper_pos)
    obj_height = obj_pos[2]
    reached = jnp.asarray(state.info["reached_box"], dtype=obj_height.dtype)

    # Joint-limit proximity: normalized distance of each LIMITED arm joint to its nearer
    # limit; take the min across joints so one joint at its stop drives the signal to ~0.
    q = d.qpos[consts["arm_qpos_adr"]]
    lo = jnp.asarray(consts["arm_jnt_lo"])
    hi = jnp.asarray(consts["arm_jnt_hi"])
    span = jnp.maximum(hi - lo, 1e-6)
    margin = jnp.minimum((q - lo) / span, (hi - q) / span)
    joint_margin = jnp.min(margin)

    return {
        "obj_to_target": obj_to_target,
        "ee_to_obj": ee_to_obj,
        "obj_height": obj_height,
        "joint_margin": joint_margin,
        "reached": reached,
    }


def _finite_arrays(traj):
    """Pull the summary arrays as numpy + a per-step non-finite mask (blow-up guard)."""
    o2t = np.asarray(traj["obj_to_target"], dtype=float)
    e2o = np.asarray(traj["ee_to_obj"], dtype=float)
    h = np.asarray(traj["obj_height"], dtype=float)
    jm = np.asarray(traj["joint_margin"], dtype=float)
    nonfinite_step = (
        ~np.isfinite(o2t) | ~np.isfinite(e2o) | ~np.isfinite(h) | ~np.isfinite(jm)
    )
    # A non-finite object-to-target is treated as "very far" so it never reads as success
    # and never becomes the min distance unless the whole rollout blew up.
    o2t = np.nan_to_num(o2t, nan=5.0, posinf=5.0, neginf=5.0)
    e2o = np.nan_to_num(e2o, nan=5.0, posinf=5.0, neginf=5.0)
    jm = np.nan_to_num(jm, nan=-1.0, posinf=-1.0, neginf=-1.0)  # non-finite => "at limit"
    return o2t, e2o, h, jm, nonfinite_step


def component_signals(traj, thresholds):
    """Pure-numpy manipulation failure signals for ONE rollout (summary arrays (T,))."""
    o2t, e2o, h, jm, nonfinite_step = _finite_arrays(traj)
    T = len(o2t)
    success_dist = _thr(thresholds, "success_dist")
    grasp_dist = _thr(thresholds, "grasp_dist")
    drop_dist = _thr(thresholds, "drop_dist")
    joint_margin_min = _thr(thresholds, "joint_margin_min")

    reached = np.nan_to_num(np.asarray(traj.get("reached", np.zeros(T)), dtype=float))

    # Task success: the object entered the target ball at some (finite) step.
    success = float((o2t < success_dist).any())
    min_obj_to_target = float(o2t.min())
    final_obj_to_target = float(o2t[-1])

    # Grasp / drop, from the PHYSICAL end-effector-to-object distance only (the env's
    # ``reached`` info flag is kept as a separate reward proxy, not conflated with the
    # geometric grasp -- otherwise a run flagged "reached" while the gripper is still far
    # would read as an instant false drop): grasped once ee-to-object dips below grasp_dist;
    # dropped if, once grasped, ee-to-object later exceeds drop_dist.
    grasp_step = e2o < grasp_dist
    grasped_by = np.maximum.accumulate(grasp_step.astype(int)).astype(bool)
    drop_event = grasped_by & (e2o > drop_dist)
    grasped = float(grasp_step.any())
    dropped = float(drop_event.any())

    # Joint-limit violation: any arm joint within joint_margin_min of a limit.
    joint_viol = jm < joint_margin_min
    joint_limit_hit = float(joint_viol.any())
    joint_limit_frac = float(joint_viol.mean())
    min_joint_margin = float(jm.min())

    # Timed catastrophic event (for the time-to-failure axis + the earliness bonus): a
    # joint-limit violation, a drop, or a non-finite step. A pure "missed target" has no
    # timed event -- it is only known at the horizon.
    event = joint_viol | drop_event | nonfinite_step
    has_event = bool(event.any())
    ttf = float(np.argmax(event)) if has_event else float(T)

    task_failure = 1.0 - success
    nonfinite = float(nonfinite_step.any())
    failure = float((task_failure > 0.0) or dropped or joint_limit_hit or nonfinite)
    # The env's reached-box success/reward proxy, as a fraction of steps (reporting only;
    # does not gate the failure logic, which uses the geometric signals above).
    reached_frac = float((reached > 0.5).mean())

    return {
        "failure": failure,
        "success": success,
        "task_failure": task_failure,
        "reached_frac": reached_frac,
        "dropped": dropped,
        "grasped": grasped,
        "joint_limit_hit": joint_limit_hit,
        "joint_limit_frac": joint_limit_frac,
        "min_joint_margin": min_joint_margin,
        "min_obj_to_target": min_obj_to_target,
        "final_obj_to_target": final_obj_to_target,
        "nonfinite": nonfinite,
        "time_to_failure": ttf,
        "has_event": float(has_event),
    }


def failure_score(sig, horizon=150):
    """Higher = more broken. Missing the target dominates; each arm catastrophe (drop /
    joint-limit) alone is strong enough to flag a run; an earlier timed catastrophe scores
    higher; a non-finite blow-up is categorically maximal. Clipped to [0, 1]. Uses the
    module-level OBJ_REACH_NORM (same normalizer as behavior_descriptor)."""
    score = 0.50 * sig["task_failure"]                                  # missed target
    score += 0.20 * min(sig["min_obj_to_target"] / OBJ_REACH_NORM, 1.0)  # graded miss
    score += 0.50 * sig["dropped"]                                       # dropped object
    score += 0.50 * sig["joint_limit_hit"]                               # joint-limit hit
    if sig["has_event"]:
        score += 0.15 * (1.0 - sig["time_to_failure"] / max(float(horizon), 1.0))
    if sig["nonfinite"]:
        score = max(score, 1.0)
    return float(min(score, 1.0))


def behavior_descriptor(one_traj, thresholds, horizon, sig=None):
    """2D MAP-Elites descriptor: ``(mode_index, ttf_norm)`` over ``FAILURE_MODES``.

    * **Axis 1 -- dominant failure mode.** A non-finite blow-up is folded into
      ``dropped_object`` (catastrophic loss of control). Otherwise the argmax over four
      severities (order == ``FAILURE_MODES``): ``missed_target`` = graded residual
      distance when the task was not achieved; ``dropped_object`` = dropped flag;
      ``joint_limit`` = fraction of steps in violation; ``timeout`` = a low residual for
      "not achieved, nothing catastrophic". Ties break toward the lower index.
    * **Axis 2 -- time-to-failure.** First timed catastrophic step / ``horizon`` (drop /
      joint-limit / non-finite) in ``[0, 1]``; ``1.0`` when there is no timed event
      (success, or a pure end-of-horizon miss), mirroring the legged ttf semantics.

    ``sig`` may be a precomputed ``component_signals`` dict (the atlas optimization).
    """
    if sig is None:
        sig = component_signals(one_traj, thresholds)

    if sig["nonfinite"] > 0.0:
        mode_index = FAILURE_MODES.index("dropped_object")
    else:
        missed_sev = (1.0 - sig["success"]) * min(
            sig["min_obj_to_target"] / max(OBJ_REACH_NORM, 1e-9), 1.0
        )
        dropped_sev = sig["dropped"]
        joint_sev = sig["joint_limit_frac"]
        timeout_sev = (1.0 - sig["success"]) * 0.5
        # Order matches FAILURE_MODES; argmax breaks ties toward the lower (earlier) index.
        severities = np.array([missed_sev, dropped_sev, joint_sev, timeout_sev])
        mode_index = int(np.argmax(severities))

    if sig["has_event"] > 0.0:
        ttf_norm = float(min(sig["time_to_failure"] / max(float(horizon), 1.0), 1.0))
    else:
        ttf_norm = 1.0
    return mode_index, ttf_norm


def _downsample(x, n):
    x = np.asarray(x)
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx]


def rollout_signature(traj):
    """Fixed-length clustering signature: downsampled object-to-target / ee-to-object /
    object-height / joint-margin channels (non-finite scrubbed)."""
    o2t, e2o, h, jm, _ = _finite_arrays(traj)
    parts = [
        _downsample(o2t, SIGNATURE_POINTS),
        _downsample(e2o, SIGNATURE_POINTS),
        _downsample(h, SIGNATURE_POINTS),
        _downsample(jm, SIGNATURE_POINTS),
    ]
    return np.concatenate(parts).astype(float).tolist()


def nominal_performance(signals, thresholds):
    """Nominal-config performance aggregate for the hardening Pareto (HIGHER better).

    Per rollout: half credit for outright task success, half graded by how close the
    object got to the target (``1 - min_obj_to_target / OBJ_REACH_NORM``), then a
    catastrophe (drop / joint-limit / non-finite) zeroes THAT rollout -- a run that
    smashed the arm into its stops does not earn partial credit. Mean over rollouts,
    in [0, 1]. Same OBJ_REACH_NORM as failure_score/behavior_descriptor so the
    performance and failure axes can never disagree about "close".

    Edge decisions: empty ``signals`` -> all-zero dict (HANDLE, no NaN); component
    signals already scrub non-finite values (``_finite_arrays``), so every field read
    here is finite by construction. Signals from another objective: OUT-OF-CONTRACT.
    """
    if not signals:
        return {"nominal_reward": 0.0, "success_rate": 0.0, "drop_rate": 0.0,
                "joint_limit_rate": 0.0, "mean_min_obj_to_target": 0.0}
    succ = np.asarray([s["success"] for s in signals], dtype=float)
    resid = np.asarray(
        [min(s["min_obj_to_target"] / max(OBJ_REACH_NORM, 1e-9), 1.0) for s in signals],
        dtype=float,
    )
    dropped = np.asarray([s["dropped"] for s in signals], dtype=float)
    joint = np.asarray([s["joint_limit_hit"] for s in signals], dtype=float)
    nonfinite = np.asarray([s["nonfinite"] for s in signals], dtype=float)
    clean = (1.0 - dropped) * (1.0 - joint) * (1.0 - nonfinite)
    per = (0.5 * succ + 0.5 * (1.0 - resid)) * clean
    return {
        "nominal_reward": float(np.clip(per, 0.0, 1.0).mean()),
        "success_rate": float(succ.mean()),
        "drop_rate": float(dropped.mean()),
        "joint_limit_rate": float(joint.mean()),
        "mean_min_obj_to_target": float(
            np.asarray([s["min_obj_to_target"] for s in signals], dtype=float).mean()
        ),
    }


MANIPULATION_CONTRACT_KEYS = (
    "obj_to_target",
    "ee_to_obj",
    "obj_height",
    "joint_margin",
    "reached",
)

MANIPULATION_OBJECTIVE = Objective(
    name="manipulation",
    summarize=manipulation_summarize,
    component_signals=component_signals,
    failure_score=failure_score,
    behavior_descriptor=behavior_descriptor,
    failure_modes=FAILURE_MODES,
    rollout_signature=rollout_signature,
    contract_keys=MANIPULATION_CONTRACT_KEYS,
    nominal_performance=nominal_performance,
)
