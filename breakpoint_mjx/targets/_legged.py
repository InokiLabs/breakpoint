"""Shared per-row MJX field math for any floating-base LEGGED target.

A MuJoCo Playground locomotion robot of this class -- a quadruped (Go1, Barkour) or
a humanoid (Unitree G1) -- shares one structural invariant: a 6-dof free base
followed by N actuated single-dof joints, affine/position actuators whose
``gainprm[:, 0] = kp``, ``biasprm[:, 1] = -kp``, ``biasprm[:, 2] = -kd``, and a
length-(6 + N) dof vector whose ``[6:]`` slice is exactly the N actuated joints. The
theta -> model-field mapping is therefore identical across these robots regardless of
N; only the ``(floor_geom_id, torso_body_id)`` constants and the env/checkpoint differ.

So ``go1``, ``barkour`` (N=12) and ``g1`` (N=29) all delegate here:
``build_legged_fields`` is the single source of truth for the per-row field math (it
IS the proven Go1 ``_build_one`` body, verbatim -- it already used the ``[6:]`` slice
and per-actuator Kp/Kd, both of which generalize to any N), and ``LEGGED_MODEL_FIELDS``
is the single tuple of batched fields. A new legged robot is "write a Target that
points its build_fields at this helper + a theta_space.yaml," with no field-math
copy-paste.

Import-safe: NO jax import at module top. ``build_legged_fields`` only uses the
``x.at[idx].{set,add,multiply}`` API on whatever array type ``base_model`` carries
(real jax arrays on Colab; the numpy-backed fake in tests/test_targets.py), so this
module imports on a CPU-only box with no JAX/GPU.
"""
from __future__ import annotations

# Single source of truth: the MJX model fields a legged target batches. render
# imports this via target.model_fields so the squeeze can never drift from
# apply_thetas, and every legged target (quadruped or humanoid) shares this tuple.
#
# geom_quat is the newest entry (Tier-3 GEOMETRY axis): the floor-slope dim tilts the
# ground plane by rotating the floor geom's quaternion. It is always batched (a no-op
# when floor_slope_rad is absent / 0), so the tuple length is 8. FOLLOW-UP: full
# heightfield terrain (steps / gaps / soft ground) needs a rough-terrain env with a
# heightfield/hfield asset, not just a tilted flat plane -- documented, out of scope here.
LEGGED_MODEL_FIELDS = (
    "geom_friction",
    "body_mass",
    "body_ipos",
    "actuator_gainprm",
    "actuator_biasprm",
    "dof_frictionloss",
    "dof_armature",
    "geom_quat",
)


def _numeric_namespace():
    """Return the numeric namespace for the floor-slope trig: ``jax.numpy`` on Colab
    (traced scalars under vmap) or ``numpy`` on a CPU-only box (python-float scalars in
    tests). Keeps this module import-safe -- imported lazily only when the slope math
    runs, and degrades to numpy where jax is not installed."""
    try:
        import jax.numpy as jnp
        return jnp
    except ImportError:
        import numpy as np
        return np


def _pitch_quat_about_y(xp, angle):
    """Unit quaternion (w, x, y, z) for a rotation of ``angle`` rad about the y-axis."""
    half = 0.5 * angle
    return xp.array([xp.cos(half), 0.0 * angle, xp.sin(half), 0.0 * angle])


def _quat_mul(xp, q1, q2):
    """Hamilton product of two (w, x, y, z) quaternions -> (w, x, y, z).

    Kept byte-identical to backends.mujoco_c._quat_mul_np (the C-backend numpy mirror),
    so cross-backend reproduction composes the floor tilt the same way (test_crossval).
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return xp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def validate_legged_model(base_model):
    """Fail loudly at setup if base_model is NOT the layout the field math assumes.

    The shared legged field math + the rollout summary depend on one structural
    invariant: **joint 0 is a 6-dof FREE base, followed by N actuated single-dof
    joints.** Two checks enforce it:

    1. ``jnt_type[0] == mjJNT_FREE`` (0). rollout.py reads the base pose/velocity
       positionally -- ``qpos[2]`` (base height), ``qvel[:3]`` (base linvel),
       ``qvel[3:6]`` (base angvel) -- and the field math slices ``[6:]`` as the
       actuated joints, all of which assume joint 0 is the free base. A layout whose
       joint 0 is NOT free (e.g. leading passive 1-dof joints) could still have the
       right dof COUNT yet mis-slice silently, so the count check alone is
       insufficient; this guards it directly.
    2. ``dof_armature`` / ``dof_frictionloss`` length == ``6 + nu`` (the free base's
       6 dofs + one dof per actuator), so ``[6:]`` is exactly the ``nu`` actuated
       joints. Holds for any N -- Go1/Barkour (6 + 12 = 18) and the G1 humanoid
       (6 + 29 = 35) all pass.

    Call this ONCE at setup (a target's resolve_constants), not per row.

    Raises
    ------
    ValueError : if ``jnt_type[0]`` is not a free joint (naming the actual type), or
                 if ``dof_armature`` / ``dof_frictionloss`` length != ``6 + nu``
                 (naming expected vs actual).

    Also prints a WARNING (does NOT raise) if the affine-actuator relation
    ``biasprm[:, 1] ~= -gainprm[:, 0]`` does not hold -- the Kp/Kd math assumes
    position/affine actuators, but column conventions vary by vendor, so this is a
    heads-up rather than a false failure on an otherwise-valid robot.
    """
    import numpy as np

    # (1) Joint 0 must be the 6-dof FREE base. MuJoCo mjJNT_FREE == 0.
    jnt0 = int(base_model.jnt_type[0])
    if jnt0 != 0:
        raise ValueError(
            f"legged field math assumes joint 0 is a 6-dof FREE base (MuJoCo "
            f"mjJNT_FREE == 0), got jnt_type[0]={jnt0}. rollout.py reads the base "
            f"pose/velocity from qpos[2] / qvel[:3] / qvel[3:6] and the field math "
            f"slices [6:] as the actuated joints, all positional on a free-base-first "
            f"layout; a non-free-base joint 0 would mis-slice silently, so it is "
            f"rejected here."
        )

    # (2) dof vectors must be length (6 free-base + nu actuated) so [6:] = the joints.
    n_act = base_model.actuator_gainprm.shape[0]
    expected_dof = 6 + n_act

    n_arm = base_model.dof_armature.shape[0]
    if n_arm != expected_dof:
        raise ValueError(
            f"legged field math expects a length-(6 + nu) dof vector for a 6-dof free "
            f"base + {n_act} actuated joints, i.e. dof_armature.shape[0]=={expected_dof}, "
            f"got {n_arm}. The shared legged target assumes a 6-dof free base + N "
            f"actuated single-dof joints (so [6:] is exactly the actuated joints); a "
            f"mismatched layout is rejected here."
        )
    n_fl = base_model.dof_frictionloss.shape[0]
    if n_fl != expected_dof:
        raise ValueError(
            f"legged field math expects a length-(6 + nu) dof vector for a 6-dof free "
            f"base + {n_act} actuated joints, i.e. dof_frictionloss.shape[0]=="
            f"{expected_dof}, got {n_fl}."
        )

    # Soft affine-actuator invariant: biasprm[:,1] ~= -gainprm[:,0]. Warn, never fail.
    kp = np.asarray(base_model.actuator_gainprm[:, 0])
    neg_kp = np.asarray(base_model.actuator_biasprm[:, 1])
    if not np.allclose(neg_kp, -kp, rtol=1e-3, atol=1e-6):
        print(
            "WARNING [legged target]: affine-actuator relation biasprm[:,1] ~= "
            "-gainprm[:,0] does not hold; the Kp/Kd field math assumes position/affine "
            "actuators. Verify this robot's actuator column convention.",
            flush=True,
        )


def build_legged_fields(scalars, base_model, consts):
    """Per-row legged model-field math (the proven Go1 ``_build_one`` body, plus the
    Tier-3 floor-slope ``geom_quat`` tilt appended at the end).

    Parameters
    ----------
    scalars    : dict {model-dim name: scalar theta value} for one row (vmapped).
    base_model : the captured original MJX model (broadcast, not vmapped).
    consts     : (floor_geom_id, torso_body_id) from the target's resolve_constants.

    Returns the dict of modified model-field arrays keyed by LEGGED_MODEL_FIELDS.
    """
    floor_geom_id, torso_body_id = consts
    friction_floor = scalars["friction_floor"]
    torso_mass_add_kg = scalars["torso_mass_add_kg"]
    com_offset_x_m = scalars["com_offset_x_m"]
    com_offset_y_m = scalars["com_offset_y_m"]
    kp_scale = scalars["kp_scale"]
    kd_scale = scalars["kd_scale"]
    com_offset_z_m = scalars["com_offset_z_m"]
    joint_frictionloss_scale = scalars["joint_frictionloss_scale"]
    joint_armature_scale = scalars["joint_armature_scale"]
    # Optional GEOMETRY dim (.get so a space without it is a no-op: slope 0 => identity).
    floor_slope_rad = scalars.get("floor_slope_rad", 0.0)

    geom_friction = base_model.geom_friction.at[floor_geom_id, 0].set(friction_floor)
    body_mass = base_model.body_mass.at[torso_body_id].add(torso_mass_add_kg)
    body_ipos = base_model.body_ipos.at[torso_body_id, 0].add(com_offset_x_m)
    body_ipos = body_ipos.at[torso_body_id, 1].add(com_offset_y_m)
    body_ipos = body_ipos.at[torso_body_id, 2].add(com_offset_z_m)
    # Kp: gainprm[:,0] and biasprm[:,1] (both scaled by kp_scale). Valid for any
    # affine/position actuator (gainprm[:,0]=kp, biasprm[:,1]=-kp, biasprm[:,2]=-kd).
    actuator_gainprm = base_model.actuator_gainprm.at[:, 0].multiply(kp_scale)
    actuator_biasprm = base_model.actuator_biasprm.at[:, 1].multiply(kp_scale)
    # Kd: biasprm[:,2] scaled by kd_scale.
    actuator_biasprm = actuator_biasprm.at[:, 2].multiply(kd_scale)
    # dof_frictionloss/dof_armature are length-(6 + nu) (6 free-base + nu actuated);
    # [6:] addresses only the nu actuated joints, mirroring domain_randomize().
    dof_frictionloss = base_model.dof_frictionloss.at[6:].multiply(joint_frictionloss_scale)
    dof_armature = base_model.dof_armature.at[6:].multiply(joint_armature_scale)
    # Floor slope (GEOMETRY axis): tilt the ground plane by pitching the floor geom's
    # quaternion floor_slope_rad about the y-axis, composed with its base orientation
    # (Hamilton product q_pitch (x) q_base). The cheap flat-env geometry perturbation;
    # full heightfield terrain is a documented follow-up (needs a rough-terrain env).
    xp = _numeric_namespace()
    q_pitch = _pitch_quat_about_y(xp, floor_slope_rad)
    new_floor_quat = _quat_mul(xp, q_pitch, base_model.geom_quat[floor_geom_id])
    geom_quat = base_model.geom_quat.at[floor_geom_id].set(new_floor_quat)
    return {
        "geom_friction": geom_friction,
        "body_mass": body_mass,
        "body_ipos": body_ipos,
        "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
        "dof_frictionloss": dof_frictionloss,
        "dof_armature": dof_armature,
        "geom_quat": geom_quat,
    }
