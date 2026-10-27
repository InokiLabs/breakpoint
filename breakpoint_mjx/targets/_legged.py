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
    # Explicit <contact><pair> entries carry their OWN friction, which overrides the geoms'
    # (MuJoCo semantics). G1's feet touch the floor only through such pairs, so setting
    # geom_friction alone left the friction axis a silent no-op there (measured: identical
    # trajectories at floor friction 0.05 and 1.0). Pairs that include the floor get the
    # same coefficient; a model without pairs (go1, t1, ...) has a (0, 5) array, a no-op.
    "pair_friction",
    # Extended physics axes (optional dims; each is a no-op unless its dim is declared,
    # so a 16-dim space builds exactly the fields it did before):
    #   motor_strength_scale       actuator torque limits (battery sag, derating)
    #   fault_motor_frac/_strength   one weak motor: its Kp, Kd and torque limit scaled
    #   joint_damping_scale        dof_damping on the actuated joints
    #   floor_solref_timeconst     ground softness (grass, mats, mud); also on floor pairs
    #   floor_solref_dampratio     ground bounce
    # Torque limits live in actuator_forcerange on the quadrupeds and in jnt_actfrcrange
    # on the humanoids (G1, T1); both are scaled, whichever the model uses.
    "actuator_forcerange",
    "jnt_actfrcrange",
    "dof_damping",
    "geom_solref",
    "pair_solref",
)

EXTENDED_MODEL_DIMS = ("motor_strength_scale", "fault_motor_frac", "fault_strength",
                       "joint_damping_scale", "floor_solref_timeconst",
                       "floor_solref_dampratio")


def floor_pair_ids(base_model, floor_geom_id):
    """Indices of explicit contact pairs that include the floor geom (numpy, static)."""
    import numpy as np

    g1 = getattr(base_model, "pair_geom1", None)
    g2 = getattr(base_model, "pair_geom2", None)
    if g1 is None or g2 is None:
        return np.zeros(0, dtype=int)
    g1, g2 = np.asarray(g1), np.asarray(g2)
    return np.nonzero((g1 == floor_geom_id) | (g2 == floor_geom_id))[0]


def phase_humanoid_non_sensor(env):
    """Non-sensor obs channels of Playground's phase-clocked joystick humanoids (G1, T1,
    Berkeley Humanoid): ``state = [linvel 3, gyro 3, gravity 3, command 3, joint angles N,
    joint vel N, last_act N, phase 4]``, so the command sits at 9 and last_act + phase
    are the final N + 4 channels. For ``rollout.set_obs_non_sensor``."""
    n = int(env.action_size)
    return [(9, 3), (-(n + 4), n + 4)]


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
    pair_friction = getattr(base_model, "pair_friction", None)
    pair_ids = floor_pair_ids(base_model, floor_geom_id)
    if pair_friction is not None and len(pair_ids):
        # both tangential coefficients; spin/roll friction keep the model's values
        pair_friction = pair_friction.at[pair_ids, 0].set(friction_floor)
        pair_friction = pair_friction.at[pair_ids, 1].set(friction_floor)
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
    out = {
        "geom_friction": geom_friction,
        "body_mass": body_mass,
        "body_ipos": body_ipos,
        "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
        "dof_frictionloss": dof_frictionloss,
        "dof_armature": dof_armature,
        "geom_quat": geom_quat,
        "pair_friction": pair_friction,
    }
    out.update(_extended_fields(xp, scalars, base_model, floor_geom_id, pair_ids, out))
    return out


def fault_index(xp, scalars, nu):
    """The weak motor: fault_motor_frac in [0, 1] picks actuator round(frac * (nu - 1)),
    so one space works for 12 motors (Go1) and 29 (G1). Actuator order is the model's."""
    return xp.round(xp.asarray(scalars.get("fault_motor_frac", 0.0)) * (nu - 1))


def actuator_strength(xp, scalars, nu):
    """Per-actuator torque multiplier, (nu,): motor_strength_scale everywhere, times
    fault_strength on actuator round(fault_actuator). Ones when neither dim is declared."""
    s = xp.ones(nu) * scalars.get("motor_strength_scale", 1.0)
    if "fault_motor_frac" in scalars or "fault_strength" in scalars:
        idx = fault_index(xp, scalars, nu)
        hit = xp.arange(nu) == idx
        s = s * xp.where(hit, scalars.get("fault_strength", 1.0), 1.0)
    return s


def fault_gain(xp, scalars, nu):
    """Per-actuator Kp/Kd multiplier of the weak motor alone (ones without the dims)."""
    if "fault_motor_frac" not in scalars and "fault_strength" not in scalars:
        return None
    idx = fault_index(xp, scalars, nu)
    return xp.where(xp.arange(nu) == idx, scalars.get("fault_strength", 1.0), 1.0)


def _extended_fields(xp, scalars, base_model, floor_geom_id, pair_ids, base):
    """The extended-axis fields. A field whose dims are absent is passed through
    unchanged (the same array as base_model's), so older spaces are byte-identical."""
    import numpy as np

    out = {f: getattr(base_model, f, None) for f in
           ("actuator_forcerange", "jnt_actfrcrange", "dof_damping", "geom_solref",
            "pair_solref")}
    if any(k in scalars for k in ("motor_strength_scale", "fault_motor_frac",
                                  "fault_strength")):
        nu = int(np.shape(base_model.actuator_trnid)[0])
        s = actuator_strength(xp, scalars, nu)
        out["actuator_forcerange"] = base_model.actuator_forcerange.at[:, :].multiply(s[:, None])
        jr = getattr(base_model, "jnt_actfrcrange", None)
        if jr is not None and jr.shape[0]:
            jid = np.asarray(base_model.actuator_trnid)[:, 0]
            out["jnt_actfrcrange"] = jr.at[jid].multiply(s[:, None])
        g = fault_gain(xp, scalars, nu)
        if g is not None:
            base["actuator_gainprm"] = base["actuator_gainprm"].at[:, 0].multiply(g)
            bias = base["actuator_biasprm"].at[:, 1].multiply(g)
            base["actuator_biasprm"] = bias.at[:, 2].multiply(g)
    if "joint_damping_scale" in scalars:
        out["dof_damping"] = base_model.dof_damping.at[6:].multiply(
            scalars["joint_damping_scale"])
    for col, name in ((0, "floor_solref_timeconst"), (1, "floor_solref_dampratio")):
        if name not in scalars:
            continue
        out["geom_solref"] = out["geom_solref"].at[floor_geom_id, col].set(scalars[name])
        if out["pair_solref"] is not None and len(pair_ids):
            out["pair_solref"] = out["pair_solref"].at[pair_ids, col].set(scalars[name])
    return out


def eval_config_overrides(env_name, impl):
    """``config_overrides`` for loading a Playground locomotion env for EVALUATION.

    Playground's joystick envs perturb the base on their own during training: G1, T1 and
    Berkeley Humanoid push it (``push_config``, up to 2 m/s every 5-10 s), Barkour kicks it
    (``velocity_kick`` up to 1 m/s), Go1's ``pert_config`` is off. Left on, every "nominal"
    rollout carries random shoves that no theta dim controls, so a calm-conditions walk
    check fails good policies (G1: fall rate 0.06-0.12 with pushes, 0.0 without) and search
    / jury / certificate results mix in disturbances they cannot attribute. BreakPoint's own
    push is a theta dim; the env's are switched off here.
    """
    from mujoco_playground import registry

    cfg = registry.get_default_config(env_name)
    out = {"impl": impl}
    for group in ("push_config", "pert_config"):
        if group in cfg and "enable" in cfg[group]:
            out[f"{group}.enable"] = False
    if "velocity_kick" in cfg:
        out["velocity_kick"] = [0.0, 0.0]
    return out

