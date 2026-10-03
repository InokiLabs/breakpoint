"""Shared per-row MJX field math + structural validator for a FIXED-BASE ARM target.

A manipulation target (Franka ``PandaPickCube``) is structurally the OPPOSITE of a
legged one: the arm is BOLTED to the world (no 6-dof free base -- joint 0 is a hinge),
and the free joint in the model belongs to the manipulated OBJECT, not the robot. So the
legged assumptions do NOT apply and MUST NOT be reused:

  * there is no free base, so the ``qpos[2]`` / ``qvel[:3]`` / ``qvel[3:6]`` base reads
    are meaningless here (the manipulation objective reads object/gripper poses instead);
  * the actuated joints are NOT the ``[6:]`` dof slice -- the arm joints come FIRST and
    the object's free-joint dofs come LAST, so ``[6:]`` would perturb the wrong dofs.

``build_manipulation_fields`` therefore perturbs arm-appropriate quantities addressed by
EXPLICIT index arrays resolved once at setup (in the target's resolve_constants) and
carried in ``consts``: the object payload mass, the object + floor friction, the arm
actuator gain (Kp), and the arm dofs' friction/armature. (The Panda uses ``position``
actuators, whose bias has no Kd term, so only Kp is scaled -- honest to the model.)

``validate_manipulation_model`` is the fixed-base mirror of ``validate_legged_model``: it
FAILS LOUD if joint 0 is a free base (that would be a floating-base/legged model, not an
arm) or if there is no actuator, rather than silently mis-slicing.

Import-safe: NO jax at module top. ``build_manipulation_fields`` uses only the
``x.at[idx].{multiply}`` update API on whatever array type ``base_model`` carries (real
jax arrays on Colab; the numpy-backed fake in the tests), so this imports on a CPU box.
"""
from __future__ import annotations

# The MJX model fields a manipulation target batches (mirrors LEGGED_MODEL_FIELDS). NO
# body_ipos/com or geom_quat/floor-slope (locomotion-only); instead the object mass +
# object/floor friction + arm Kp + arm dof friction/armature.
MANIPULATION_MODEL_FIELDS = (
    "body_mass",
    "geom_friction",
    "actuator_gainprm",
    "actuator_biasprm",
    "dof_frictionloss",
    "dof_armature",
)

# MuJoCo mjtJoint enum values used by the validator / constant resolver.
MJ_JNT_FREE = 0
MJ_JNT_BALL = 1
MJ_JNT_SLIDE = 2
MJ_JNT_HINGE = 3

# MuJoCo mjtTrn enum value used to identify JOINT-transmission actuators (arm actuators),
# as opposed to the tendon-driven gripper (mjTRN_TENDON == 3).
MJ_TRN_JOINT = 0


def validate_manipulation_model(base_model):
    """Fail loudly at setup if ``base_model`` is NOT a fixed-base arm.

    Two checks (the fixed-base mirror of ``validate_legged_model``):

    1. ``jnt_type[0] != mjJNT_FREE`` -- the ARM's first joint must be a hinge/slide, i.e.
       the arm is bolted to the world. A free joint 0 means a floating base (a legged
       model), for which the ``[6:]`` legged field math and base-pose reads -- NOT the
       manipulation ones -- would apply; reject it here rather than mis-model silently.
       (The manipulated OBJECT may still carry its own free joint later in the layout;
       only joint 0, the robot's root, is constrained.)
    2. at least one actuator (``actuator_gainprm.shape[0] >= 1``) -- an un-actuated
       "arm" has nothing to perturb / drive.

    Raises
    ------
    ValueError : if joint 0 is a free base (naming the actual type), or if there are no
                 actuators.
    """
    jnt0 = int(base_model.jnt_type[0])
    if jnt0 == MJ_JNT_FREE:
        raise ValueError(
            "manipulation field math assumes a FIXED-BASE arm (joint 0 is a hinge/slide "
            f"bolted to the world), but got jnt_type[0]={jnt0} == mjJNT_FREE (0). A free "
            "joint 0 is a FLOATING base (a legged robot), for which the legged objective + "
            "[6:] field math apply, not the manipulation ones. Use a legged target for a "
            "floating-base robot; this fixed-base arm path is rejected here."
        )
    nu = base_model.actuator_gainprm.shape[0]
    if nu < 1:
        raise ValueError(
            f"manipulation field math needs at least one actuator, got nu={nu}. A "
            "fully-passive model has no arm to drive or perturb."
        )


def build_manipulation_fields(scalars, base_model, consts):
    """Per-row manipulation model-field math (arm-appropriate; NOT the legged [6:] slice).

    Parameters
    ----------
    scalars    : dict {model-dim name: scalar theta value} for one row (vmapped).
    base_model : the captured original MJX model (broadcast, not vmapped).
    consts     : the panda target's resolve_constants() dict. Reads the object body id,
                 the object geom ids, the floor geom id, and the arm dof indices; every
                 index is a concrete array resolved once at setup, so no [6:] guessing.

    Returns the dict of modified model-field arrays keyed by MANIPULATION_MODEL_FIELDS.
    """
    obj_body_id = consts["obj_body_id"]
    obj_geom_ids = consts["obj_geom_ids"]
    floor_geom_id = consts["floor_geom_id"]
    arm_dof_ids = consts["arm_dof_ids"]
    arm_actuator_ids = consts["arm_actuator_ids"]

    object_mass_scale = scalars["object_mass_scale"]
    object_friction_scale = scalars["object_friction_scale"]
    floor_friction_scale = scalars["floor_friction_scale"]
    kp_scale = scalars["kp_scale"]
    joint_frictionloss_scale = scalars["joint_frictionloss_scale"]
    joint_armature_scale = scalars["joint_armature_scale"]

    # Payload: scale the manipulated object's body mass (heavier/lighter cube).
    body_mass = base_model.body_mass.at[obj_body_id].multiply(object_mass_scale)
    # Grip / surface friction: the object's sliding friction (column 0) and the floor's.
    geom_friction = base_model.geom_friction.at[obj_geom_ids, 0].multiply(object_friction_scale)
    geom_friction = geom_friction.at[floor_geom_id, 0].multiply(floor_friction_scale)
    # Arm actuator Kp: gainprm[:,0]=kp and biasprm[:,1]=-kp for a position actuator, both
    # scaled by kp_scale on the ARM actuators ONLY (arm_actuator_ids, resolved at setup) --
    # NOT the tendon-driven gripper, whose gain is grip strength and would couple with the
    # drop objective. (Panda position actuators have no Kd term, so none is scaled.)
    actuator_gainprm = base_model.actuator_gainprm.at[arm_actuator_ids, 0].multiply(kp_scale)
    actuator_biasprm = base_model.actuator_biasprm.at[arm_actuator_ids, 1].multiply(kp_scale)
    # Arm joint friction / armature: ONLY the arm dofs (explicit indices), NOT [6:] and
    # NOT the object's free-joint dofs.
    dof_frictionloss = base_model.dof_frictionloss.at[arm_dof_ids].multiply(joint_frictionloss_scale)
    dof_armature = base_model.dof_armature.at[arm_dof_ids].multiply(joint_armature_scale)

    return {
        "body_mass": body_mass,
        "geom_friction": geom_friction,
        "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
        "dof_frictionloss": dof_frictionloss,
        "dof_armature": dof_armature,
    }
