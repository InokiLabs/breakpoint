"""Franka Emika Panda ``PandaPickCube`` target: the first MANIPULATION / robot-arm
target, and the proof that the engine generalizes from "stay upright" locomotion to a
completely different failure vertical via the per-target Objective abstraction (T7).

Unlike go1/barkour/g1 (floating-base LEGGED), this is a FIXED-BASE arm with a DIFFERENT
failure objective (task success / end-effector-to-target / dropped object / joint-limit),
so it routes through ``objectives.manipulation.MANIPULATION_OBJECTIVE`` instead of the
legged one, uses the fixed-base validator + arm field math (NOT the legged [6:] slice),
and its own manipulation theta space.

Confirmed env facts (MuJoCo Playground ``manipulation`` suite; sources in the T7 report /
commit message):
- Registry env id .............. "PandaPickCube"  (manipulation registry _envs ->
                                  franka_emika_panda.pick.PandaPickCube;
                                  variants: PandaPickCubeOrientation / PandaPickCubeCartesian).
- Class hierarchy .............. PandaPickCube(panda.PandaBase(mjx_env.MjxEnv)).
- FIXED BASE ................... the arm links attach to worldbody (NO freejoint on the
                                  arm); joint 0 is the arm's hinge "joint1". The <freejoint/>
                                  in mjx_single_cube.xml belongs to the "box" OBJECT, not the
                                  robot -- so validate_manipulation_model checks joint 0 is
                                  NOT free (fixed base), the mirror of the legged check.
- Actuators / dof layout ....... 8 actuators (7 arm position actuators "actuator1..7" + 1
                                  gripper). qpos length 16 = 9 arm-side (7 arm joints + 2
                                  finger joints) + 7 for the box free joint (3 pos + 4 quat);
                                  nv = 9 arm-side + 6 box free-joint = 15. So the actuated
                                  arm dofs come FIRST and the object dofs LAST -- the legged
                                  [6:] slice would perturb the wrong dofs; we resolve the arm
                                  dof indices explicitly at setup instead.
- Obs structure ................ _get_obs returns a PLAIN jp.array (NOT a {"state": ...}
                                  dict): concatenated qpos, qvel, gripper pos + orientation,
                                  object orientation, object-gripper offset, target-object
                                  offset, orientation diff, and ctrl deviation. rollout_one's
                                  obs-adversary path handles a flat-array obs generically
                                  (perturbing the whole vector), keeping the legged dict path
                                  byte-identical.
- info keys .................... {"rng", "target_pos", "reached_box"}. target_pos is the
                                  sampled target position (summary reads st.info["target_pos"]);
                                  reached_box is a gripper-reached-box success/reward proxy
                                  (threshold 0.012 m) used as the summary's "reached" signal.
- Key names (mj_name2id) ....... object body "box"; gripper site "gripper"; floor geom
                                  "floor"; target body "mocap_target"; finger pad geoms
                                  "left_finger_pad"/"right_finger_pad"; "hand_capsule".
- Reward / success ............. reward = gripper_box + box_target + no_floor_collision +
                                  robot_target_qpos; episode_length 150, ctrl_dt 0.02.
- Pretrained checkpoint ........ NOT shipped by Playground -> train on Colab
                                  (`python scripts/train.py --target panda`), like the
                                  legged targets. domain="manipulation" so policy.py picks
                                  manipulation_params.brax_ppo_config.

# VERIFY (Colab): PandaPickCube's default_config sets impl='warp' (newer Warp backend);
#   BreakPoint's whole engine is MJX (mjx.Model tree_replace / rollout_one), so we force
#   config_overrides={"impl": "jax"} like the legged targets. Confirm the installed
#   Playground build accepts impl="jax" for PandaPickCube (it historically ran under MJX);
#   if a build is Warp-only, this target needs the MJX variant / an older Playground.
# VERIFY (Colab): obs is a flat array -> env.observation_size is an int; make_ppo_networks
#   handles that. reached_box / target_pos live in state.info every step (set at reset,
#   carried through step); confirm the exact info keys on the installed build.
# VERIFY (Colab): the gripper SITE is named "gripper" and the object BODY "box"; if a build
#   renames them, adjust GRIPPER_SITE_NAME / OBJECT_BODY_NAME (mj_name2id raises otherwise).

Import-safe: jax / mujoco / mujoco_playground are imported INSIDE callables.
"""
from __future__ import annotations

from breakpoint_mjx.objectives.manipulation import MANIPULATION_OBJECTIVE
from breakpoint_mjx.targets import Target
from breakpoint_mjx.targets._manipulation import (
    MANIPULATION_MODEL_FIELDS,
    MJ_JNT_HINGE,
    MJ_JNT_SLIDE,
    MJ_TRN_JOINT,
    build_manipulation_fields,
    validate_manipulation_model,
)

ENV_NAME = "PandaPickCube"
IMPL = "jax"
CHECKPOINT_NAME = "panda_policy.pkl"

# Confirmed from the Playground franka_emika_panda pick env + mjx_single_cube.xml.
OBJECT_BODY_NAME = "box"
GRIPPER_SITE_NAME = "gripper"
FLOOR_GEOM_NAME = "floor"

MODEL_FIELDS = MANIPULATION_MODEL_FIELDS

# Manipulation failure thresholds (metres / unitless). Supplied HERE (not in the shared
# search.yaml) via Target.thresholds; they merge OVER the legged defaults and the
# manipulation objective reads ONLY these keys (ignoring the legged trunk/up_proj ones).
THRESHOLDS = {
    "success_dist": 0.05,      # object within 5 cm of target => task achieved
    "grasp_dist": 0.06,        # gripper within 6 cm of object => grasped
    "drop_dist": 0.15,         # once grasped, ee-to-object beyond 15 cm => dropped
    "joint_margin_min": 0.02,  # arm joint within 2% of a limit => violation
    # (the graded-miss normalizer is objectives.manipulation.OBJ_REACH_NORM, a fixed part
    # of the scoring definition -- NOT a threshold override, since failure_score has no
    # thresholds arg and it must stay consistent with behavior_descriptor.)
}


def load_env():
    """Load PandaPickCube, forcing the MJX JAX backend (see the impl VERIFY note above)."""
    from mujoco_playground import registry

    return registry.load(ENV_NAME, config_overrides={"impl": IMPL})


def resolve_constants(env):
    """Resolve the manipulation consts dict from the model (setup-time hook).

    Validates the fixed-base arm layout ONCE, then resolves by name/structure:
      * obj_body_id / gripper_site_id / floor_geom_id via mj_name2id;
      * obj_geom_ids   -- all geoms whose body is the object (for object friction);
      * arm_dof_ids    -- the dofs of every single-dof (hinge/slide) joint, i.e. the arm +
                          fingers, EXCLUDING the object's free joint (for friction/armature);
      * arm_actuator_ids -- the actuators driving a HINGE joint (the arm's revolute joints),
                          EXCLUDING the tendon-driven gripper, so the Kp scale perturbs arm
                          gain only (not grip strength, which would couple with the drop
                          objective). Falls back to all actuators if none resolve cleanly;
      * arm_qpos_adr / arm_jnt_lo / arm_jnt_hi -- the qpos addresses + limits of the LIMITED
                          single-dof joints (for the summary's joint-limit-margin signal).

    Returned as a dict (opaque to the core; only this target's build_fields + the
    manipulation objective's summarize interpret it), unlike the legged
    (floor_geom_id, torso_body_id) tuple.
    """
    import mujoco
    import numpy as np

    validate_manipulation_model(env.mjx_model)

    m = getattr(env, "mj_model", None)
    if m is None:
        m = env.unwrapped.mj_model

    def _bid(name):
        i = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name))
        if i < 0:
            raise ValueError(f"body {name!r} not found in PandaPickCube model")
        return i

    def _sid(name):
        i = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name))
        if i < 0:
            raise ValueError(f"site {name!r} not found in PandaPickCube model")
        return i

    def _gid(name):
        i = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name))
        if i < 0:
            raise ValueError(f"geom {name!r} not found in PandaPickCube model")
        return i

    obj_body_id = _bid(OBJECT_BODY_NAME)
    gripper_site_id = _sid(GRIPPER_SITE_NAME)
    floor_geom_id = _gid(FLOOR_GEOM_NAME)

    obj_geom_ids = np.where(np.asarray(m.geom_bodyid) == obj_body_id)[0].astype(int)

    arm_dof_ids, arm_qpos_adr, arm_jnt_lo, arm_jnt_hi = [], [], [], []
    for j in range(int(m.njnt)):
        jt = int(m.jnt_type[j])
        if jt in (MJ_JNT_HINGE, MJ_JNT_SLIDE):  # single-dof joints = arm + fingers
            arm_dof_ids.append(int(m.jnt_dofadr[j]))
            if int(m.jnt_limited[j]) == 1:
                arm_qpos_adr.append(int(m.jnt_qposadr[j]))
                lo, hi = m.jnt_range[j]
                arm_jnt_lo.append(float(lo))
                arm_jnt_hi.append(float(hi))

    if not arm_dof_ids:
        raise ValueError("no single-dof (hinge/slide) arm joints found in PandaPickCube model")
    if not arm_qpos_adr:
        raise ValueError(
            "no LIMITED single-dof joints found in PandaPickCube model; the joint-limit "
            "margin signal has nothing to measure (expected the Panda arm joints to be "
            "range-limited)"
        )

    # Arm actuators = those transmitting to a HINGE joint (the arm's revolute joints); the
    # gripper is tendon-driven, so this excludes it. The Kp scale then perturbs arm gain
    # only, not grip strength (which would couple with the drop objective). Fall back to all
    # actuators if the transmission info doesn't resolve a clean arm-only set.
    nu = int(m.actuator_gainprm.shape[0])
    arm_actuator_ids = []
    for a in range(nu):
        if int(m.actuator_trntype[a]) == MJ_TRN_JOINT:
            jid = int(m.actuator_trnid[a, 0])
            if 0 <= jid < int(m.njnt) and int(m.jnt_type[jid]) == MJ_JNT_HINGE:
                arm_actuator_ids.append(a)
    if not arm_actuator_ids:
        # Couldn't isolate arm-only actuators -> scale all (documented fallback).
        arm_actuator_ids = list(range(nu))

    return {
        "obj_body_id": obj_body_id,
        "gripper_site_id": gripper_site_id,
        "floor_geom_id": floor_geom_id,
        "obj_geom_ids": np.asarray(obj_geom_ids, dtype=int),
        "arm_dof_ids": np.asarray(arm_dof_ids, dtype=int),
        "arm_actuator_ids": np.asarray(arm_actuator_ids, dtype=int),
        "arm_qpos_adr": np.asarray(arm_qpos_adr, dtype=int),
        "arm_jnt_lo": np.asarray(arm_jnt_lo, dtype=float),
        "arm_jnt_hi": np.asarray(arm_jnt_hi, dtype=float),
    }


# The Panda's per-row field math IS the shared manipulation field math (fixed-base arm:
# object mass/friction + floor friction + arm Kp + arm dof friction/armature).
build_fields = build_manipulation_fields


PANDA = Target(
    name="panda",
    env_name=ENV_NAME,
    checkpoint_name=CHECKPOINT_NAME,
    theta_space_path="configs/panda/theta_space.yaml",
    model_fields=MODEL_FIELDS,
    load_env=load_env,
    resolve_constants=resolve_constants,
    build_fields=build_fields,
    thresholds=THRESHOLDS,
    objective=MANIPULATION_OBJECTIVE,
    domain="manipulation",
    # PandaPickCube's scene (mjx_scene.xml / mjx_panda.xml) defines NO named camera (no
    # locomotion "track" camera), so render with the env's default view.
    # VERIFY (Colab): env.render(states, camera=None, ...) yields frames for PandaPickCube;
    # if a named camera exists in the installed build, set it here for a nicer angle.
    camera=None,
    # PandaPickCube trains on a 150-step episode (default_config episode_length=150), so
    # roll out 150 steps (not the 500-step locomotion default) -- otherwise the arm runs
    # 3.3x past its trained episode and TTF normalizes against the wrong length.
    horizon=150,
)
