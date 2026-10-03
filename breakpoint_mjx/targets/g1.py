"""Unitree G1 target: the first HUMANOID target, and the cross-morphology proof that
the engine generalizes from quadrupeds to any floating-base legged robot.

The G1 is NOT the same morphology class as the Go1/Barkour quadrupeds -- it is a
biped with ~3x the dof -- yet it satisfies the ONE structural invariant the shared
field math depends on: a 6-dof free base followed by N actuated single-dof joints,
affine/position actuators (``gainprm[:, 0] = kp``, ``biasprm[:, 1] = -kp``,
``biasprm[:, 2] = -kd``) and a length-(6 + N) dof vector whose ``[6:]`` slice is
exactly the actuated joints. So it REUSES the shared legged field math
(``targets._legged.build_legged_fields``) byte-for-byte; only the env id, checkpoint,
the ``(floor_geom, base_body)`` names, the dof count, and the per-target failure
thresholds differ. This is exactly what generalizing the quadruped-only field math
into the shared ``targets._legged`` helper unlocked: the field math already used the
``[6:]`` slice + per-actuator Kp/Kd, both of which generalize to N=29.

Confirmed facts (sources in the report / commit message):
- Playground env id ............ "G1JoystickFlatTerrain"  (locomotion registry; also
                                  "G1JoystickRoughTerrain"). Class g1.joystick.Joystick.
- Free-base body name .......... "pelvis"  -- the body carrying <freejoint
                                  name="floating_base_joint"/> in g1_mjx_feetonly.xml,
                                  at pos="0 0 0.793" (nominal pelvis standing height).
                                  This is what we use for torso_body_id: qpos[2] is the
                                  pelvis free-joint z (standing height) and the rollout's
                                  up_proj = pelvis xmat[2,2], so the height and the up-axis
                                  read the SAME floating base. (Playground's g1_constants
                                  ROOT_BODY is "torso_link", used only for its internal
                                  sensors; the freejoint body is "pelvis".)
- Floor plane geom name ........ "floor"   (scene_mjx_feetonly_flat_terrain.xml plane geom)
- Actuators .................... 29 position actuators (6+6 legs, 3 waist, 7+7 arms) in
                                  g1_mjx_feetonly.xml -> 6 free + 29 = 35 dof; [6:] = the
                                  29 actuated joints. validate_legged_model checks
                                  dof == 6 + nu at setup, so a version-drifted count
                                  (some builds expose a 23-dof reduced G1) fails loudly
                                  rather than silently mis-slicing.
- Obs / command contract ....... _get_obs returns {"state": ..., "privileged_state": ...}
                                  and the velocity command lives in info["command"] =
                                  hstack([vx, vy, yaw]) -- matches the rollout/summary
                                  contract (st.obs["state"], st.info["command"]). rollout_one
                                  perturbs only obs["state"] and passes privileged_state
                                  through unchanged via {**st.obs, ...}.
- Domain randomizer ............ REGISTERED (g1.randomize.domain_randomize), so the policy
                                  trains WITH DR; policy.py's try/except tolerates either way.
- Pretrained checkpoint ........ NOT shipped by Playground -> train on Colab. A humanoid
                                  is materially pricier to train than a quadruped (more dof,
                                  longer to converge), so budget extra Colab time.

# VERIFY (Colab): the actuator count is 29 in g1_mjx_feetonly.xml; if a Playground build
#   loads a reduced (e.g. 23-dof) G1, validate_legged_model will raise with the actual
#   count -- adjust nothing in the field math (it is N-agnostic), just expect the message.
# VERIFY (Colab): registry.load("G1JoystickFlatTerrain", config_overrides={"impl": "jax"})
#   -- same "impl" key Go1 uses; drop the override if an older build rejects it.
# VERIFY (Colab): up_proj read from "pelvis" reads the base orientation (not "torso_link",
#   which the waist joints can rotate relative to the pelvis); confirm up_proj ~1.0 upright.

Import-safe: jax / mujoco / mujoco_playground are imported INSIDE callables, so
importing this module does not require a GPU/JAX install.
"""
from __future__ import annotations

from breakpoint_mjx.targets import Target, resolve_constants_by_name
from breakpoint_mjx.targets._legged import (
    LEGGED_MODEL_FIELDS,
    build_legged_fields,
    validate_legged_model,
)

ENV_NAME = "G1JoystickFlatTerrain"
IMPL = "jax"
CHECKPOINT_NAME = "g1_policy.pkl"

# Confirmed from the Playground G1 MJCF (scene_mjx_feetonly_flat_terrain.xml + g1_mjx_feetonly.xml).
FLOOR_GEOM_NAME = "floor"
# The floating-base body = the one carrying the <freejoint>. qpos[2] is its z (standing
# height) and the rollout up_proj = its xmat[2,2], so both read the SAME base. (NOT
# Playground's ROOT_BODY "torso_link", which sits above the waist joints.)
BASE_BODY_NAME = "pelvis"

# Shared single-source-of-truth legged fields (identical 7 to Go1/Barkour; N-agnostic).
MODEL_FIELDS = LEGGED_MODEL_FIELDS

# Per-target failure-threshold override. The default trunk_height_min: 0.18 suits a
# knee-high Go1 (nominal trunk ~0.28 m) but the G1 pelvis stands at ~0.793 m (the
# freejoint pos in g1_mjx_feetonly.xml), so a 0.18 m cutoff would essentially NEVER
# trigger -- the humanoid would have to be lying flat to register a fall. A robust
# "fallen" cutoff is roughly half the standing height: 0.793 / 2 ~= 0.40 m, low enough
# that a still-walking G1 (pelvis ~0.7-0.8 m) never trips it, high enough that a real
# collapse/squat-down (pelvis dropping below ~0.4 m) reads as a fall. up_proj_min and
# the tracking/ang-vel thresholds are orientation/scale-agnostic, so we leave them at
# the search.yaml defaults; only the height cutoff is robot-scale-specific.
THRESHOLDS = {"trunk_height_min": 0.40}


def load_env():
    """Load the G1 flat-terrain joystick env, forcing the MJX JAX backend."""
    from mujoco_playground import registry

    return registry.load(ENV_NAME, config_overrides={"impl": IMPL})


def resolve_constants(env):
    """Name-based (floor_geom_id, base_body_id) via mj_name2id on "floor"/"pelvis".

    Does NOT depend on a Go1-style randomizer module exposing FLOOR_GEOM_ID /
    TORSO_BODY_ID. Returns the (floor_geom_id, torso_body_id) tuple convention the
    evaluator/renderer expect (torso_body_id == the pelvis floating-base body here).

    Also validates ONCE here (setup-time hook) that the model is the 6-dof-free-base +
    N-actuated-joint layout the shared field math assumes (G1: 29 actuators / 35 dof) --
    so a mismatched/version-drifted model fails loudly instead of mis-slicing.
    """
    validate_legged_model(env.mjx_model)
    return resolve_constants_by_name(env, FLOOR_GEOM_NAME, BASE_BODY_NAME)


# G1's per-row field math IS the shared legged field math -- same affine position
# actuators and 6 + N dof layout as the quadrupeds, just N=29, so no robot-specific
# field code. The [6:] slice and per-actuator Kp/Kd already generalize.
build_fields = build_legged_fields


G1 = Target(
    name="g1",
    env_name=ENV_NAME,
    checkpoint_name=CHECKPOINT_NAME,
    theta_space_path="configs/g1/theta_space.yaml",
    model_fields=MODEL_FIELDS,
    load_env=load_env,
    resolve_constants=resolve_constants,
    build_fields=build_fields,
    thresholds=THRESHOLDS,
)
