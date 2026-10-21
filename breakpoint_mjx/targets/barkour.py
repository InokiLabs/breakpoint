"""Barkour target: Google's open-source MJX reference quadruped (DeepMind
Menagerie ``google_barkour_vb``), the SECOND bring-your-own-policy target and the
proof that the Target abstraction enables BYO (Upgrade #1 Definition of Done).

Same morphology class as the Go1 -- a 6-dof floating base plus 12 actuated joints,
with affine position actuators (``gainprm[:, 0] = kp``, ``biasprm[:, 1] = -kp``,
``biasprm[:, 2] = -kd``) and an 18-long dof vector whose ``[6:]`` slice is the 12
actuated joints -- so it REUSES the shared legged field math
(``targets._legged.build_legged_fields``) byte-for-byte. Only the env id, the
checkpoint, and the (floor_geom, torso_body) names differ from Go1.

Constants are resolved BY NAME (``targets.resolve_constants_by_name``) from the
model, NOT from a randomizer module's FLOOR_GEOM_ID / TORSO_BODY_ID. This is the
whole point of the abstraction, and it is required here: BarkourJoystick ships no
domain randomizer in the Playground locomotion registry, so the Go1
``inspect.getmodule(...).FLOOR_GEOM_ID`` path does not exist for it.

Confirmed facts (sources in the Task 3 report / commit message):
- Playground env id ............ "BarkourJoystick"  (locomotion registry _envs)
- Free-base body name .......... "torso"            (Menagerie barkour_vb_mjx.xml: <freejoint name="torso"/>)
- Floor plane geom name ........ "floor"            (Menagerie scene_mjx.xml: plane geom "floor")
- Actuators .................... 12 general/affine  (gainprm="50 0 0", biasprm="0 -50 -0.5")
- DOF layout .................... 6 free + 12 joints (qpos[7:], qvel[6:]) -> [6:] = the 12 actuated
- PPO config / network ......... default brax_ppo_config("BarkourJoystick") (no custom tuning)
- Pretrained checkpoint ........ NOT shipped by Playground -> reduced Colab train (like Go1)
- Domain randomizer ............ NONE registered -> trains with randomization_fn=None

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

ENV_NAME = "BarkourJoystick"
IMPL = "jax"
CHECKPOINT_NAME = "barkour_policy.pkl"

# Confirmed from the Menagerie google_barkour_vb MJCF (scene_mjx.xml + barkour_vb_mjx.xml).
FLOOR_GEOM_NAME = "floor"
TORSO_BODY_NAME = "torso"

# Shared single-source-of-truth legged fields (identical 7 to Go1).
MODEL_FIELDS = LEGGED_MODEL_FIELDS


def load_env():
    """Load the Barkour joystick env, forcing the MJX JAX backend.

    # VERIFY (Colab): registry.load("BarkourJoystick", config_overrides={"impl": "jax"})
    # -- the "impl" override is the same key Go1 uses; confirm Barkour's config accepts it
    # (older Playground builds may not expose "impl"; drop the override if so).
    """
    from mujoco_playground import registry

    from breakpoint_mjx.obs_adversary import set_obs_non_sensor

    env = registry.load(ENV_NAME, config_overrides={"impl": IMPL})
    # obs is a 15-frame history, newest first; each 31-dim frame is [yaw rate 1,
    # gravity 3, command 3, joint angles 12, last_act 12] (BarkourJoystick._get_obs).
    return set_obs_non_sensor(env, [(4, 3), (19, 12)], frame=31)


def resolve_constants(env):
    """Name-based (floor_geom_id, torso_body_id) via mj_name2id on "floor"/"torso".

    Does NOT depend on a Go1-style randomizer module exposing FLOOR_GEOM_ID /
    TORSO_BODY_ID (BarkourJoystick has no registered randomizer at all). Returns the
    (floor_geom_id, torso_body_id) tuple convention the evaluator/renderer expect.

    Also validates ONCE here (setup-time hook) that the model is the 6-dof-free-base +
    N-actuated-joint layout the shared field math assumes (Barkour: 12 actuators /
    18 dof) -- so a mismatched robot fails loudly instead of producing silently-wrong
    perturbations.
    """
    validate_legged_model(env.mjx_model)
    return resolve_constants_by_name(env, FLOOR_GEOM_NAME, TORSO_BODY_NAME)


# Barkour's per-row field math IS the shared legged field math -- same affine
# position actuators and 6+12 dof layout as Go1, so no robot-specific field code.
build_fields = build_legged_fields


BARKOUR = Target(
    name="barkour",
    env_name=ENV_NAME,
    checkpoint_name=CHECKPOINT_NAME,
    theta_space_path="configs/barkour/theta_space.yaml",
    model_fields=MODEL_FIELDS,
    load_env=load_env,
    resolve_constants=resolve_constants,
    build_fields=build_fields,
)
