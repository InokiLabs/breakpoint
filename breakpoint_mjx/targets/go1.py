"""Go1 reference target: the first-class Unitree Go1 locomotion target.

Preserves EXACT current behavior so nothing regresses:
- model_fields is the shared LEGGED_MODEL_FIELDS tuple (the proven 7 dynamics fields
  plus the T6 floor-slope geom_quat = 8; sourced from targets._legged so it can never
  drift from apply_thetas / render).
- resolve_constants uses the proven Go1 randomizer-module constants
  (FLOOR_GEOM_ID / TORSO_BODY_ID via inspect.getmodule). The name-based generic
  path (targets.resolve_constants_by_name) is the option for other robots.
- build_fields IS the shared legged field math (targets._legged), which is
  the former thetas._build_one body verbatim -- so Go1 output is byte-identical
  while other legged targets (barkour, g1) reuse the same helper instead of copy-paste.

Import-safe: jax / mujoco_playground are imported INSIDE callables, so importing
this module does not require a GPU/JAX install.
"""
from __future__ import annotations

from breakpoint_mjx.targets import Target
from breakpoint_mjx.targets._legged import (
    LEGGED_MODEL_FIELDS,
    build_legged_fields,
    validate_legged_model,
)

ENV_NAME = "Go1JoystickFlatTerrain"
IMPL = "jax"
CHECKPOINT_NAME = "go1_policy.pkl"

# Single source of truth: the MJX model fields apply_thetas batches for Go1.
# render imports this from the active target (target.model_fields) so the squeeze
# can never drift from apply_thetas. Shared with all legged targets.
MODEL_FIELDS = LEGGED_MODEL_FIELDS


def load_env():
    """Load the Go1 flat-terrain joystick env, forcing the MJX JAX backend.

    Actuation-faithful BYO: when ``BP_OBS_SPEC`` names a contract whose ``actuation:``
    block says ``honor: true``, the actuators are set to the policy's own PD gains on
    both the mj model and the MJX model (``adapters/actuation.py``) BEFORE any evaluator
    captures ``env.mjx_model``, so every kp/kd_scale multiplier applies on top of the
    policy's gains. ``BP_HONOR_ACTUATION=0`` keeps the native Kp 35 / Kd 0.5.
    """
    from mujoco_playground import registry

    from breakpoint_mjx.adapters.actuation import maybe_apply_from_env

    env = registry.load(ENV_NAME, config_overrides={"impl": IMPL})
    maybe_apply_from_env(env, "go1")
    return env


def resolve_constants(env):
    """Proven Go1 constants: (FLOOR_GEOM_ID, TORSO_BODY_ID) from the randomizer module.

    These are integer constants defined at module level in the Playground
    randomizer for Go1JoystickFlatTerrain. Returns the (floor_geom_id,
    torso_body_id) tuple that build_fields and the evaluator/renderer expect.

    Also validates ONCE here (resolve_constants is the setup-time hook) that the
    model is the 6-dof-free-base + N-actuated-joint layout the shared field math
    assumes (Go1: 12 actuators / 18 dof).
    """
    import inspect

    from mujoco_playground import registry  # imported here to keep import-safe

    validate_legged_model(env.mjx_model)
    mod = inspect.getmodule(registry.get_domain_randomizer(ENV_NAME))
    return mod.FLOOR_GEOM_ID, mod.TORSO_BODY_ID


# Go1's per-row field math IS the shared legged field math (floating base + 12
# actuated joints, affine position actuators). Pointed at the single source so it
# stays byte-identical to the original verbatim _build_one and other legged targets
# (barkour, g1) reuse the same code path. Kept as a module-level name so any direct
# import (or target.build_fields) resolves to that single source.
build_fields = build_legged_fields


GO1 = Target(
    name="go1",
    env_name=ENV_NAME,
    checkpoint_name=CHECKPOINT_NAME,
    theta_space_path="configs/go1/theta_space.yaml",
    model_fields=MODEL_FIELDS,
    load_env=load_env,
    resolve_constants=resolve_constants,
    build_fields=build_fields,
)


def full_collision_scene():
    """(xml_path, assets) of Playground's FULL-collision Go1 flat scene.

    The search runs the feet-only scene (only the four foot spheres collide with the
    floor), so after a genuine tip-over the collision-less trunk sinks through the floor.
    The independent replay (crossval_engines.replay_minimal_counterexamples) re-simulates
    each certified minimal counterexample in C MuJoCo on THIS scene instead: hip, thigh,
    calf and trunk primitives all collide with the floor. A genuinely different contact
    geometry, not a re-seed. Imported lazily so this module stays import-safe.
    """
    from mujoco_playground._src.locomotion.go1 import base, go1_constants

    return str(go1_constants.FULL_COLLISIONS_FLAT_TERRAIN_XML), base.get_assets()
