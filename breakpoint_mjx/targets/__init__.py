"""Target abstraction (Upgrade #1): bundle everything robot-specific behind one
interface so the core search / eval / render machinery stays generic.

A Target captures the env, checkpoint, theta-space path, the MJX model fields it
batches, and the robot-specific callables (load_env, resolve_constants,
build_fields). Adding a robot is "write a target module + a theta_space.yaml,"
never editing the core.

Import-safe: heavy deps (jax, mujoco, mujoco_playground) are imported INSIDE the
target callables / helpers, not at module top, so this module imports on a
CPU-only box (no JAX/GPU). The Target dataclass is defined before the go1 import
so the registry can reference it without a circular-import hazard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

# The per-class failure objective + summary hooks (T7). Imported before Target so the
# dataclass can default to the legged objective. objectives imports only numpy/failure
# (no jax), so this stays import-safe on a CPU-only box.
from breakpoint_mjx.objectives import Objective
from breakpoint_mjx.objectives.legged import LEGGED_OBJECTIVE


@dataclass(frozen=True)
class Target:
    """Everything robot-specific, behind one interface.

    Fields
    ------
    name             : registry key, e.g. "go1".
    env_name         : mujoco_playground env id, e.g. "Go1JoystickFlatTerrain".
    checkpoint_name  : pickled-params filename, e.g. "go1_policy.pkl".
    theta_space_path : repo-relative YAML, e.g. "configs/go1/theta_space.yaml".
    model_fields     : tuple of MJX model fields apply_thetas batches (render
                       squeezes exactly these, so the squeeze can't drift).
    load_env         : () -> env. Force the MJX JAX backend inside.
    resolve_constants: (env) -> consts. By convention a (floor_geom_id,
                       torso_body_id) tuple: the geom build_fields modifies and
                       the torso body the rollout summary reads.
    build_fields     : (scalars, base_model, consts) -> dict. Per-row model-field
                       math; scalars is {model-dim name: scalar value} for one row
                       (vmapped over N by apply_thetas).
    thresholds       : OPTIONAL per-robot failure-threshold overrides (dict), or
                       None for "use the search.yaml defaults unchanged". Merged
                       OVER cfg.thresholds in run_pipeline (target keys win), so a
                       robot whose fall height differs from a knee-high Go1 (e.g. a
                       humanoid that stands ~0.8 m, so the 0.18 m fall cutoff would
                       essentially never trigger) can raise trunk_height_min without
                       touching the shared config. go1/barkour leave this None. A
                       manipulation target uses this to supply its OWN threshold keys
                       (success/grasp/drop/joint-margin distances) -- its objective
                       reads only those keys, ignoring the legged ones.
    objective        : the per-class failure objective + summary hooks (T7). Defaults to
                       the wrapped LEGGED objective, so go1/barkour/g1 (and any target
                       that predates T7) route through the byte-identical legged path.
                       A new vertical (manipulation, ...) sets this to its own Objective;
                       rollout_one calls objective.summarize, and scores/atlas/margins/
                       crossval call objective.component_signals/failure_score/descriptor.
    domain           : which Playground suite the env belongs to ("locomotion" default or
                       "manipulation"), used ONLY by policy.py to pick the right PPO config
                       (locomotion_params vs manipulation_params). Orthogonal to the
                       failure objective; kept here so training stays a one-liner per target.
    camera           : env camera name render_failure passes to env.render ("track" default
                       for the legged tracking camera; a target lacking it sets its own or
                       None for the env default).
    horizon          : OPTIONAL rollout-length override (env steps) or None to use the
                       shared search.yaml horizon. Lets a target match its trained episode
                       length so scoring + TTF normalization are correct.

    Rollout dims (kind == "rollout" in the theta space) are handled generically by
    apply_thetas / rollout_one; the recognized names are "obs_noise_scale" and
    "latency_steps". A space may declare neither, either, or both.

    The callables are stored as plain functions on the frozen instance, so they
    are NOT bound as methods (instance attributes don't trigger descriptor
    binding) -- target.load_env() calls the function with no implicit self.
    """

    name: str
    env_name: str
    checkpoint_name: str
    theta_space_path: str
    model_fields: tuple[str, ...]
    load_env: Callable[[], object]
    resolve_constants: Callable[[object], object]
    build_fields: Callable[[dict, object, object], dict]
    # Optional; defaults to None so go1/barkour (and any pre-existing target) are
    # unchanged. A dict of {threshold_name: value} overrides for this robot only.
    thresholds: Optional[dict] = None
    # T7: the failure objective + summary hooks. Defaults to the wrapped legged objective
    # so every existing target is byte-identical; a new vertical points this at its own.
    objective: Objective = LEGGED_OBJECTIVE
    # T7: Playground suite for the PPO config picker in policy.py ("locomotion" default).
    domain: str = "locomotion"
    # T7: the env camera render_failure passes to env.render. Defaults to "track" (the
    # legged locomotion tracking camera) so go1/barkour/g1 are unchanged; a target whose
    # model has no "track" camera sets its own name, or None for the env's default view.
    camera: Optional[str] = "track"
    # T7: OPTIONAL per-target rollout horizon override (env steps). None -> use the shared
    # search.yaml horizon (legged unchanged). A target trained on a shorter episode sets
    # this so the rollout length + the time-to-failure normalization match its training
    # (e.g. PandaPickCube's 150-step episode vs the 500-step locomotion default). Applied
    # in run_pipeline via dataclasses.replace(cfg, horizon=...), so evaluator / scoring /
    # margins / crossval all see the effective horizon.
    horizon: Optional[int] = None


def resolve_constants_by_name(env, geom_name: str, body_name: str):
    """Generic, name-based constants: resolve (geom_id, body_id) from the model.

    Uses mujoco's mj_name2id so a new robot does not depend on its randomizer
    module exposing module-level FLOOR_GEOM_ID / TORSO_BODY_ID constants (only the
    Go1 randomizer happens to). Returns the (floor_geom_id, torso_body_id) tuple
    convention shared by the rest of the engine.

    # VERIFY on Colab: a mujoco_playground MJX env exposes the host-side MjModel as
    # `env.mj_model`; if a target wraps the env, fall back to env.unwrapped.mj_model.
    """
    import mujoco

    m = getattr(env, "mj_model", None)
    if m is None:
        m = env.unwrapped.mj_model
    gid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
    bid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if gid < 0:
        raise ValueError(f"geom {geom_name!r} not found in model")
    if bid < 0:
        raise ValueError(f"body {body_name!r} not found in model")
    return gid, bid


# Registry. Imported at the bottom so Target / resolve_constants_by_name are defined
# before the target modules reference them.
from breakpoint_mjx.targets.go1 import GO1  # noqa: E402
from breakpoint_mjx.targets.barkour import BARKOUR  # noqa: E402
from breakpoint_mjx.targets.g1 import G1  # noqa: E402
from breakpoint_mjx.targets.panda import PANDA  # noqa: E402
from breakpoint_mjx.targets.generic import make_legged_target  # noqa: E402
from breakpoint_mjx.targets._legged import phase_humanoid_non_sensor  # noqa: E402

# Factory-made legged targets (targets/generic.py): one line per robot. Env ids +
# base-body names + fall thresholds are best-effort from the Playground registry/MJCFs;
# each is guarded fail-loud (validate_legged_model + candidate-name resolution that
# lists the model's REAL names on a miss), and the multi-policy Colab notebook's
# registry cell prints the installed env ids for paste-back correction.
# Fall thresholds ~= half the robot's nominal standing base height (the g1 rule).
# # VERIFY (Colab): env ids below against `registry.locomotion.ALL_ENVS`; base-body
# # candidate order against each MJCF's freejoint body.
SPOT = make_legged_target(
    "spot", "SpotFlatTerrainJoystick",
    base_candidates=("body", "trunk", "base", "torso"),
    thresholds={"trunk_height_min": 0.26},   # Spot body stands ~0.52 m
    # state ends [..., last_act 12, command 3]
    obs_non_sensor=lambda env: ([(-(int(env.action_size) + 3), int(env.action_size) + 3)],
                                None),
)
H1 = make_legged_target(
    "h1", "H1JoystickGaitTracking",
    base_candidates=("pelvis", "torso_link", "base", "torso"),
    thresholds={"trunk_height_min": 0.50},   # H1 pelvis stands ~0.98 m
    # JoystickGaitTracking state = [gyro 3, gravity 3, joint pos 19, joint vel 19,
    # last_act 19, command 3, qvel/qpos-error history, contact 2, phase 4, gait_freq 1,
    # gait 1, foot_height 1]: last_act + command at 44, the gait clock is the final 7.
    obs_non_sensor=lambda env: ([(44, 22), (-7, 7)], None),
)
T1 = make_legged_target(
    "t1", "T1JoystickFlatTerrain",
    base_candidates=("Trunk", "trunk", "torso", "pelvis", "base"),
    thresholds={"trunk_height_min": 0.35},   # Booster T1 trunk stands ~0.7 m
    obs_non_sensor=lambda env: (phase_humanoid_non_sensor(env), None),
)
BERKELEY_HUMANOID = make_legged_target(
    "berkeley_humanoid", "BerkeleyHumanoidJoystickFlatTerrain",
    base_candidates=("torso", "pelvis", "base", "trunk"),
    thresholds={"trunk_height_min": 0.30},   # ~0.85 m robot, base ~0.6 m
    obs_non_sensor=lambda env: (phase_humanoid_non_sensor(env), None),
)

TARGETS: dict[str, Target] = {
    GO1.name: GO1,
    BARKOUR.name: BARKOUR,
    G1.name: G1,
    PANDA.name: PANDA,
    SPOT.name: SPOT,
    H1.name: H1,
    T1.name: T1,
    BERKELEY_HUMANOID.name: BERKELEY_HUMANOID,
}


def get_target(name: str) -> Target:
    """Look up a registered target by name."""
    try:
        return TARGETS[name]
    except KeyError:
        raise KeyError(
            f"unknown target {name!r}; available: {sorted(TARGETS)}"
        ) from None
