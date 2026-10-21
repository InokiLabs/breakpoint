"""Generic legged-target FACTORY: register any Playground floating-base legged env in
one line instead of one hand-written module per robot.

The G1 port proved the shared legged field math (targets/_legged.py) is N-agnostic: any
env satisfying "6-dof free base + N single-dof actuated joints, affine position
actuators" reuses ``build_legged_fields`` byte-for-byte, and ``validate_legged_model``
hard-fails at setup on anything else. So the ONLY robot-specific facts a new legged
target needs are:

  * the Playground env id,
  * the floor-geom / base-body NAMES (for constants), and
  * a fall-height threshold matched to the robot's scale (~half standing height).

``make_legged_target`` bundles exactly those. Constants resolve through CANDIDATE name
lists tried in order; on a miss the error lists the model's ACTUAL geom/body names, so a
wrong guess on a new Playground version is a one-line fix straight from the Colab
traceback (paste-back friendly), never a silent mis-index.

Import-safe: mujoco / mujoco_playground are imported INSIDE the callables. The factory
itself is pure metadata assembly and unit-tested on CPU.
"""
from __future__ import annotations

from breakpoint_mjx.targets import Target
from breakpoint_mjx.targets._legged import (
    LEGGED_MODEL_FIELDS,
    build_legged_fields,
    validate_legged_model,
)

# Tried IN ORDER. Floor planes are consistently named across Playground scenes;
# base-body naming varies by vendor (pelvis/torso/trunk/body/...), so robots pass
# their own ordering when it matters (e.g. humanoids put "pelvis" first).
DEFAULT_FLOOR_CANDIDATES = ("floor", "ground", "plane", "terrain")
DEFAULT_BASE_CANDIDATES = (
    "torso", "trunk", "base", "pelvis", "body", "base_link", "torso_link", "Trunk",
)

# Shared generic theta space (a copy of go1's proven 16 dims; scale/offset dims transfer
# across legged robots). A robot with materially different scale gets its own file, like
# configs/g1/theta_space.yaml widened the mass-add range for the heavier G1.
DEFAULT_THETA_SPACE = "configs/legged_default/theta_space.yaml"


def resolve_constants_from_candidates(env, floor_candidates, base_candidates):
    """Resolve (floor_geom_id, base_body_id) by trying candidate names in order.

    Fail-loud contract: if NO candidate matches, raise with the model's actual geom/body
    name lists in the message, so the fix (add the right name) is copy-pasteable from the
    traceback. Never returns a guessed index.

    # VERIFY (Colab): mj_model access matches resolve_constants_by_name (env.mj_model,
    # falling back to env.unwrapped.mj_model).
    """
    import mujoco

    m = getattr(env, "mj_model", None)
    if m is None:
        m = env.unwrapped.mj_model

    def names_of(objtype, count):
        return [mujoco.mj_id2name(m, objtype, i) or f"<unnamed {i}>" for i in range(count)]

    def find(objtype, cands, count, what):
        for name in cands:
            i = int(mujoco.mj_name2id(m, objtype, name))
            if i >= 0:
                return i, name
        raise ValueError(
            f"none of the candidate {what} names {list(cands)} exist in this model; "
            f"actual {what} names: {names_of(objtype, count)} -- add the right one to "
            f"the target's {what} candidates"
        )

    gid, gname = find(mujoco.mjtObj.mjOBJ_GEOM, floor_candidates, m.ngeom, "geom")
    bid, bname = find(mujoco.mjtObj.mjOBJ_BODY, base_candidates, m.nbody, "body")
    print(f"[target] resolved floor geom {gname!r} (id {gid}), base body {bname!r} (id {bid})",
          flush=True)
    return gid, bid


def make_legged_target(name, env_name, *, floor_candidates=DEFAULT_FLOOR_CANDIDATES,
                       base_candidates=DEFAULT_BASE_CANDIDATES, thresholds=None,
                       theta_space_path=DEFAULT_THETA_SPACE, checkpoint_name=None,
                       impl="jax", obs_non_sensor=None) -> Target:
    """Build a legged Target from names alone (see module docstring).

    ``obs_non_sensor`` is an optional ``env -> (slices, frame)`` naming the obs channels
    that are not sensor readings (command, last action, phase); see
    ``rollout.set_obs_non_sensor``. Without it every obs channel stays perturbable.

    ``thresholds`` should at minimum set ``trunk_height_min`` to ~half the robot's
    standing base height (the go1-scale default of 0.18 m under-triggers on anything
    taller; see the g1 module's derivation). All the structural safety comes from
    ``validate_legged_model`` at setup + the candidate-name fail-loud resolution.
    """
    ckpt = checkpoint_name or f"{name}_policy.pkl"

    def load_env():
        from mujoco_playground import registry

        from breakpoint_mjx.obs_adversary import set_obs_non_sensor

        env = registry.load(env_name, config_overrides={"impl": impl})
        if obs_non_sensor is not None:
            slices, frame = obs_non_sensor(env)
            set_obs_non_sensor(env, slices, frame=frame)
        return env

    def resolve_constants(env):
        validate_legged_model(env.mjx_model)
        return resolve_constants_from_candidates(env, floor_candidates, base_candidates)

    return Target(
        name=name,
        env_name=env_name,
        checkpoint_name=ckpt,
        theta_space_path=theta_space_path,
        model_fields=LEGGED_MODEL_FIELDS,
        load_env=load_env,
        resolve_constants=resolve_constants,
        build_fields=build_legged_fields,
        thresholds=thresholds,
    )
