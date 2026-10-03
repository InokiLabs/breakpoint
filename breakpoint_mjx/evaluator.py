"""GPU-parallel adversarial evaluator -- Phase 2 core.

This module is executed on Colab; the per-env-model vmap pattern has been
validated live. See colab/cells.md CELL 4 for the Phase 2 acceptance test.

Upgrade #1 (Target abstraction): make_evaluator takes a Target + consts instead of
raw geom/body ids, and apply_thetas dispatches the robot-specific field math.
Upgrade #6 (exact reproduction): the per-env rollout is the shared rollout_one
(also used by the renderer), and per-env keys are fold_in(key, i) rather than
split(key, N), so the `key` passed to evaluate IS the batch key and row i is
position-stable for any N.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from breakpoint_mjx.config import ThetaSpace
from breakpoint_mjx.thetas import apply_thetas
from breakpoint_mjx.rollout import LMAX, env_key, rollout_one  # noqa: F401  (LMAX re-exported)


def make_param_evaluator(
    env,
    make_policy,
    space: ThetaSpace,
    horizon: int,
    target,
    consts,
):
    """Like make_evaluator, but params are an ARGUMENT: evaluate(params, thetas, key).

    make_evaluator closes over a policy_fn that has params baked in, so a new
    checkpoint means a new callable, a new jit cache key, and a full recompile of the
    horizon-length scan. Threading params through as a traced argument means ONE
    compile serves every policy version -- the difference between a hardening sweep
    paying a dozen compiles and paying one.

    ``make_policy`` is params -> policy_fn(obs, key); see policy.make_policy_factory.
    """
    base_model = env.mjx_model
    summarize = lambda state: target.objective.summarize(state, consts)  # noqa: E731

    def _evaluate(params, thetas, key):
        policy_fn = make_policy(params)
        batched, in_axes, rollout = apply_thetas(target, space, base_model, thetas, consts)
        n = thetas.shape[0]
        keys = jax.vmap(lambda i: env_key(key, i))(jnp.arange(n))
        rollout_in_axes = {k: 0 for k in rollout}

        def one(model, rollout_i, row_key):
            return rollout_one(
                env, policy_fn, model, rollout_i, row_key, horizon, summarize,
                collect_states=False,
            )

        return jax.vmap(one, in_axes=(in_axes, rollout_in_axes, 0))(batched, rollout, keys)

    return jax.jit(_evaluate)


def make_evaluator(
    env,
    policy_fn,
    space: ThetaSpace,
    horizon: int,
    target,
    consts,
):
    """Build a jit-compiled, GPU-parallel evaluator for the given env + policy + target.

    The returned callable is compiled once on first call and cached thereafter.

    CONTRACT -- evaluate(thetas, key) returns a trajectory dict of shapes (N, H, ...)
    whose KEYS are the ACTIVE target's objective summary keys (T7). For a LEGGED target
    (the default objective) these are exactly:

      "trunk_height"  : (N, H)      -- trunk CoM height above ground (qpos[2])
      "up_proj"       : (N, H)      -- R[2,2]; ~1.0 when upright, negative when inverted
      "lin_vel"       : (N, H, 3)   -- body-frame linear velocity (R.T @ qvel[:3])
      "ang_vel"       : (N, H, 3)   -- world-frame base angular velocity (qvel[3:6])
      "command"       : (N, H, 3)   -- (vx, vy, yaw) velocity command

    A MANIPULATION target returns its own keys instead (obj_to_target / ee_to_obj /
    obj_height / joint_margin / reached); scores_from_trajectories + the atlas consume
    whatever keys the target's objective emits, via target.objective.*.

    Per-env key (Upgrade #6): row i uses fold_in(key, i). `key` is therefore the
    batch key; record it (as two uint32 ints) + env_index to reproduce a row's
    rollout exactly via render_failure / rollout_one.

    # VERIFY
    # up_proj ~1.0 upright; the Phase 3 fall criterion is up_proj < threshold (e.g. 0.5).
    # lin_vel is body-frame: R.T @ global_lin_vel, comparable directly to command.

    Parameters
    ----------
    env       : MJX env from target.load_env(); _mjx_model is patched per-env under vmap.
    policy_fn : callable (obs_dict, key) -> (action, extras) from make_policy_fn().
    space     : ThetaSpace from the target's theta_space.yaml.
    horizon   : rollout length in env steps (from search.yaml, typically 500).
    target    : the active Target (provides build_fields + model_fields).
    consts    : target.resolve_constants(env); by convention (floor_geom_id, torso_body_id).

    Returns
    -------
    evaluate : jit-compiled callable (thetas: (N, D), key) -> traj dict (N, H, ...).
    """
    base_model = env.mjx_model
    # T7: the per-target summary hook, bound to this target's consts. rollout_one calls
    # summarize(state) each step instead of hardcoding the legged summary; consts is an
    # opaque per-target blob (legged: (floor_geom_id, torso_body_id); manipulation: a
    # dict of object/gripper/joint ids). The full consts also feeds build_fields via
    # apply_thetas. For a legged target this is the byte-identical inline summary.
    summarize = lambda state: target.objective.summarize(state, consts)  # noqa: E731

    def one(model, rollout_i, key):
        """Single-env rollout via the shared rollout_one (vmapped over N below)."""
        return rollout_one(
            env, policy_fn, model, rollout_i, key, horizon, summarize,
            collect_states=False,
        )

    def _evaluate(thetas, key):
        batched, in_axes, rollout = apply_thetas(target, space, base_model, thetas, consts)
        # Per-row keys via fold_in over arange(N): row i == fold_in(key, i), so the
        # batch key is `key` and rows are position-stable regardless of N.
        n = thetas.shape[0]
        keys = jax.vmap(lambda i: env_key(key, i))(jnp.arange(n))
        # Map axis 0 over whatever rollout dims the space declared (keys of rollout).
        rollout_in_axes = {k: 0 for k in rollout}
        return jax.vmap(
            one,
            in_axes=(in_axes, rollout_in_axes, 0),
        )(batched, rollout, keys)

    evaluate = jax.jit(_evaluate)
    return evaluate


def make_demo_collector(env, policy_fn, space, horizon, target, consts):
    """Like make_evaluator, but the trajectory dict ALSO carries ``obs_state`` (N, H,
    obs_dim) and ``action`` (N, H, act_dim) -- the exact (input, label) pairs a
    behavior-cloned student needs (portfolio distillation, RMA-style teacher-student:
    Kumar et al. 2021). Same rollout body, same key discipline, so a demo of world i is
    the byte-identical rollout the evaluator scored for world i under the same batch key.
    Collecting is trace-time, so make_evaluator's own throughput is unaffected."""
    base_model = env.mjx_model
    summarize = lambda state: target.objective.summarize(state, consts)  # noqa: E731

    def one(model, rollout_i, key):
        return rollout_one(env, policy_fn, model, rollout_i, key, horizon, summarize,
                           collect_states=False, collect_obs_act=True)

    def _collect(thetas, key):
        batched, in_axes, rollout = apply_thetas(target, space, base_model, thetas, consts)
        n = thetas.shape[0]
        keys = jax.vmap(lambda i: env_key(key, i))(jnp.arange(n))
        rollout_in_axes = {k: 0 for k in rollout}
        return jax.vmap(one, in_axes=(in_axes, rollout_in_axes, 0))(batched, rollout, keys)

    return jax.jit(_collect)
