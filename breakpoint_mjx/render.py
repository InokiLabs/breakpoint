"""Phase 6 + Upgrade #6: replay ONE recorded failure (single env) and write an mp4.

Exact reproduction: the render replays the SCORED rollout bit-for-bit by
(a) reconstructing the evaluator's per-env key from the record's (batch_key,
env_index) via fold_in, and (b) running the SAME rollout_one the evaluator vmaps
(collect_states=True). There is no divergent step loop here anymore, so the per-env
key / per-step-split divergences that used to make renders only approximate cannot
recur. The squeeze uses the ACTIVE target's model_fields, so it can't drift from
apply_thetas.

# VERIFY on Colab: the env.render(...) CALL signature (camera/height/width + the
# write_video fps) mirrors eval_nominal.py's confirmed call. The STATES list, however,
# is built like the prior render.py (reset state prepended, then the post-step states),
# which differs from eval_nominal.py (post-step states only). If env.render's signature
# differs, match eval_nominal.py:
#   frames = env.render(states[::render_every])
#   media.write_video(path, frames, fps=int(1/env.dt/2))
# VERIFY on Colab: the reconstructed key fold_in(uint32(batch_key), env_index) yields
# the SAME rollout as the evaluator row (final trunk_height matches the record).
"""
from __future__ import annotations
import os

import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np

from breakpoint_mjx.thetas import apply_thetas
from breakpoint_mjx.rollout import rollout_one


def fall_step(summary, thresholds):
    """First step at which the legged fall criterion holds, else None. Pure numpy.

    Same criterion as ``failure.component_signals``: the fall-height channel
    (``floor_height`` when present, else ``trunk_height``) below ``trunk_height_min`` OR
    ``up_proj`` below ``up_proj_min`` OR a non-finite value."""
    if not summary or "up_proj" not in summary:
        return None
    h = np.asarray(summary["floor_height"] if "floor_height" in summary
                   else summary.get("trunk_height"), dtype=float)
    up = np.asarray(summary["up_proj"], dtype=float)
    bad = ~np.isfinite(h) | ~np.isfinite(up)
    fallen = bad | (np.nan_to_num(h, nan=-1.0) < float(thresholds["trunk_height_min"])) | (
        np.nan_to_num(up, nan=-1.0) < float(thresholds["up_proj_min"]))
    return int(np.argmax(fallen)) if bool(fallen.any()) else None


def min_floor_height(summary):
    """Lowest torso height above the floor plane over a rollout (None if not recorded)."""
    if not summary:
        return None
    key = "floor_height" if "floor_height" in summary else "trunk_height"
    if key not in summary:
        return None
    v = np.asarray(summary[key], dtype=float)
    v = v[np.isfinite(v)]
    return float(v.min()) if v.size else None


def clip_len(n_steps, fall, dt, after_s=1.0):
    """Number of post-step states to keep: through ``after_s`` seconds past the fall step
    (all of them when the rollout never fell)."""
    if fall is None:
        return int(n_steps)
    return int(min(n_steps, fall + 1 + int(round(after_s / float(dt)))))


def release_memory(per_family=False):
    """Collect garbage and drop JAX's compile caches before rendering.

    Called once before the render phase (``per_family=False``): the search/CMC compiles
    are not reused by the render, so dropping them is free memory. Called before each
    family (``per_family=True``) it clears the caches only when ``BP_RENDER_CLEAR=1``:
    measured on the CPU demo (2026-09-27, 4 families), clearing before every family cut
    the peak from 4582 to 4442 MB RSS and 6484 to 6030 MB commit but doubled rendering
    time (584 s to 1230 s), because every family then recompiles. ``BP_RENDER_CLEAR=0``
    also skips the one-time clear. Returns whether the caches were cleared."""
    import gc

    gc.collect()
    v = os.environ.get("BP_RENDER_CLEAR", "").strip().lower()
    clear = (v == "1") if per_family else (v != "0")
    if clear:
        jax.clear_caches()
        gc.collect()
    return clear


def render_states(env, states, out_path, camera="track", mj_model=None, render_every=2,
                  height=480, width=640):
    """Write an mp4 of an already-simulated state list.

    ``mj_model`` renders with a different host model (e.g. the full-collision scene an
    independent C-MuJoCo replay ran on) through Playground's ``render_array``; default is
    ``env.render`` on the env's own model."""
    sub = list(states)[::render_every]
    if mj_model is not None:
        from mujoco_playground._src.mjx_env import render_array

        frames = render_array(mj_model, sub, height=height, width=width, camera=camera)
    else:
        frames = env.render(sub, camera=camera, height=height, width=width)
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    media.write_video(out_path, frames, fps=max(1, int(1 / env.dt / render_every)))
    return out_path


def render_failure(env, policy_fn, target, space, theta_vector, batch_key, env_index,
                   horizon, out_path, consts, base_model=None,
                   render_every=2, camera="track", height=480, width=640,
                   thresholds=None, clip_after_fall_s=None, max_steps=None,
                   return_info=False):
    """Replay one world and write its mp4.

    ``thresholds`` + ``clip_after_fall_s`` end the video that many seconds after the fall
    step; ``max_steps`` caps it (a nominal twin is cut at its counterexample's length).
    ``return_info=True`` returns ``(out_path, info)`` with the fall step, the number of
    steps shown and the lowest torso height above the floor over the WHOLE rollout (the
    physics-fidelity flag: a feet-only scene lets a fallen trunk sink through the floor).
    """
    # Guard: a reused failures.json from before exact-repro (Upgrade #6) lacks the
    # batch_key/env_index needed to reconstruct the eval key. Fail clearly instead of
    # crashing on int(None)/asarray(None) below.
    if batch_key is None or env_index is None:
        raise ValueError(
            "this failure record predates exact-repro (Upgrade #6); "
            "delete failures.json and re-run the search"
        )
    # Use the explicit ORIGINAL base model, never env.mjx_model: the evaluator's jit
    # trace and prior renders leave env._mjx_model mutated, so reading it here would
    # compound perturbations across families. Caller passes the captured original.
    base = base_model if base_model is not None else env.mjx_model
    # T7: the per-target summary hook, bound to this target's consts (opaque per-target
    # blob), so the replay summarizes exactly like the evaluator that scored it.
    summarize = lambda state: target.objective.summarize(state, consts)  # noqa: E731

    thetas = jnp.asarray(np.asarray(theta_vector, dtype=float)[None, :])  # (1, D)
    batched, _, rollout = apply_thetas(target, space, base, thetas, consts)
    # apply_thetas batches exactly target.model_fields (leading dim n=1 here); every
    # other leaf is the un-batched original, so squeeze just these to a single env model.
    model_i = batched.tree_replace(
        {f: getattr(batched, f)[0] for f in target.model_fields}
    )
    # Squeeze whatever rollout dims the space declared (n=1 here) into scalars.
    rollout_i = {k: v[0] for k, v in rollout.items()}

    # Reconstruct the EXACT evaluator key for this row: fold_in(batch_key, env_index).
    key = jax.random.fold_in(jnp.asarray(batch_key, dtype=jnp.uint32), int(env_index))

    summary, states = rollout_one(
        env, policy_fn, model_i, rollout_i, key, horizon, summarize,
        collect_states=True,
    )
    summary = {k: np.asarray(v) for k, v in summary.items()}
    fall = fall_step(summary, thresholds) if thresholds else None
    n = int(horizon)
    if clip_after_fall_s is not None and thresholds:
        n = clip_len(n, fall, env.dt, clip_after_fall_s)
    if max_steps is not None:
        n = min(n, int(max_steps))
    states = list(states)[: n + 1]          # states[0] is the reset state

    # camera is the ACTIVE target's camera (target.camera): the legged "track" camera
    # follows the trunk as the robot walks/falls; a manipulation target with no such camera
    # passes None -> env.render uses its default view.
    # VERIFY (Colab): env.render accepts camera=None (default view) for PandaPickCube.
    frames = env.render(states[::render_every], camera=camera, height=height, width=width)
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    media.write_video(out_path, frames, fps=max(1, int(1 / env.dt / render_every)))
    if return_info:
        return out_path, {"fall_step": fall, "n_steps": n,
                          "min_floor_height": min_floor_height(summary)}
    return out_path
