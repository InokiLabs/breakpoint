"""Nominal (un-perturbed) rollout validation for the frozen Go1 policy.

Run on Colab after train_policy() has written go1_policy.pkl:
    python scripts/eval_nominal.py

Reads  BP_OUT env var (default "out") and works under its per-target go1 subdir
(<BP_OUT>/go1) for both the saved params and mp4 output.
Reports fall count, mean episode reward, per-rollout min trunk height, and
prints state.info / state.metrics keys for Phase 3 field wiring.
"""

import os

import jax
import numpy as np
import mediapy as media

from breakpoint_mjx.policy import load_env, make_policy_fn

# go1-only helper: artifacts live under the per-target subdir, matching train.py /
# run_pipeline.py (which now namespace BP_OUT by target).
OUT = os.path.join(os.environ.get("BP_OUT", "out"), "go1")

# VERIFY: if env.render(list_of_states) has a different signature on the installed
# version of mujoco_playground, inspect with:
#   help(env.render)
# The goal is a list/array of RGB frames (H, W, 3) suitable for mediapy.write_video.
# Adjust the call accordingly (e.g. it may accept a State pytree directly, or need
# individual state objects).


def main(k=5, horizon=500):
    env = load_env()
    policy = make_policy_fn(out_dir=OUT)

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    pol = jax.jit(policy)

    key = jax.random.PRNGKey(0)
    falls = 0
    rewards = []
    min_h = []
    rollout = []  # states from rollout 0 for rendering

    for i in range(k):
        key, sk = jax.random.split(key)
        s = reset(sk)
        ep_r = 0.0
        mh = 1e9

        for t in range(horizon):
            sk, ak = jax.random.split(sk)
            act, _ = pol(s.obs, ak)
            s = step(s, act)
            ep_r += float(s.reward)
            mh = min(mh, float(s.data.qpos[2]))
            if i == 0:
                rollout.append(s)

        falls += int(mh < 0.18)
        rewards.append(ep_r)
        min_h.append(mh)

    # Print keys so Phase 3 can wire exact tracking fields.
    print("state.info keys:   ", list(rollout[0].info.keys()))
    print("state.metrics keys:", list(rollout[0].metrics.keys()))

    print(
        f"NOMINAL: falls={falls}/{k}  mean_reward={np.mean(rewards):.1f}  "
        f"min_trunk_heights={np.round(min_h, 3)}"
    )

    # Render every other frame from rollout 0 with the tracking camera (follows trunk).
    frames = env.render(rollout[::2], camera="track", height=480, width=640)
    path = os.path.join(OUT, "nominal_walk.mp4")
    media.write_video(path, frames, fps=int(1 / env.dt / 2))
    print("wrote", path)


if __name__ == "__main__":
    main()
