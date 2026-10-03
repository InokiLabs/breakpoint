"""Measure a policy's REALISTIC-WORLD robustness: the failure rate over a uniform sample of
each dim's PLAUSIBLE (sim-to-real) band -- the same honest 'is the policy robust' number the
hardening Pareto uses as its primary axis, but standalone so it can compare two policies.

Usage (GPU / Colab):
  BP_OUT=<dir> python scripts/eval_robustness.py --target go1 [--n 2048] [--seed 0]

Loads ``<BP_OUT>/<target>/<target>_policy.pkl``. To compare the frozen vs a robust policy
(trained with ``scripts/train.py --robust --out <other_dir>``), run this twice with the two
BP_OUT dirs and compare the printed rates. Prints a machine-grep line
``REALISTIC_FAILRATE=<x>`` for scripting.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breakpoint_mjx.config import (  # noqa: E402
    load_search_config,
    load_theta_space,
    merge_target_thresholds,
)
from breakpoint_mjx.targets import get_target  # noqa: E402
from breakpoint_mjx.paths import config_path  # noqa: E402


def plausible_band_sample(space, n, seed):
    """Uniform sample over each dim's plausible band (realistic worlds). Pure numpy."""
    import numpy as np

    names = space.names()
    plo = np.asarray([space.dims[nm].plausible[0] for nm in names])
    phi = np.asarray([space.dims[nm].plausible[1] for nm in names])
    rng = np.random.default_rng(int(seed))
    return (plo + rng.random((int(n), len(names))) * (phi - plo)).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Realistic-world (plausible-band) robustness eval.")
    ap.add_argument("--target", default="go1")
    ap.add_argument("--n", type=int, default=2048, help="plausible-band worlds to evaluate")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import numpy as np
    import jax
    import jax.numpy as jnp

    from breakpoint_mjx.evaluator import make_evaluator
    from breakpoint_mjx.policy import make_policy_fn
    from breakpoint_mjx.search import scores_from_trajectories

    target = get_target(a.target)
    OUT = os.path.join(os.environ.get("BP_OUT", "out"), target.name)
    ckpt = os.path.join(OUT, target.checkpoint_name)
    if not os.path.exists(ckpt):
        raise SystemExit(f"no policy at {ckpt}; train it first (BP_OUT={os.environ.get('BP_OUT')})")
    cfg = merge_target_thresholds(load_search_config(config_path("search.yaml")), target)
    space = load_theta_space(config_path(target.theta_space_path))

    policy_fn = make_policy_fn(out_dir=OUT, target=target)
    env = target.load_env()
    consts = target.resolve_constants(env)
    evaluate = make_evaluator(env, policy_fn, space, cfg.horizon, target, consts)

    thetas = plausible_band_sample(space, a.n, a.seed)
    traj = evaluate(jnp.asarray(thetas), jax.random.PRNGKey(int(a.seed)))
    scores, _, _ = scores_from_trajectories(traj, cfg.thresholds, cfg.horizon, target.objective)
    failrate = float((np.asarray(scores) >= cfg.failure_threshold).mean())
    print(f"[eval_robustness] {target.name} @ {ckpt}", flush=True)
    print(f"[eval_robustness] realistic-world failure rate = {failrate:.4f} over {a.n} "
          f"plausible-band worlds (LOWER = more robust)", flush=True)
    print(f"REALISTIC_FAILRATE={failrate:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
