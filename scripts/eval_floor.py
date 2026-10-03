"""Failure FLOOR + policy-portfolio analysis over a shared set of realistic worlds.

Evaluates SEVERAL policies on the SAME plausible-band world sample and reports:
  * per-policy realistic-world failure rate (same number eval_robustness prints),
  * FLOOR_UNION  -- the fraction of worlds where EVERY policy fails: an empirical
    estimate of the unwinnable core of the band (physically unrecoverable worlds --
    e.g. a max-magnitude shove on slick ground). No training method can beat this
    floor, so it is the honest asymptote for "how low can the failure rate go";
  * PORTFOLIO    -- the failure rate of the best-per-world portfolio (min over
    policies per world): what an oracle policy-selector would achieve. The gap
    between the best single policy and PORTFOLIO is the headroom a better single
    policy (or an ensemble/distillation, cf. RMA-style teacher-student distillation,
    Kumar et al. 2021) could still capture.

Usage (GPU / Colab):
  python scripts/eval_floor.py --target go1 [--n 2048] [--seed 0] \
      --run frozen=/path/frozen_root --run dr_full=/path/retrain_dr_full [...]
Each --run root must hold <root>/<target>/<target>_policy.pkl. Prints machine-grep
lines FLOOR_UNION= and PORTFOLIO=.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def floor_stats(fail_matrix):
    """Pure: bool matrix [n_policies, n_worlds] -> per-policy rates + the union floor.

    The oracle best-per-world portfolio fails a world iff EVERY policy fails it, so the
    portfolio failure rate IS the union floor -- one number, two readings: (a) the
    empirical unwinnable core, (b) what a perfect policy-selector would score."""
    F = np.asarray(fail_matrix, dtype=bool)
    if F.ndim != 2 or F.shape[0] == 0 or F.shape[1] == 0:
        raise ValueError(f"fail_matrix must be [n_policies, n_worlds], got {F.shape}")
    floor = float(F.all(axis=0).mean())
    return {"per_policy": [float(r.mean()) for r in F], "floor_union": floor,
            "portfolio": floor}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Union failure floor + portfolio over policies.")
    ap.add_argument("--target", default="go1")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", action="append", required=True, help="label=policy_root_dir")
    a = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp

    from breakpoint_mjx.config import (load_search_config, load_theta_space,
                                       merge_target_thresholds)
    from breakpoint_mjx.evaluator import make_evaluator
    from breakpoint_mjx.policy import make_policy_fn
    from breakpoint_mjx.search import scores_from_trajectories
    from breakpoint_mjx.targets import get_target

    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_robustness import plausible_band_sample

    target = get_target(a.target)
    cfg = merge_target_thresholds(load_search_config(config_path("search.yaml")), target)
    space = load_theta_space(config_path(target.theta_space_path))
    thetas = plausible_band_sample(space, a.n, a.seed)

    labels, rows = [], []
    for spec in a.run:
        label, root = spec.split("=", 1)
        out = os.path.join(root, target.name)
        ckpt = os.path.join(out, target.checkpoint_name)
        if not os.path.exists(ckpt):
            print(f"[skip] {label}: no policy at {ckpt}", flush=True)
            continue
        env = target.load_env()
        consts = target.resolve_constants(env)
        policy_fn = make_policy_fn(out_dir=out, target=target)
        evaluate = make_evaluator(env, policy_fn, space, cfg.horizon, target, consts)
        traj = evaluate(jnp.asarray(thetas), jax.random.PRNGKey(int(a.seed)))
        scores, _, _ = scores_from_trajectories(traj, cfg.thresholds, cfg.horizon,
                                                target.objective)
        fails = np.asarray(scores) >= cfg.failure_threshold
        labels.append(label)
        rows.append(fails)
        print(f"[floor] {label:12s} failrate={float(fails.mean()):.4f}", flush=True)
    if not rows:
        raise SystemExit("no runnable --run policies")

    st = floor_stats(np.stack(rows))
    print(f"[floor] worlds={a.n} policies={len(labels)}", flush=True)
    print(f"[floor] UNION floor (all {len(labels)} policies fail) -- no single policy can "
          f"beat this without solving currently-unsolved worlds", flush=True)
    print(f"FLOOR_UNION={st['floor_union']:.6f}", flush=True)
    print(f"PORTFOLIO={st['portfolio']:.6f}", flush=True)
    best = min(st["per_policy"])
    print(f"[floor] best single policy {best:.4f} vs oracle portfolio "
          f"{st['portfolio']:.4f} -> headroom {best - st['portfolio']:.4f} for a better "
          f"single policy / distilled ensemble", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
