"""CPU real-path check for Certified Minimal Cause: the REAL evaluator, scoring, clustering,
CMC stage (make_evaluator_fail_fn + run_stage) and report writer, at a toy budget.

    PYTHONPATH=. python scripts/mincause_realpath.py --horizon 100 --n 64 --families 2 --k 4

POLICY: with no --policy spec it uses a ZERO-ACTION policy (the robot holds its default pose
under PD control). That makes the physics and the plumbing real but the policy is NOT a
trained locomotion policy, so the causes it finds say nothing about a real Go1 policy. A
trained checkpoint (params:/onnx: via --policy) or the GPU run in docs/mincause.md is what
the science needs. Every number printed here is CPU / toy-budget and UNVERIFIED on GPU.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="go1")
    ap.add_argument("--policy", default=None, help="'params:<npz>' or 'onnx:<path>'")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--n", type=int, default=64, help="random-search worlds")
    ap.add_argument("--families", type=int, default=2)
    ap.add_argument("--k", type=int, default=4, help="CMC seeds per configuration")
    ap.add_argument("--out", default="out/mincause_realpath")
    args = ap.parse_args(argv)

    import jax

    from breakpoint_mjx.clustering import build_families
    from breakpoint_mjx.config import (load_search_config, load_theta_space,
                                       merge_target_thresholds)
    from breakpoint_mjx.evaluator import make_evaluator
    from breakpoint_mjx.mincause import make_evaluator_fail_fn, run_stage
    from breakpoint_mjx.paths import config_path
    from breakpoint_mjx.report import write_report
    from breakpoint_mjx.schemas import BreakPointReport
    from breakpoint_mjx.search import random_search
    from breakpoint_mjx.targets import get_target

    t0 = time.time()
    target = get_target(args.target)
    cfg = merge_target_thresholds(load_search_config(config_path("search.yaml")), target)
    cfg = dataclasses.replace(cfg, horizon=args.horizon, n_eval=args.n, popsize=args.n,
                              generations=1, n_families=args.families)
    space = load_theta_space(config_path(target.theta_space_path))
    env = target.load_env()
    consts = target.resolve_constants(env)
    if args.policy:
        from breakpoint_mjx.adapters import build_policy
        policy_fn = build_policy(args.policy).policy_fn
        pol_desc = args.policy
    else:
        from breakpoint_mjx.adapters import from_params
        obs_dim = int(env.observation_size["state"][0])
        act = env.action_size
        policy_fn = from_params([(np.zeros((obs_dim, 8)), np.zeros(8)),
                                 (np.zeros((8, act)), np.zeros(act))]).policy_fn
        pol_desc = "zero-action (hold default pose), NOT a trained policy"
    evaluate = make_evaluator(env, policy_fn, space, cfg.horizon, target, consts)
    print(f"[realpath] {target.name} policy={pol_desc} H={cfg.horizon} n={cfg.n_eval}",
          flush=True)

    recs = random_search(evaluate, space, cfg, jax.random.PRNGKey(cfg.seed),
                         objective=target.objective)
    failures = [r for r in recs if r.failure_score >= cfg.failure_threshold]
    print(f"[realpath] search: {len(failures)}/{len(recs)} failures "
          f"({time.time() - t0:.0f}s)", flush=True)
    if not failures:
        print("[realpath] no failures to explain; nothing to do")
        return 1
    families, _ = build_families(failures, space, cfg.n_families)
    before = [(f.id, f.name) for f in families]
    families, ran = run_stage(
        families, {r.id: r for r in failures}, space,
        lambda: make_evaluator_fail_fn(evaluate, space, cfg.thresholds, cfg.horizon,
                                       cfg.failure_threshold,
                                       jax.random.PRNGKey(cfg.seed + 5_000_011),
                                       objective=target.objective, max_batch=256),
        environ={**os.environ, "BP_MINCAUSE_K": str(args.k)})
    nominal = dict(zip(space.names(), space.nominal_vector()))
    ok, sizes = True, []
    for f in families:
        print(f"  {f.id}: {f.name!r}  | {f.margin}")
        if f.minimal_cause is not None:
            sizes.append(len(f.minimal_cause))
            for c in f.minimal_cause:
                shown = f.triggering_parameters[c["dim"]]
                side = "high" if shown > nominal[c["dim"]] else "low"
                vside = "high" if c["value"] > nominal[c["dim"]] else "low"
                ok &= (side == c["direction"] == vside)
    os.makedirs(args.out, exist_ok=True)
    rep = BreakPointReport(run_id="mincause-realpath", policy_descriptor=target.env_name,
                           comparison={}, families=families, mp4_paths={})
    write_report(rep, os.path.join(args.out, "report.html"), embed=False)
    summary = {
        "label": "CPU real-path, toy budget, UNVERIFIED on GPU",
        "policy": pol_desc, "horizon": cfg.horizon, "n": cfg.n_eval, "k": args.k,
        "cmc_ran": ran, "names_before": before,
        "families": [{"id": f.id, "name": f.name, "margin": f.margin,
                      "minimal_cause": f.minimal_cause, "min_distance": f.min_distance,
                      "cause_certified": f.cause_certified,
                      "cause_necessary": f.cause_necessary,
                      "merged_from": f.merged_from, "members": len(f.member_ids)}
                     for f in families],
        "directions_agree": bool(ok),
        "median_cause_size": float(np.median(sizes)) if sizes else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out, "mincause_realpath.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("cmc_ran", "directions_agree",
                                              "median_cause_size", "elapsed_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
