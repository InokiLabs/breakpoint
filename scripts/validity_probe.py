"""Physical-validity probe: calibrate + sanity-check the sim-invalid gate on CPU.

Runs the REAL evaluator (make_evaluator -> rollout_one -> legged_summarize) for the Go1
target with a frozen checkpoint, and for each rollout logs the three validity channels
(floor_height, penetration, qacc_max) plus the gate verdict and failure score:

  * nominal     : N rollouts at the nominal theta (calibration: must flag 0)
  * dropout sweep: obs_dropout_prob one-at-a-time, all else nominal
  * slope sweep  : floor_slope_rad one-at-a-time, all else nominal

Every batch has the same size N so the evaluator compiles once.

    PYTHONPATH=. python scripts/validity_probe.py \
        --ckpt ../breakpoint-mjx/media/dl/frozen_go1_policy.pkl --horizon 150 --n 32 \
        --out docs/validity_probe.json

CPU / short-horizon numbers only (horizon 150 = 3 s at 50 Hz vs the 500-step default).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np


def _summ(traj, i):
    fh = np.asarray(traj["floor_height"][i], dtype=float)
    pen = np.asarray(traj["penetration"][i], dtype=float)
    qa = np.asarray(traj["qacc_max"][i], dtype=float)
    z = np.asarray(traj["trunk_height"][i], dtype=float)
    return {
        "min_floor_height": float(np.nanmin(fh)), "final_floor_height": float(fh[-1]),
        "min_world_z": float(np.nanmin(z)), "final_world_z": float(z[-1]),
        "min_penetration": float(np.nanmin(pen)), "max_qacc": float(np.nanmax(qa)),
        "nonfinite": bool(not (np.isfinite(fh).all() and np.isfinite(z).all())),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default="docs/validity_probe.json")
    ap.add_argument("--dropout", default="0,0.05,0.1,0.15,0.2")
    ap.add_argument("--slope", default="-0.1,-0.05,0,0.05,0.1")
    args = ap.parse_args(argv)

    import jax

    from breakpoint_mjx import failure
    from breakpoint_mjx.config import load_search_config, load_theta_space
    from breakpoint_mjx.evaluator import make_evaluator
    from breakpoint_mjx.paths import config_path
    from breakpoint_mjx.policy import make_policy_fn
    from breakpoint_mjx.targets import get_target

    target = get_target("go1")
    cfg = load_search_config(config_path("search.yaml"))
    thr = dict(cfg.thresholds)
    space = load_theta_space(config_path(target.theta_space_path))
    env = target.load_env()
    with open(args.ckpt, "rb") as fh:
        params = pickle.load(fh)
    policy_fn = make_policy_fn(params=params, target=target)
    consts = target.resolve_constants(env)
    evaluate = make_evaluator(env, policy_fn, space, args.horizon, target, consts)
    names = space.names()
    nominal = np.array([space.dims[n].nominal for n in names], dtype=np.float32)

    def run(label, overrides):
        thetas = np.tile(nominal, (args.n, 1))
        for k, v in overrides.items():
            thetas[:, names.index(k)] = v
        t0 = time.time()
        traj = evaluate(jax.numpy.asarray(thetas), jax.random.PRNGKey(0))
        traj = {k: np.asarray(v) for k, v in traj.items()}
        rows = []
        for i in range(args.n):
            one = {k: v[i] for k, v in traj.items()}
            sig = failure.component_signals(one, thr)
            legacy = {k: v for k, v in one.items() if k not in failure.VALIDITY_KEYS}
            lsig = failure.component_signals(legacy, thr)
            wsig = failure.component_signals(one, {**thr, "validity_whole_rollout": 1.0})
            rows.append({
                **_summ(traj, i),
                "sim_invalid": failure.is_sim_invalid(sig),
                "reasons": failure.sim_invalid_reasons(sig),
                "fall": bool(sig["fall"]),
                "score": failure.failure_score(sig, args.horizon),
                "legacy_fall": bool(lsig["fall"]),
                "legacy_score": failure.failure_score(lsig, args.horizon),
                "whole_rollout_sim_invalid": failure.is_sim_invalid(wsig),
                "whole_rollout_reasons": failure.sim_invalid_reasons(wsig),
            })
        agg = {
            "label": label, "overrides": overrides, "n": args.n,
            "seconds": round(time.time() - t0, 1),
            "n_sim_invalid": sum(r["sim_invalid"] for r in rows),
            "n_fall": sum(r["fall"] for r in rows),
            "n_legacy_fall": sum(r["legacy_fall"] for r in rows),
            "n_whole_rollout_sim_invalid": sum(r["whole_rollout_sim_invalid"] for r in rows),
            "reasons": {n: sum(n in r["reasons"] for r in rows)
                        for _, n in failure.SIM_INVALID_REASONS},
            "whole_rollout_reasons": {n: sum(n in r["whole_rollout_reasons"] for r in rows)
                                      for _, n in failure.SIM_INVALID_REASONS},
            "n_below_floor_end": sum(r["final_floor_height"] < 0 for r in rows),
            "min_penetration": min(r["min_penetration"] for r in rows),
            "max_qacc": max(r["max_qacc"] for r in rows),
            "min_floor_height": min(r["min_floor_height"] for r in rows),
            "min_world_z": min(r["min_world_z"] for r in rows),
            "rows": rows,
        }
        print(f"{label:>28}: invalid={agg['n_sim_invalid']}/{args.n} fall={agg['n_fall']} "
              f"legacy_fall={agg['n_legacy_fall']} whole_rollout_invalid={agg['n_whole_rollout_sim_invalid']} below_floor_end={agg['n_below_floor_end']} "
              f"min_pen={agg['min_penetration']:.4f} max_qacc={agg['max_qacc']:.1f} "
              f"min_fh={agg['min_floor_height']:.3f} min_z={agg['min_world_z']:.3f} "
              f"({agg['seconds']}s)", flush=True)
        return agg

    results = {"ckpt": os.path.basename(args.ckpt), "horizon": args.horizon, "n": args.n,
               "thresholds": {"penetration_max": thr.get("penetration_max",
                                                        failure.DEFAULT_PENETRATION_MAX),
                              "qacc_max": thr.get("qacc_max", failure.DEFAULT_QACC_MAX),
                              "trunk_height_min": thr["trunk_height_min"]},
               "label": "CPU, short horizon; not a GPU/500-step result", "runs": []}
    results["runs"].append(run("nominal", {}))
    for v in [float(x) for x in args.dropout.split(",") if x]:
        if v != 0.0:
            results["runs"].append(run(f"obs_dropout_prob={v}", {"obs_dropout_prob": v}))
    for v in [float(x) for x in args.slope.split(",") if x]:
        if v != 0.0:
            results["runs"].append(run(f"floor_slope_rad={v}", {"floor_slope_rad": v}))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
