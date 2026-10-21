"""Inoki Labs / BreakPoint benchmark runner.

    python -m benchmark.run_benchmark --config benchmark/configs/smoke.yaml
    python -m benchmark.run_benchmark --config benchmark/configs/pilot.yaml --stage search
    python -m benchmark.run_benchmark --config benchmark/configs/pilot.yaml --stage analyze

Stages (default ``all``, in order): ``search`` (Experiment A, with Experiment B's
fresh-seed validation of each run's selected candidates), ``replay`` (cross-engine
replay of validated failures), ``regression`` (Experiment C), ``analyze`` (CSVs, figures,
reports). Every unit of work writes its own file under the run directory and is skipped
when that file exists, so an interrupted run resumes where it stopped. Existing files are
never overwritten (pass a new ``--out`` for a fresh run).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from benchmark.analysis import metrics as M  # noqa: E402
from benchmark.evaluation.setup import (REPO, load_config, regime_space,  # noqa: E402
                                        run_id, sha256_file)

NOMINAL_EXCLUDE_RATE = 0.25    # BENCHMARK_PLAN.md: a configuration that falls in more than
                               # 25% of nominal rollouts does not walk and is excluded.


def log(msg):
    print(f"[bench {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_new(path: Path, obj) -> None:
    """Write JSON atomically; refuse to overwrite an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} exists; benchmark artifacts are never overwritten")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=_json_default))
    tmp.rename(path)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


# --------------------------------------------------------------------------- context
class Context:
    """Engine objects for one policy configuration (one compile per batch shape)."""

    def __init__(self, pc, bc):
        import dataclasses

        from breakpoint_mjx.accel import tune_for_gpu
        tune_for_gpu()
        from breakpoint_mjx.check_policy import _load_env, load_policy
        from breakpoint_mjx.config import (load_search_config, load_theta_space,
                                           merge_target_thresholds)
        from breakpoint_mjx.evaluator import make_evaluator
        from breakpoint_mjx.paths import config_path
        from breakpoint_mjx.targets import get_target

        from benchmark.evaluation.recorder import Recorder

        self.pc = pc
        self.target = get_target(pc.robot)
        self.theta_path = Path(config_path(self.target.theta_space_path))
        self.full_space = load_theta_space(self.theta_path)
        cfg = merge_target_thresholds(load_search_config(config_path("search.yaml"), env={}),
                                      self.target)
        self.cfg = dataclasses.replace(cfg, horizon=int(bc.horizon))
        ckpt = pc.checkpoint_path()
        if not ckpt.is_file():
            raise FileNotFoundError(f"{pc.name}: checkpoint {ckpt} not found")
        obs = pc.obs_spec_path()
        self.env = _load_env(self.target, str(obs) if obs else None, log=log)
        opts = {"obs_spec": str(obs)} if obs else {}
        self.policy_fn, _n, _m, self.policy_label = load_policy(self.target, self.env,
                                                                 str(ckpt), opts)
        self.consts = self.target.resolve_constants(self.env)
        self.base_model = self.env._mjx_model
        self.evaluate = make_evaluator(self.env, self.policy_fn, self.full_space,
                                       self.cfg.horizon, self.target, self.consts)
        self.thr = self.cfg.thresholds
        mk = lambda: Recorder(self.evaluate, self.target, self.thr, self.env,  # noqa: E731
                              self.base_model, self.cfg.horizon)
        self.new_recorder = mk
        self.batch = bc.plan.batch

    def nominal_check(self, n_batches=2):
        import jax

        rec = self.new_recorder()
        X = np.tile(np.asarray(self.full_space.nominal_vector(), dtype=np.float32),
                    (self.batch, 1))
        for b in range(n_batches):
            rec(X, jax.random.PRNGKey(424_242 + b))
        t = rec.table()
        n = len(t["failure"])
        k = int(t["failure"].sum())
        return {"n": n, "valid_falls": k, "rate": k / n, "invalid": int(t["invalid"].sum()),
                "walks": (k / n) <= NOMINAL_EXCLUDE_RATE,
                "rule": f"excluded if valid-fall rate > {NOMINAL_EXCLUDE_RATE}"}


# ------------------------------------------------------------------- Experiment A + B
def analyze_run(tab, rspace, bc, rid):
    lo, hi = np.asarray(rspace.lower()), np.asarray(rspace.upper())
    U = M.box_normalize(tab["theta"], lo, hi)
    fail = tab["failure"] > 0.5
    fidx = np.flatnonzero(fail)
    out = {"n_failures": int(fail.sum()), "n_invalid": int((tab["invalid"] > 0.5).sum()),
           "clusters": {}}
    for eps in [bc.cluster_eps, *bc.cluster_eps_sensitivity]:
        labels, leaders = M.leader_clusters(U[fidx], eps) if fidx.size else (np.zeros(0), [])
        out["clusters"][f"{eps:g}"] = {
            "n": len(leaders),
            "leader_rows": [int(fidx[i]) for i in leaders],
            "curve": M.discovery_curve(fail, U, eps).tolist(),
        }
    from breakpoint_mjx.equal_budget import dominant_modes, nominal_distances

    modes = dominant_modes(tab["theta"][fidx], rspace) if fidx.size else []
    out["modes"] = sorted(set(modes))
    out["n_modes"] = len(out["modes"])
    if fidx.size:
        l2, linf = nominal_distances(tab["theta"][fidx], rspace)
        out["closest_l2"] = float(l2.min())
        out["closest_linf"] = float(linf.min())
        out["median_ttf_steps"] = float(np.median(tab["ttf"][fidx]))
    else:
        out["closest_l2"] = out["closest_linf"] = out["median_ttf_steps"] = None
    return out


def search_stage(bc, outdir: Path):
    from benchmark.evaluation.search import run_method
    from benchmark.evaluation.validation import validate

    status = {}
    for pc in bc.policies:
        ctx = None
        for regime in bc.regimes:
            for method in bc.methods:
                for rep in bc.reps():
                    rid = run_id(pc.name, regime, method, rep)
                    path = outdir / "runs" / f"{rid}.json"
                    if path.exists():
                        continue
                    if ctx is None:
                        log(f"loading {pc.name} ({pc.robot}, {pc.checkpoint})")
                        ctx = Context(pc, bc)
                        nom = ctx.nominal_check()
                        status[pc.name] = nom
                        log(f"{pc.name}: nominal valid falls {nom['valid_falls']}/{nom['n']}"
                            f" -> {'walks' if nom['walks'] else 'EXCLUDED (does not walk)'}")
                        write_status(outdir, pc.name, nom)
                        if not nom["walks"]:
                            break
                    if not status[pc.name]["walks"]:
                        break
                    rspace = regime_space(ctx.full_space, regime)
                    log(f"run {rid}: budget {bc.plan.budget}")
                    res = run_method(method, ctx.new_recorder(), rspace, ctx.cfg, bc.plan,
                                     rep, ctx.target.objective,
                                     variant=bc.method_variants.get(method))
                    tab = res["table"]
                    an = analyze_run(tab, rspace, bc, rid)
                    # Experiment B: fixed candidate rule, fresh seeds
                    fidx = np.flatnonzero(tab["failure"] > 0.5)
                    leaders = an["clusters"][f"{bc.cluster_eps:g}"]["leader_rows"]
                    seed = zlib.crc32(rid.encode())
                    first, sampled, exact = M.select_candidates(
                        fidx, leaders, bc.validation_first_k, bc.validation_leaders, seed)
                    rows = sorted(set(first) | set(sampled))
                    t0 = time.time()
                    vres = validate(ctx.new_recorder(), tab["theta"][rows], bc.validation_n,
                                    ctx.batch, key_index=seed % 1_000_000)
                    vtime = time.time() - t0
                    vmap = {int(r): v for r, v in zip(rows, vres)}
                    k_lead = sum(1 for r in sampled if vmap[r]["validated"])
                    ttfv = M.time_to_first_validated(first, {r: vmap[r]["validated"]
                                                             for r in first})
                    record = {
                        "run_id": rid, "policy": pc.name, "robot": pc.robot,
                        "regime": regime, "method": method, "rep": rep,
                        "horizon": ctx.cfg.horizon, "plan": res["plan"],
                        "timing": res["timing"],
                        "sim_steps": int(res["evals"]) * int(ctx.cfg.horizon),
                        "box": {"lo": rspace.lower(), "hi": rspace.upper(),
                                "names": rspace.names()},
                        "rows": {
                            "theta": np.round(tab["theta"], 6),
                            "failure": tab["failure"].astype(int),
                            "fall": tab["fall"].astype(int),
                            "invalid": tab["invalid"].astype(int),
                            "invalid_reason": tab["invalid_reason"],
                            "ttf": tab["ttf"], "min_height": np.round(tab["min_height"], 4),
                        },
                        "analysis": an,
                        "validation": {
                            "n_per_candidate": bc.validation_n,
                            "first_rows": first, "leader_rows_sampled": sampled,
                            "all_leaders_checked": exact,
                            "results": {str(r): v for r, v in vmap.items()},
                            "leaders_validated": k_lead,
                            "leaders_checked": len(sampled),
                            "estimated_validated_distinct": M.estimated_validated_distinct(
                                an["clusters"][f"{bc.cluster_eps:g}"]["n"], k_lead,
                                len(sampled), exact),
                            "time_to_first_validated_rollouts": ttfv,
                            "wall_s": round(vtime, 2),
                            "rollouts": len(rows) * bc.validation_n,
                        },
                    }
                    write_new(path, record)
                    log(f"  {rid}: {an['n_failures']} failures, "
                        f"{an['clusters'][f'{bc.cluster_eps:g}']['n']} clusters, "
                        f"{an['n_modes']} modes; validated {k_lead}/{len(sampled)} leaders; "
                        f"ttfv={ttfv}; search {res['timing']['method_wall_s']:.0f}s "
                        f"validate {vtime:.0f}s")
        if ctx is not None:
            del ctx


def write_status(outdir, name, nom):
    p = outdir / "configs_status" / f"{name}.json"
    if not p.exists():
        write_new(p, nom)


# ----------------------------------------------------------------------------- replay
def replay_stage(bc, outdir: Path):
    from benchmark.evaluation.replay import make_replayer

    for pc in bc.policies:
        runs = sorted((outdir / "runs").glob(f"{pc.name}__*.json"))
        todo = [r for r in runs if not (outdir / "replay" / r.name).exists()]
        if not todo or bc.replay_per_run <= 0:
            continue
        ctx = Context(pc, bc)
        fn, why = make_replayer(ctx.env, ctx.policy_fn, ctx.target, ctx.full_space,
                                ctx.consts, ctx.cfg, ctx.base_model, bc.replay_seeds)
        for rp in todo:
            rec = json.loads(rp.read_text())
            out = {"run_id": rec["run_id"], "seeds": bc.replay_seeds}
            if fn is None:
                out.update({"skipped": why})
            else:
                val = rec["validation"]["results"]
                vrows = sorted(int(r) for r, v in val.items() if v["validated"])
                rng = np.random.default_rng([zlib.crc32(rec["run_id"].encode()), 7])
                pick = (sorted(rng.choice(vrows, size=min(bc.replay_per_run, len(vrows)),
                                          replace=False).tolist()) if vrows else [])
                X = np.asarray(rec["rows"]["theta"], dtype=float)[pick] if pick else np.zeros((0, 1))
                t0 = time.time()
                res = fn(X) if pick else {"n": 0, "confirmed": 0, "rows": []}
                out.update({"rows_replayed": pick, "result": res,
                            "wall_s": round(time.time() - t0, 1)})
                log(f"replay {rec['run_id']}: {res.get('confirmed')}/{res.get('n')} "
                    f"reproduce in C MuJoCo [{out['wall_s']} s]")
            write_new(outdir / "replay" / rp.name, out)


# ------------------------------------------------------------------------- regression
def regression_stage(bc, outdir: Path):
    from benchmark.evaluation.regression import (adversarial_battery, decide_adversarial,
                                                 make_synthetic_candidate, run_gate)

    rc = bc.regression or {}
    pairs = rc.get("pairs", [])
    if not pairs:
        return
    base_pc = next(p for p in bc.policies if p.name == rc["policy"])
    rdir = outdir / "regression"
    for pair in pairs:
        name = pair["name"]
        path = rdir / f"{name}.json"
        if path.exists():
            continue
        kind = pair["kind"]           # real | identical | synthetic
        base = base_pc.checkpoint_path()
        if kind == "identical":
            cand = base
        elif kind == "synthetic":
            cand = rdir / "checkpoints" / f"{name}.pkl"
            if not cand.exists():
                make_synthetic_candidate(base, cand, float(pair["sigma"]), int(pair["seed"]))
        elif kind == "real":
            cand = Path(pair["candidate"])
            cand = cand if cand.is_absolute() else REPO / cand
            if not cand.exists():
                write_new(path, {"name": name, "kind": kind, "status": "incomplete",
                                 "reason": f"candidate checkpoint {cand} not found"})
                continue
        else:
            raise ValueError(kind)
        gate_json = rdir / f"{name}.gate.json"
        g = run_gate(base_pc.robot, base, cand, gate_json, budget=rc.get("budget", "standard"),
                     alpha=float(rc.get("alpha", 0.05)), escalate_to=rc.get("escalate_to"),
                     log=log)
        # gate exit codes: 0 pass, 1 regression, anything else an error (never a verdict)
        status = "ran" if g["exit_code"] in (0, 1) else "gate_error"
        out = {"name": name, "kind": kind, "status": status,
               "baseline": str(base), "candidate": str(cand),
               "baseline_sha256": sha256_file(base), "candidate_sha256": sha256_file(cand),
               "gate": g, "acceptance": "gate statistical decision, alpha 0.05, tolerance 0"}
        if kind == "synthetic":
            out["sigma"] = pair["sigma"]
            out["note"] = "SYNTHETIC weight-noise candidate; not a real policy"
        # adversarial battery: validated failures from Experiment A, reported separately
        adv = rc.get("adversarial", {})
        if adv:
            thetas = adversarial_thetas(outdir, base_pc.name, adv.get("regime", "conservative"),
                                        int(adv.get("max_conditions", 24)))
            if len(thetas):
                from dataclasses import replace
                cb = Context(base_pc, bc)
                cc = Context(replace(base_pc, checkpoint=str(cand)), bc)
                per = adversarial_battery(cb.new_recorder(), cc.new_recorder(), thetas,
                                          int(adv.get("seeds", 8)), cb.batch)
                out["adversarial"] = {"conditions": len(thetas), "per_condition": per,
                                      "decision": decide_adversarial(per)}
        write_new(path, out)
        log(f"regression {name}: gate exit {g['exit_code']}")
    boundary_battery_stage(bc, outdir)


def boundary_thetas(outdir, policy, regimes, p_range, cap, seed=2025):
    """Conditions where the BASELINE fails sometimes, not always: Experiment B candidates
    of ``policy`` whose 20-seed fall rate lies in ``p_range``. Chosen from baseline data
    only, so the battery can show a candidate getting worse (or better)."""
    lo, hi = map(float, p_range)
    rows = []
    for regime in regimes:
        for rp in sorted((outdir / "runs").glob(f"{policy}__{regime}__*.json")):
            rec = json.loads(rp.read_text())
            th = np.asarray(rec["rows"]["theta"], dtype=float)
            for r, v in rec["validation"]["results"].items():
                if lo <= v["p"] <= hi:
                    rows.append(th[int(r)])
    if not rows:
        return np.zeros((0, 0))
    rows = np.asarray(rows)
    if len(rows) > cap:
        rng = np.random.default_rng(seed)
        rows = rows[np.sort(rng.choice(len(rows), cap, replace=False))]
    return rows


def boundary_battery_stage(bc, outdir: Path):
    """Second adversarial battery (config ``regression.boundary``), one file per pair,
    reported separately from both the fixed battery and the validated-failure battery."""
    from dataclasses import replace

    from benchmark.evaluation.regression import adversarial_battery, decide_adversarial

    rc = bc.regression or {}
    bb = rc.get("boundary")
    if not bb:
        return
    base_pc = next(p for p in bc.policies if p.name == rc["policy"])
    rdir = outdir / "regression"
    thetas = boundary_thetas(outdir, base_pc.name, bb.get("regimes", ["conservative"]),
                             bb.get("p_range", [0.1, 0.6]), int(bb.get("max_conditions", 24)))
    if not len(thetas):
        log("boundary battery: no candidate conditions in range; skipped")
        return
    cb = None
    for pair in rc.get("pairs", []):
        name = pair["name"]
        src = rdir / f"{name}.json"
        path = rdir / f"{name}.boundary.json"
        if path.exists() or not src.exists():
            continue
        info = json.loads(src.read_text())
        if info.get("status") == "incomplete":
            continue
        cb = cb or Context(base_pc, bc)
        cc = Context(replace(base_pc, checkpoint=info["candidate"]), bc)
        per = adversarial_battery(cb.new_recorder(), cc.new_recorder(), thetas,
                                  int(bb.get("seeds", 8)), cb.batch)
        write_new(path, {"name": name, "kind": info["kind"], "battery": "boundary",
                         "selection": bb, "conditions": len(thetas),
                         "thetas": thetas, "per_condition": per,
                         "decision": decide_adversarial(per)})
        dec = decide_adversarial(per)
        log(f"boundary battery {name}: regressed={dec.get('regressed')}")


def adversarial_thetas(outdir, policy, regime, cap):
    rows = []
    for rp in sorted((outdir / "runs").glob(f"{policy}__{regime}__*.json")):
        rec = json.loads(rp.read_text())
        th = np.asarray(rec["rows"]["theta"], dtype=float)
        for r, v in rec["validation"]["results"].items():
            if v["validated"]:
                rows.append(th[int(r)])
    if not rows:
        return np.zeros((0, 0))
    rng = np.random.default_rng(2024)
    rows = np.asarray(rows)
    if len(rows) > cap:
        rows = rows[np.sort(rng.choice(len(rows), cap, replace=False))]
    return rows


# ------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="run directory (default results/<name>)")
    ap.add_argument("--stage", default="all",
                    choices=["all", "search", "replay", "regression", "analyze"])
    a = ap.parse_args(argv)
    bc = load_config(a.config)
    outdir = Path(a.out) if a.out else HERE / "results" / bc.name
    outdir.mkdir(parents=True, exist_ok=True)
    prov = outdir / "provenance.json"
    if not prov.exists():
        from benchmark.evaluation.provenance import collect
        from breakpoint_mjx.paths import config_path
        from breakpoint_mjx.targets import get_target

        ph = {p.name: {"path": str(p.checkpoint_path()), "sha256": sha256_file(p.checkpoint_path()),
                       "obs_spec": str(p.obs_spec_path()) if p.obs_spec else None,
                       "obs_spec_sha256": sha256_file(p.obs_spec_path()) if p.obs_spec else None,
                       "independent_policy": p.independent_policy, "note": p.note}
              for p in bc.policies}
        th = {}
        for p in bc.policies:
            tp = Path(config_path(get_target(p.robot).theta_space_path))
            th[p.robot] = {"path": str(tp), "sha256": sha256_file(tp)}
        write_new(prov, collect(bc, ph, th))
        (outdir / "config.yaml").write_text(Path(a.config).read_text()) \
            if not (outdir / "config.yaml").exists() else None
    stages = ["search", "replay", "regression", "analyze"] if a.stage == "all" else [a.stage]
    for s in stages:
        log(f"stage {s}")
        if s == "search":
            search_stage(bc, outdir)
        elif s == "replay":
            replay_stage(bc, outdir)
        elif s == "regression":
            regression_stage(bc, outdir)
        elif s == "analyze":
            from benchmark.analysis.report import analyze
            analyze(bc, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
