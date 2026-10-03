"""Calibrate the campaign-coverage certificate on SYNTHETIC planted landscapes.

EVERY NUMBER THIS SCRIPT PRINTS IS SYNTHETIC (planted Gaussian / slab landscapes on CPU,
real multi_cmaes_search + real scorer, no physics). None of it is a robot result.

Landscapes (ground truth known):
  homog-K   scripts/bench_search_scaling.make_landscape: K equal Gaussian modes, drive
            sigma 0.55. Uniform failure mass is ~0, so the audit has almost nothing to
            find here; this family tests the stopping rule and the richness estimate.
  hetero-K  one HUGE FLAT mode plus K-1 small tall ones. The huge mode is a slab
            u[d] > 0.9 on one normalized axis d (10% of the box by construction), at a
            drive level just above the failure threshold, so it has no gradient and a
            lower score than the tall peaks. All tall centres get u[d] = 0.15, so the slab
            is disjoint from them. This is where basin counting is expected to be blind
            (the incumbent of a restart goes to the higher peak) and where the uniform
            audit has real mass to find.

True modes = connected components of the true failing set (balls of the calibrated
failure radius, plus the slab); overlapping balls are one component (K_conn <= K).

Protocol per run: 4 waves of 16 restarts x popsize 128 x 15 generations (the bench's
cost point). Every stop rule is replayed on the same wave prefixes (paired design); the
fixed-16 and fixed-64 budgets are prefixes 1 and 4. A 2048-world Sobol audit is scored
once per run and re-classified at every prefix. Truth for unseen mass = 200,000 uniform
draws classified analytically.

PRE-REGISTERED BARS (frozen in this docstring before the first run):
  B1 audit calibration (lead): audit 95% CP upper bound >= true unseen mass in >= 90% of
     runs, at the chao2 rule's stopping prefix.
  B2 audit power: among runs whose true unseen mass >= 0.01 at that prefix, the audit
     reports k_unseen > 0 in >= 90%.
  B3 stop safety: chao2 rule false-complete (declares complete while true components
     found < K_conn) in <= 10% of all runs.
  B4 stop power: chao2 rule declares complete within 4 waves in >= 50% of homog-3 runs
     (so a rule that never stops cannot pass B3 vacuously).
  B5 efficiency: on homog-3, mean restarts spent by the chao2 rule < 64 while mean true
     components found >= fixed-64 mean - 0.1.
  B6 estimator (from the original proposal): at the final prefix, mean |chao2 - K_conn|
     <= 0.6 x mean |S_obs - K_conn|; Chao2 95% CI contains K_conn in >= 70% of runs.
     Expected to FAIL on hetero; reported either way.
  B7 baseline decision: if Boender-Rinnooy Kan has false-complete <= chao2's AND power
     >= chao2's on every family, BRK is the better rule and the doc must say so.
  The certificate is fit to ship only if B1, B3 and B4 pass.

  ADDED AFTER ONE SMOKE RUN (hetero-4 seed 0 found the slab in wave 0, so B2 might have
  no qualifying runs), disclosed as such: B2' counterfactual audit power. On hetero runs,
  drop the slab's incumbents and re-run the audit at the chao2 stopping prefix; bar: CP
  upper >= true unseen mass AND k_unseen > 0 in >= 90% of hetero runs.

  EXPLORATORY RESULT (seeds 0-9, docs/data/coverage_bench_seeds0-9.json): the chao2 rule
  FAILED B3 (28% false-complete), so the chao2 certificate does not ship. The BRK rule
  met B1/B3/B4 on the same runs, but choosing it after seeing them is post hoc.
  CONFIRMATION (frozen before running, FRESH seeds 10-19, same families and protocol):
  score B1-B5 on --rule brk. BRK becomes the product default only if B1, B3 and B4 pass
  on the fresh seeds; otherwise BP_COVERAGE ships the audit only, with no stop claim.
    python scripts/bench_coverage.py --family homog --K 3 6 --seed-start 10 --out ...
    python scripts/bench_coverage.py --rule brk --summarize <the fresh-seed files>

  python scripts/bench_coverage.py --family homog --out out/cov_homog.json
  python scripts/bench_coverage.py --family hetero --out out/cov_hetero.json
  python scripts/bench_coverage.py --summarize out/cov_homog.json out/cov_hetero.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bench_search_scaling import (  # noqa: E402
    DRIVE_SIGMA, HORIZON, make_landscape, normalized, trajectories_from_drive,
)
from breakpoint_mjx import coverage as cov  # noqa: E402
from breakpoint_mjx.config import load_search_config, load_theta_space  # noqa: E402
from breakpoint_mjx.paths import config_path  # noqa: E402

SLAB_T = 0.90
TALL_ON_SLAB_AXIS = 0.15
WAVE, WAVES, POP, GENS = 16, 4, 128, 15
N_AUDIT, N_TRUTH = 2048, 200_000
FAMILIES = {"homog": [3, 6, 10], "hetero": [4, 7]}


def drive_threshold(cfg):
    from breakpoint_mjx.search import scores_from_trajectories
    d = np.linspace(0.0, 1.0, 4001)
    s, _, _ = scores_from_trajectories(trajectories_from_drive(d), cfg.thresholds, HORIZON)
    fail = np.asarray(s) >= cfg.failure_threshold
    first = int(np.argmax(fail))
    assert fail.any() and fail[first:].all(), "score is not monotone in drive"
    return float(d[first])


class Landscape:
    def __init__(self, space, cfg, family, K, seed):
        self.space, self.family, self.K = space, family, K
        D = len(space.names())
        self.thr = drive_threshold(cfg)
        self.R = DRIVE_SIGMA * np.sqrt(2 * np.log(1.0 / self.thr))
        if family == "homog":
            self.centres = make_landscape(space, K, seed)
            self.slab_dim = None
        else:
            self.centres = make_landscape(space, K - 1, seed)
            self.slab_dim = int(seed % D)
            self.centres[:, self.slab_dim] = TALL_ON_SLAB_AXIS
            self.a_slab = min(1.0, self.thr + 0.05)
        self._components()

    def drive(self, u):
        d = np.linalg.norm(u[:, None, :] - self.centres[None], axis=2).min(axis=1)
        dr = np.exp(-(d ** 2) / (2 * DRIVE_SIGMA ** 2))
        if self.slab_dim is not None:
            dr = np.maximum(dr, np.where(u[:, self.slab_dim] > SLAB_T, self.a_slab, 0.0))
        return dr

    def evaluate(self, thetas, key):
        return trajectories_from_drive(self.drive(normalized(self.space, thetas)))

    def _components(self):
        n = len(self.centres) + (self.slab_dim is not None)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        C = self.centres
        for i in range(len(C)):
            for j in range(i):
                if np.linalg.norm(C[i] - C[j]) < 2 * self.R:
                    parent[find(i)] = find(j)
            if self.slab_dim is not None and SLAB_T - C[i, self.slab_dim] < self.R:
                parent[find(i)] = find(n - 1)
        roots = sorted({find(i) for i in range(n)})
        self.node_comp = np.array([roots.index(find(i)) for i in range(n)])
        self.K_conn = len(roots)

    def comp_of(self, u):
        """True component of each normalized point, -1 if it does not fail (analytic)."""
        u = np.atleast_2d(u)
        d = np.linalg.norm(u[:, None, :] - self.centres[None], axis=2)
        inside = d < self.R
        lab = np.full(u.shape[0], -1)
        hit = inside.any(axis=1)
        lab[hit] = self.node_comp[np.argmax(inside[hit], axis=1)]
        if self.slab_dim is not None:
            s = (u[:, self.slab_dim] > SLAB_T) & ~hit
            lab[s] = self.node_comp[-1]
        return lab


def run_one(space, cfg, family, K, seed):
    import jax

    from breakpoint_mjx.multi_search import multi_cmaes_search
    from breakpoint_mjx.search import scores_from_trajectories

    L = Landscape(space, cfg, family, K, seed)
    rcfg = replace(cfg, seed=int(seed))
    score_fn = cov.make_score_fn(L.evaluate, rcfg, jax.random.PRNGKey(7), batch=2048)
    graph = cov.ModeGraph(score_fn, cfg.failure_threshold)

    rng = np.random.default_rng(10_000 + seed)
    truth_lab = L.comp_of(rng.random((N_TRUTH, len(space.names()))))
    lo, hi = np.array(space.lower()), np.array(space.upper())
    audit_th = cov.uniform_audit_thetas(space, N_AUDIT, seed=seed)
    audit_sc = score_fn(audit_th)
    audit_lab = L.comp_of(normalized(space, audit_th))
    # scorer vs analytic truth on the audit sample (they must agree)
    agree = float(((audit_sc >= cfg.failure_threshold) == (audit_lab >= 0)).mean())

    records, prefixes = [], []
    t0 = time.perf_counter()
    for w in range(WAVES):
        wcfg = replace(rcfg, seed=int(seed) + 100003 * w)
        with contextlib.redirect_stdout(io.StringIO()):
            recs = multi_cmaes_search(L.evaluate, space, wcfg, jax.random.PRNGKey(0),
                                      restarts=WAVE)
        for i, r in enumerate(recs):
            r.id = f"cmaes-w{w}-r{cov.restart_unit(r.id)[1]}-{len(records) + i}"
        records += recs
        units, inc = cov.restart_incumbents(records, cfg.failure_threshold)
        for k in units:
            if k in inc:
                graph.add(k, inc[k].theta_vector)
        est = cov.estimate_from_graph(graph, units, inc)
        keys = [u for u in units if u in inc]
        labels = graph.labels(keys)
        inc_u = normalized(space, np.array([inc[k].theta_vector for k in keys])) \
            if keys else np.zeros((0, len(lo)))
        inc_comp = L.comp_of(inc_u) if keys else np.zeros(0, dtype=int)
        found = sorted({int(c) for c in inc_comp if c >= 0})
        true_unseen = float(((truth_lab >= 0) & ~np.isin(truth_lab, found)).mean())
        a = cov.residual_audit(audit_th, audit_sc, graph, labels, cfg.failure_threshold,
                               space=space)
        truth_unseen_audit = set(np.where((audit_lab >= 0)
                                          & ~np.isin(audit_lab, found))[0].tolist())
        said = set(a.pop("unseen_index"))
        drop = None
        if L.slab_dim is not None:
            # COUNTERFACTUAL (added after the first smoke run found the slab in wave 0):
            # pretend the slab was never found by dropping its incumbents, then ask
            # whether the audit sees the ~10% mass it now leaves unexplained.
            slab = int(L.node_comp[-1])
            keep = [k for k, c in zip(keys, inc_comp) if c != slab]
            found_d = [c for c in found if c != slab]
            a_d = cov.residual_audit(audit_th, audit_sc, graph, graph.labels(keep),
                                     cfg.failure_threshold, space=space)
            a_d.pop("unseen_index")
            drop = {"audit": a_d, "true_unseen_mass": float(
                ((truth_lab >= 0) & ~np.isin(truth_lab, found_d)).mean())}
        c2 = est["chao2"]
        prefixes.append({
            "wave": w, "restarts_spent": WAVE * (w + 1), "units": est["units"],
            "failing_units": est["failing_units"], "S_obs": c2["S_obs"],
            "true_found": len(found), "unmatched_incumbents": int((inc_comp < 0).sum()),
            "chao2": c2["estimate"], "ci": c2["ci"], "Q1": c2["Q1"], "Q2": c2["Q2"],
            "new_mode_prob": c2["new_mode_prob"], "chao1": est["chao1"]["estimate"],
            "brk_w_est": est["brk"]["w_est"], "stop": est["stop"],
            "audit": a, "true_unseen_mass": true_unseen,
            "audit_vs_truth": {"both": len(said & truth_unseen_audit),
                               "audit_only": len(said - truth_unseen_audit),
                               "truth_only": len(truth_unseen_audit - said)},
            "slab_dropped": drop,
        })
    return {"family": family, "K": K, "seed": seed, "K_conn": L.K_conn,
            "fail_radius": float(L.R), "drive_threshold": L.thr,
            "slab_dim": L.slab_dim,
            "true_uniform_fail_mass": float((truth_lab >= 0).mean()),
            "audit_scorer_agreement": agree, "probe_evals": graph.probe_evals,
            "seconds": round(time.perf_counter() - t0, 1), "prefixes": prefixes}


# --------------------------------------------------------------------------- summary

RULES = ("chao2", "brk", "chao1")


def stop_prefix(run, rule):
    for p in run["prefixes"]:
        if p["stop"][rule]:
            return p, True
    return run["prefixes"][-1], False


def summarize(runs, rule="chao2"):
    def pct(x):
        return f"{100 * np.mean(x):5.1f}%" if len(x) else "  n/a"

    groups = sorted({(r["family"], r["K"]) for r in runs})
    print("ALL NUMBERS BELOW ARE SYNTHETIC (planted landscapes, CPU).")
    print(f"{len(runs)} runs; waves of {WAVE} restarts x pop {POP} x {GENS} gens; "
          f"audit n={N_AUDIT}; truth n={N_TRUTH}\n")
    hdr = (f"{'group':<10}{'Kconn':>6}{'Sobs':>6}{'found':>6}{'chao2':>7}{'fix16':>6}"
           f"{'fix64':>6}{'unifmass':>9}{'agree':>7}")
    print(hdr)
    for g in groups:
        rs = [r for r in runs if (r["family"], r["K"]) == g]
        last = [r["prefixes"][-1] for r in rs]
        print(f"{g[0] + '-' + str(g[1]):<10}{np.mean([r['K_conn'] for r in rs]):6.1f}"
              f"{np.mean([p['S_obs'] for p in last]):6.1f}"
              f"{np.mean([p['true_found'] for p in last]):6.1f}"
              f"{np.mean([p['chao2'] for p in last]):7.1f}"
              f"{np.mean([r['prefixes'][0]['true_found'] for r in rs]):6.1f}"
              f"{np.mean([p['true_found'] for p in last]):6.1f}"
              f"{np.mean([r['true_uniform_fail_mass'] for r in rs]):9.4f}"
              f"{np.mean([r['audit_scorer_agreement'] for r in rs]):7.3f}")

    print("\nSTOP RULES (declared = fired within 4 waves; false = declared while "
          "true found < K_conn)")
    print(f"{'group':<10}{'rule':<7}{'declared':>9}{'false/all':>10}{'false|decl':>11}"
          f"{'restarts':>9}{'found':>7}")
    table = {}
    for g in groups + [("ALL", 0)]:
        rs = runs if g[0] == "ALL" else [r for r in runs if (r["family"], r["K"]) == g]
        for rr in RULES:
            st = [stop_prefix(r, rr) for r in rs]
            dec = [d for _, d in st]
            false = [d and p["true_found"] < r["K_conn"] for (p, d), r in zip(st, rs)]
            fcond = [f for f, d in zip(false, dec) if d]
            table[(g, rr)] = (np.mean(dec), np.mean(false))
            print(f"{(g[0] + '-' + str(g[1])) if g[1] else 'ALL':<10}{rr:<7}"
                  f"{pct(dec):>9}{pct(false):>10}{pct(fcond):>11}"
                  f"{np.mean([p['restarts_spent'] for p, _ in st]):9.1f}"
                  f"{np.mean([p['true_found'] for p, _ in st]):7.2f}")

    print(f"\nAUDIT at the {rule} rule's stopping prefix")
    cover, power_n, power_hit, rows = [], 0, 0, []
    for r in runs:
        p, _ = stop_prefix(r, rule)
        a = p["audit"]
        cover.append(a["cp_upper"] >= p["true_unseen_mass"])
        if p["true_unseen_mass"] >= 0.01:
            power_n += 1
            power_hit += a["k_unseen"] > 0
        rows.append((r["family"], r["K"], r["seed"], a["k_fail"], a["k_unseen"],
                     a["cp_upper"], p["true_unseen_mass"], p["audit_vs_truth"]))
    for g in groups:
        sel = [x for x in rows if (x[0], x[1]) == g]
        print(f"  {g[0]}-{g[1]}: mean k_fail {np.mean([x[3] for x in sel]):.1f}/"
              f"{N_AUDIT}, mean k_unseen {np.mean([x[4] for x in sel]):.1f}, mean CP upper "
              f"{np.mean([x[5] for x in sel]):.4f}, mean true unseen mass "
              f"{np.mean([x[6] for x in sel]):.4f}, runs with true unseen >= 1%: "
              f"{sum(x[6] >= 0.01 for x in sel)}/{len(sel)}")
    avt = {k: sum(x[7][k] for x in rows) for k in ("both", "audit_only", "truth_only")}
    print(f"  audit-vs-truth point classification (all runs): {avt}")
    drops = []
    for r in runs:
        p, _ = stop_prefix(r, rule)
        if p.get("slab_dropped"):
            d = p["slab_dropped"]
            drops.append((d["audit"]["k_unseen"], d["audit"]["cp_upper"],
                          d["true_unseen_mass"]))
    if drops:
        print(f"  COUNTERFACTUAL slab dropped ({len(drops)} hetero runs): mean k_unseen "
              f"{np.mean([x[0] for x in drops]):.1f}/{N_AUDIT} = "
              f"{np.mean([x[0] for x in drops]) / N_AUDIT:.4f}, mean CP upper "
              f"{np.mean([x[1] for x in drops]):.4f}, mean true unseen mass "
              f"{np.mean([x[2] for x in drops]):.4f}")

    fin = [r["prefixes"][-1] for r in runs]
    e_c2 = [abs(p["chao2"] - r["K_conn"]) for p, r in zip(fin, runs)]
    e_s = [abs(p["S_obs"] - r["K_conn"]) for p, r in zip(fin, runs)]
    ci_ok = [p["ci"][0] - 1e-9 <= r["K_conn"] <= p["ci"][1] + 1e-9
             for p, r in zip(fin, runs)]
    ratio = [p["S_obs"] / r["K_conn"] for p, r in zip(fin, runs)]
    split = [p["S_obs"] - p["true_found"] for p in fin]
    print("\nESTIMATOR at the final prefix (64 restarts)")
    for g in groups:
        idx = [i for i, r in enumerate(runs) if (r["family"], r["K"]) == g]
        print(f"  {g[0]}-{g[1]}: mean|chao2-K| {np.mean([e_c2[i] for i in idx]):.2f}  "
              f"mean|Sobs-K| {np.mean([e_s[i] for i in idx]):.2f}  CI covers K "
              f"{pct([ci_ok[i] for i in idx])}  Sobs/Kconn "
              f"{np.mean([ratio[i] for i in idx]):.2f}  oversplit "
              f"{np.mean([split[i] for i in idx]):.2f}")

    homog3 = [r for r in runs if (r["family"], r["K"]) == ("homog", 3)]
    b = {}
    b["B1"] = np.mean(cover) >= 0.90
    b["B2"] = (power_hit / power_n >= 0.90) if power_n else None
    b["B3"] = table[(("ALL", 0), rule)][1] <= 0.10
    b["B4"] = (table[(("homog", 3), rule)][0] >= 0.50) if homog3 else None
    if homog3:
        st = [stop_prefix(r, rule)[0] for r in homog3]
        b["B5"] = (np.mean([p["restarts_spent"] for p in st]) < 64 and
                   np.mean([p["true_found"] for p in st]) >=
                   np.mean([r["prefixes"][-1]["true_found"] for r in homog3]) - 0.1)
    b["B6"] = (np.mean(e_c2) <= 0.6 * np.mean(e_s)) and np.mean(ci_ok) >= 0.70
    brk_dom = all(table[(g, "brk")][1] <= table[(g, "chao2")][1]
                  and table[(g, "brk")][0] >= table[(g, "chao2")][0] for g in groups)
    print("\nPRE-REGISTERED BARS (SYNTHETIC)")
    print(f"  B1 audit CP upper >= truth: {pct(cover)} of runs -> "
          f"{'PASS' if b['B1'] else 'FAIL'}")
    print(f"  B2 audit detects >=1% unseen mass: {power_hit}/{power_n} -> "
          f"{'n/a (no such run)' if b['B2'] is None else 'PASS' if b['B2'] else 'FAIL'}")
    print(f"  B3 {rule} false-complete (all runs): "
          f"{100 * table[(('ALL', 0), rule)][1]:.1f}% -> {'PASS' if b['B3'] else 'FAIL'}")
    if homog3:
        print(f"  B4 {rule} declares on homog-3: "
              f"{100 * table[(('homog', 3), rule)][0]:.1f}% -> "
              f"{'PASS' if b['B4'] else 'FAIL'}")
        print(f"  B5 efficiency on homog-3 -> {'PASS' if b['B5'] else 'FAIL'}")
    print(f"  B6 estimator: mean|chao2-K| {np.mean(e_c2):.2f} vs mean|Sobs-K| "
          f"{np.mean(e_s):.2f}, CI coverage {pct(ci_ok)} -> {'PASS' if b['B6'] else 'FAIL'}")
    if drops:
        ok = [x[0] > 0 and x[1] >= x[2] for x in drops]
        b["B2p"] = np.mean(ok) >= 0.90
        print(f"  B2' counterfactual audit power (slab dropped): {pct(ok)} -> "
              f"{'PASS' if b['B2p'] else 'FAIL'}")
    print(f"  B7 BRK dominates chao2 on every group: {brk_dom}")
    ship = bool(b["B1"] and b["B3"] and b["B4"])
    print(f"  SHIP (B1 and B3 and B4): {ship}")
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["homog", "hetero", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--K", type=int, nargs="*", help="restrict to these K values")
    ap.add_argument("--out", default="out/bench_coverage.json")
    ap.add_argument("--summarize", nargs="*")
    ap.add_argument("--rule", default="chao2", choices=RULES,
                    help="stop rule the B1-B5 bars are scored on")
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args()
    if args.summarize:
        runs = []
        for f in args.summarize:
            runs += json.loads(Path(f).read_text())
        summarize(runs, args.rule)
        return
    space = load_theta_space(config_path("go1/theta_space.yaml"))
    cfg = replace(load_search_config(config_path("search.yaml")), popsize=POP,
                  horizon=HORIZON, generations=GENS)
    fams = list(FAMILIES) if args.family == "all" else [args.family]
    runs = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for fam in fams:
        for K in [k for k in FAMILIES[fam] if not args.K or k in args.K]:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                r = run_one(space, cfg, fam, K, seed)
                runs.append(r)
                last = r["prefixes"][-1]
                print(f"SYNTHETIC {fam}-{K} seed {seed}: Kconn={r['K_conn']} "
                      f"found={[p['true_found'] for p in r['prefixes']]} "
                      f"Sobs={[p['S_obs'] for p in r['prefixes']]} "
                      f"chao2={last['chao2']:.1f} stops="
                      f"{[[k for k, v in p['stop'].items() if v] for p in r['prefixes']]} "
                      f"audit k_unseen={last['audit']['k_unseen']} "
                      f"true_unseen={last['true_unseen_mass']:.4f} ({r['seconds']}s)",
                      flush=True)
                Path(args.out).write_text(json.dumps(runs))
    summarize(runs, args.rule)


if __name__ == "__main__":
    main()
