"""SYNTHETIC benchmark: does DR-IS + a betting CS beat plain Monte Carlo at equal budget?

Everything here is SYNTHETIC and CPU-only. The simulator is replaced by an analytic
failure set with an EXACT known probability, so coverage and width can be checked
against ground truth. No number from this script is a robot or customer result.

Landscape
  theta in [0,1]^d (d = 6 or 12). Operational p = uniform on the plausible sub-box
  [0.1, 0.9]^d. Failure = inside any of 4 disjoint axis-aligned ellipsoids living in the
  first 3 dims (the other dims are inert, as most DR axes are for any one failure).
  Because the ellipsoids are disjoint and inside p's box, P_p(fail) = sum of their
  volumes / 0.8^3 EXACTLY. We set it to 1e-2 or 1e-3.

Pipeline under test (the real code, not a re-implementation)
  1. search   : breakpoint_mjx.multi_search.multi_cmaes_search, 16 restarts x 128, fed a
                legged-contract synthetic evaluate (the real scorer runs on it).
  2. centers  : the best theta of each restart (deduplicated).
  3. surrogate: breakpoint_mjx.surrogate.Ensemble on the search records, with the toy's
                continuous score as label; g = riskcert.failure_prob_link(score).
  4. proposal : riskcert.build_proposal (defensive mixture, alpha = 0.3), fit="moments".
                Also run: fit="fixed" (the first version), centers placed AWAY from the
                regions, and an ORACLE proposal built from the hidden truth (reference
                only; no real run can build it).
  5. estimate : riskcert.estimate_from_samples on N FRESH draws.

Pre-registered primary outcome (written before the first run):
  (b) median CI half-width ratio  CP_plain_MC / DR_betting  at P_fail = 1e-3, N = 2000.
      PASS if >= 5.  FAIL if < 2 (then the proposal/floor is the binding constraint and
      we say so). The analytic floor predicts FAIL; this run measures it.
Disclosure: the default proposal fit was changed from "fixed" to "moments" AFTER the
  first full landscape (d=6, P=1e-2) showed the fixed proposal losing to plain MC even
  under an asymptotic CI. Both are reported, so that choice stays visible.
Secondary: the same ratio at N = 20,000 and 200,000 (where the floor stops binding), the
  cost of the guarantee vs an asymptotic Wald CI, coverage as a sanity check only, and
  sequential-stopping coverage.

  PYTHONPATH=. python scripts/bench_riskcert.py            # full run (~30-40 min CPU)
  PYTHONPATH=. python scripts/bench_riskcert.py --quick    # smoke (~2 min)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breakpoint_mjx import riskcert as rc  # noqa: E402
from breakpoint_mjx.config import ThetaDim, ThetaSpace, load_search_config  # noqa: E402
from breakpoint_mjx.paths import config_path  # noqa: E402

HORIZON = 8
P_LO, P_HI = 0.1, 0.9
CENTERS = np.array([[0.30, 0.30, 0.30], [0.70, 0.65, 0.30],
                    [0.30, 0.70, 0.70], [0.68, 0.30, 0.72]])
ANISO = np.array([[1.0, 0.6, 1.5], [1.4, 1.0, 0.7], [0.8, 1.3, 1.0], [1.0, 1.0, 1.0]])
SHARE = np.array([0.4, 0.3, 0.2, 0.1])


def make_space(d: int) -> ThetaSpace:
    return ThetaSpace(dims={f"x{i}": ThetaDim(f"x{i}", "model", (0.0, 1.0), 0.5, (P_LO, P_HI))
                            for i in range(d)})


def make_regions(p_fail: float):
    """Semi-axes so the 4 ellipsoids carry exactly p_fail of p's mass, SHARE-split."""
    box = (P_HI - P_LO) ** 3
    semi = []
    for a, s in zip(ANISO, SHARE):
        vol = s * p_fail * box                       # target ellipsoid volume
        scale = (vol / (4.0 / 3.0 * math.pi * a.prod())) ** (1.0 / 3.0)
        semi.append(scale * a)
    semi = np.array(semi)
    # exactness: inside p's box and pairwise disjoint (bounding spheres do not touch)
    assert np.all(CENTERS - semi.max(1, keepdims=True) > P_LO)
    assert np.all(CENTERS + semi.max(1, keepdims=True) < P_HI)
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.linalg.norm(CENTERS[i] - CENTERS[j]) > semi[i].max() + semi[j].max()
    truth = float(sum(4.0 / 3.0 * math.pi * s.prod() for s in semi) / box)
    return semi, truth


def min_mahal(x, semi):
    u = np.asarray(x)[:, :3]
    return np.min(np.sqrt((((u[:, None, :] - CENTERS[None]) / semi[None]) ** 2).sum(-1)), axis=1)


def toy_score(x, semi):
    """Continuous 'failure score' of the synthetic simulator: 2/(1+m^2); >=1 iff failed."""
    return 2.0 / (1.0 + min_mahal(x, semi) ** 2)


def fails(x, semi):
    return min_mahal(x, semi) <= 1.0


def legged_evaluate(semi):
    """Legged trajectory contract so the REAL scorer drives the real CMA-ES."""
    import jax.numpy as jnp

    def evaluate(thetas, key):
        m = min_mahal(np.asarray(thetas), semi)
        drive = 1.0 / (1.0 + (m / 2.0) ** 2)
        n = len(m)
        t = np.arange(HORIZON)[None, :]
        return {
            "trunk_height": jnp.asarray(0.30 - 0.25 * drive[:, None] * (t / HORIZON)),
            "up_proj": jnp.asarray(1.0 - 0.9 * drive[:, None] * (t / HORIZON)),
            "lin_vel": jnp.zeros((n, HORIZON, 3)),
            "ang_vel": jnp.asarray(np.repeat(drive[:, None, None], HORIZON, 1)
                                   * np.ones((1, 1, 3)) * 9.0),
            "command": jnp.zeros((n, HORIZON, 3)),
        }
    return evaluate


def search_and_fit(space, semi, restarts, gens, seed):
    import jax
    from breakpoint_mjx.multi_search import multi_cmaes_search
    from breakpoint_mjx.surrogate import Ensemble
    cfg = load_search_config(config_path("search.yaml"))
    cfg = replace(cfg, popsize=128, horizon=HORIZON, generations=gens, seed=seed)
    recs = multi_cmaes_search(legged_evaluate(semi), space, cfg, jax.random.PRNGKey(seed),
                              restarts=restarts, novelty=False)  # as recorded
    th = np.array([r.theta_vector for r in recs])
    # centers: best record per restart, deduplicated
    by_restart = {}
    for r in recs:
        k = (r.id or "").split("-")[1] if (r.id or "").count("-") >= 2 else r.id
        if k not in by_restart or r.failure_score > by_restart[k].failure_score:
            by_restart[k] = r
    cands = sorted(by_restart.values(), key=lambda r: -r.failure_score)
    centers = []
    for r in cands:
        v = np.array(r.theta_vector)
        if all(np.linalg.norm(v - c) > 0.1 for c in centers):
            centers.append(v)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(th), size=min(4000, len(th)), replace=False)
    ens = Ensemble.train(th[idx], toy_score(th[idx], semi), space, k=3, epochs=200)
    return np.array(centers), ens, len(recs)


def bad_centers(k, d, rng):
    out = []
    while len(out) < k:
        c = rng.uniform(P_LO, P_HI, size=d)
        if np.min(np.linalg.norm(CENTERS - c[:3], axis=1)) > 0.3:
            out.append(c)
    return np.array(out)


NAMES = ["mc_cp", "mc_bet", "is_bet", "dr_bet", "dr_bet_wrong_g", "dr_bet_bad_centers",
         "dr_bet_fixed_sd", "dr_bet_oracle_q", "is_wald", "dr_wald", "dr_wald_oracle_q"]


def oracle_proposal(p, semi, d, alpha):
    """Reference ONLY (uses the hidden truth): components on the true ellipsoids, flat in
    the inert dims. Shows what a near-perfect proposal buys, i.e. how much of the gap is
    the proposal and how much is the betting floor. No real run can build this."""
    comps = [rc.DiagTruncNormal(np.r_[c, np.full(d - 3, 0.5)], np.r_[s, np.full(d - 3, 10.0)],
                                p.lo, p.hi) for c, s in zip(CENTERS, semi)]
    return rc.DefensiveMixture(p, comps, SHARE, alpha)


def run_cell(p, qs, g_good, g_wrong, semi, truth, n, reps, alpha, delta, seed,
             c_good, c_wrong):
    """One (landscape, N) cell: every method on fresh draws, `reps` times."""
    names = NAMES
    hw = {k: [] for k in names}
    cov = {k: [] for k in names}
    wmax = []
    for r in range(reps):
        rng = np.random.default_rng(seed + 7919 * r)
        xp = p.sample(n, rng)
        fp = fails(xp, semi)
        k = int(fp.sum())
        ivs = {"mc_cp": rc.clopper_pearson(k, n, delta),
               "mc_bet": rc.betting_cs(fp.astype(float), delta)}
        xq, _ = qs["good"].sample(n, rng)
        fq, wq = fails(xq, semi), qs["good"].weight(xq)
        wmax.append(float(wq.max()))
        gq, gw = g_good(xq), g_wrong(xq)
        for key, meth, g, (ch, cr) in [
                ("is_bet", "is", None, (0.0, 0.0)), ("is_wald", "is_wald", None, (0.0, 0.0)),
                ("dr_bet", "dr", gq, c_good), ("dr_wald", "dr_wald", gq, c_good),
                ("dr_bet_wrong_g", "dr", gw, c_wrong)]:
            e = rc.estimate_from_samples(fq, wq, g, ch, cr, alpha=alpha, delta=delta,
                                         method=meth)
            ivs[key] = (e["lo"], e["hi"])
        for qk in ("bad", "fixed", "oracle"):
            xb, _ = qs[qk].sample(n, rng)
            fb, wb, gb = fails(xb, semi), qs[qk].weight(xb), g_good(xb)
            wmax.append(float(wb.max()))
            meths = [("dr", {"bad": "dr_bet_bad_centers", "fixed": "dr_bet_fixed_sd",
                             "oracle": "dr_bet_oracle_q"}[qk])]
            if qk == "oracle":
                meths.append(("dr_wald", "dr_wald_oracle_q"))
            for meth, key in meths:
                e = rc.estimate_from_samples(fb, wb, gb, *c_good, alpha=alpha, delta=delta,
                                             method=meth)
                ivs[key] = (e["lo"], e["hi"])
        for key, (lo, hi) in ivs.items():
            hw[key].append(0.5 * (hi - lo))
            cov[key].append(lo <= truth <= hi)
    out = {}
    for key in names:
        out[key] = {"median_halfwidth": float(np.median(hw[key])),
                    "coverage": float(np.mean(cov[key]))}
    for key in names:
        out[key]["cp_over_method_width"] = (out["mc_cp"]["median_halfwidth"]
                                            / max(out[key]["median_halfwidth"], 1e-300))
    out["max_w_observed"] = float(max(wmax))
    out["w_bound"] = 1.0 / alpha
    return out


def sequential(p, q, g, c, semi, truth, alpha, delta, batch, n_max, tol, reps, seed):
    """(d) anytime validity: stop at the first batch with half-width <= tol."""
    covered, used = [], []
    for r in range(reps):
        rng = np.random.default_rng(seed + 104729 * r)
        res = rc.operational_risk(lambda x: fails(x, semi), p, q, g_fn=g, n_max=n_max,
                                  batch=batch, target_halfwidth=tol, delta=delta,
                                  n_quad=0, rng=rng, c_override=c)
        covered.append(res["lo"] <= truth <= res["hi"])
        used.append(res["n_rollouts"])
    return {"coverage": float(np.mean(covered)), "median_n_used": float(np.median(used)),
            "reps": reps, "tol": tol, "batch": batch, "n_max": n_max}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "docs" / "riskcert_results.json"))
    args = ap.parse_args()
    alpha, delta, seed = 0.3, 0.05, 0
    if args.quick:
        dims, pfs = [6], [1e-3]
        budgets = [(2000, 20)]
        restarts, gens, seq_reps = 4, 4, 5
    else:
        dims, pfs = [6, 12], [1e-2, 1e-3]
        budgets = [(2000, 200), (20000, 50), (200000, 6)]
        restarts, gens, seq_reps = 16, 15, 60

    t0 = time.time()
    floors = {}
    for a in (0.1, 0.3, 0.5):
        floors[str(a)] = {str(n): rc.dr_floor_halfwidth(n, a, delta)
                          for n in (2000, 20000, 200000)}
    cps = {str(pf): {str(n): 0.5 * np.subtract(*rc.clopper_pearson(round(n * pf), n, delta)[::-1])
                     for n in (2000, 20000, 200000)} for pf in (1e-2, 1e-3)}
    cross = {str(pf): {str(a): rc.crossover_n(pf, a, delta) for a in (0.1, 0.3, 0.5)}
             for pf in (1e-2, 1e-3)}
    print("ANALYTIC FLOOR (perfect surrogate) half-width:", json.dumps(floors))
    print("plain-MC CP half-width at k=round(N p):", json.dumps(cps))
    print("crossover N (DR floor < CP), optimistic:", json.dumps(cross))

    results = {"label": "SYNTHETIC (analytic failure set, CPU); not a robot or customer result",
               "alpha": alpha, "delta": delta, "floor_halfwidth": floors,
               "cp_halfwidth_expected": cps, "crossover_n": cross, "cells": []}
    for d in dims:
        space = make_space(d)
        p = rc.OperationalDist.from_space(space)
        for pf in pfs:
            semi, truth = make_regions(pf)
            centers, ens, n_search = search_and_fit(space, semi, restarts, gens, seed)
            thr = 1.0

            def g_good(x, ens=ens):
                return rc.failure_prob_link(ens.predict(x)[0], thr)

            def g_wrong(x, ens=ens):   # deliberately wrong: the mirror image of the space
                return rc.failure_prob_link(ens.predict(1.0 - np.asarray(x))[0], thr)

            rng = np.random.default_rng(seed + 1)
            qs = {"good": rc.build_proposal(p, centers, g_fn=g_good, alpha=alpha, rng=rng),
                  "fixed": rc.build_proposal(p, centers, g_fn=g_good, alpha=alpha,
                                             fit="fixed", rng=rng),
                  "bad": rc.build_proposal(p, bad_centers(len(centers), d, rng),
                                           g_fn=g_good, alpha=alpha, rng=rng),
                  "oracle": oracle_proposal(p, semi, d, alpha)}
            # variance factor V = E_p[w f] / mu; the IS variance is mu*V - mu^2, so plain
            # MC corresponds to V = 1 and anything above 1 is WORSE than plain MC.
            xv = p.sample(1_000_000, rng)
            fv = fails(xv, semi)
            vf = {k: float(q.weight(xv[fv]).sum() / max(fv.sum(), 1)) for k, q in qs.items()}
            nq = 20_000 if args.quick else 2_000_000
            c_good = rc.surrogate_mean_bound(g_good(p.sample(nq, rng)), delta / 2)
            c_wrong = rc.surrogate_mean_bound(g_wrong(p.sample(nq, rng)), delta / 2)
            hit = int(sum(np.min(np.linalg.norm(centers[:, :3] - c, axis=1)) < 0.1
                          for c in CENTERS))
            print(f"\n[d={d} P_fail={pf:g} truth={truth:.6g}] search evals={n_search} "
                  f"centers={len(centers)} true regions hit by a center={hit}/4 "
                  f"E_p[g]={c_good[0]:.5f}+-{c_good[1]:.5f}")
            print("  variance factor E_p[w f]/mu (plain MC = 1; lower is better):",
                  {k: round(v, 3) for k, v in vf.items()})
            for n, reps in budgets:
                t1 = time.time()
                cell = run_cell(p, qs, g_good, g_wrong, semi, truth, n, reps,
                                alpha, delta, seed, c_good, c_wrong)
                cell.update({"d": d, "p_fail": pf, "truth": truth, "n": n, "reps": reps,
                             "variance_factor_E_p_wf_over_mu": vf,
                             "n_centers": len(centers), "regions_hit": hit,
                             "surrogate_term": list(c_good)})
                results["cells"].append(cell)
                print(f"  N={n:>6} reps={reps:>3} ({time.time() - t1:.0f}s)  "
                      f"max w={cell['max_w_observed']:.3f} (bound {cell['w_bound']:.3f})")
                for k in NAMES:
                    v = cell[k]
                    print(f"    {k:<20} half-width {v['median_halfwidth']:.6f}  "
                          f"CP/method {v['cp_over_method_width']:.3f}  cov {v['coverage']:.3f}")
            if d == dims[0] and pf == pfs[0]:
                sq = sequential(p, qs["good"], g_good, c_good, semi, truth, alpha, delta,
                                batch=500, n_max=20000 if not args.quick else 2000,
                                tol=0.5 * pf if not args.quick else pf, reps=seq_reps,
                                seed=seed)
                sq.update({"d": d, "p_fail": pf, "truth": truth})
                results["sequential"] = sq
                print("  sequential stopping:", sq)
    results["seconds"] = round(time.time() - t0, 1)
    if not args.quick:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print("wrote", args.out)
    print(f"done in {results['seconds']}s")


if __name__ == "__main__":
    main()
