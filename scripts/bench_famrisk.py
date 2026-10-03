"""Pre-registered SYNTHETIC benchmark for Cause-Stratified Operational Risk (docs/famrisk.md).

Pure numpy on CPU. No physics, no policy, no robot: planted AND-regions over the real Go1
ThetaSpace (16 dims). Every number this prints is SYNTHETIC.

Oracle. A planted region R is an AND of one-sided thresholds on 1-3 dims whose EXACT mass
under the declared operating domain p (uniform on each dim's plausible band) is m. A world
fails with probability r inside R and never outside it. CMC sees the same failure process
plus p=0.1 label noise (each outcome flipped with probability 0.1), so a region is
"recovered by CMC" through the real ``mincause.certified_minimal_cause`` code path.

Three estimators of a family's operational risk P_p(fail AND theta in R_hat), all at N
rollouts per family:
  (a) plain MC   : N draws from p, failures that land in R_hat, Clopper-Pearson on k/N.
  (b) riskcert   : riskcert.operational_risk (defensive-mixture IS + betting confidence
                   sequence; no surrogate, so the IS variant) on the indicator
                   fail AND theta in R_hat, proposal centred inside R_hat.
  (c) stratified : famrisk.family_risk (exact mass x CP on N fresh conditional rollouts).

Truth for a recovered region R_hat is r * P_p(R_hat intersect R), computed exactly.

    PYTHONPATH=. python scripts/bench_famrisk.py                 # 500 reps, N=2000
    PYTHONPATH=. python scripts/bench_famrisk.py --reps 20       # quick look

Writes docs/famrisk_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breakpoint_mjx import famrisk as fr  # noqa: E402
from breakpoint_mjx.config import load_theta_space  # noqa: E402
from breakpoint_mjx.mincause import certified_minimal_cause  # noqa: E402
from breakpoint_mjx.paths import config_path  # noqa: E402
from breakpoint_mjx.riskcert import OperationalDist, build_proposal, operational_risk  # noqa: E402

MASSES = (1e-2, 1e-3, 1e-4)
RATES = (0.3, 0.7, 1.0)
FLIP = 0.1
DELTA = 0.05
EXCLUDE = frozenset({"latency_steps"})   # rounded dim: not planted (keeps truth exact)
RELAX = True                             # --no-relax: certified-region-only ablation


# --------------------------------------------------------------------------- #
# Planted regions
# --------------------------------------------------------------------------- #
def _eligible(space, p, t):
    """(dim index, direction) pairs whose plausible band has >= t of its width beyond
    nominal on that side, so a tail of fraction t sits strictly away from nominal."""
    out = []
    for j, n in enumerate(space.names()):
        if n in EXCLUDE:
            continue
        nom, lo, hi = space.dims[n].nominal, p.lo[j], p.hi[j]
        w = hi - lo
        if (hi - nom) / w > t + 1e-9:
            out.append((j, "high"))
        if (nom - lo) / w > t + 1e-9:
            out.append((j, "low"))
    return out


def plant(rng, space, p, mass, rate, avoid=(), sizes=(1, 2, 3)):
    """Planted region with EXACT p-mass ``mass``: per-dim tail fraction mass**(1/size)."""
    for _ in range(1000):
        size = int(rng.choice(sizes))
        t = mass ** (1.0 / size)
        el = [e for e in _eligible(space, p, t) if e[0] not in avoid]
        dims = {}
        for i in rng.permutation(len(el)):
            j, d = el[i]
            if j not in dims:
                dims[j] = d
            if len(dims) == size:
                break
        if len(dims) < size:
            continue
        names = space.names()
        cause = []
        for j, d in dims.items():
            w = p.hi[j] - p.lo[j]
            thr = p.hi[j] - t * w if d == "high" else p.lo[j] + t * w
            cause.append({"dim": names[j], "value": float(thr), "direction": d})
        region = fr.region_from_cause(cause, names)
        m, _ = fr.stratum_mass(p, region)
        assert abs(m - mass) / mass < 1e-6, (m, mass)
        return SimpleNamespace(region=region, mass=mass, rate=rate, dims=set(dims))
    raise RuntimeError("could not plant a region")


def oracle(planted, flip=0.0, salt=0):
    """fail_fn(X, seeds): Bernoulli(rate) inside any planted region, 0 outside; optional
    label-flip noise. Deterministic in (salt, seeds[0], batch size); ``salt`` keeps
    different cells / reps from reusing the same uniform draws."""
    planted = planted if isinstance(planted, list) else [planted]

    def fail_fn(X, seeds):
        X = np.atleast_2d(X)
        seeds = np.asarray(seeds if seeds is not None else np.arange(len(X)), np.int64)
        u = np.random.default_rng([int(salt), int(seeds[0]) & 0x7FFFFFFF, len(X), 17]
                                  ).random((len(X), 2))
        out = np.zeros(len(X), bool)
        for pl in planted:
            out |= fr.in_region(X, pl.region) & (u[:, 0] < pl.rate)
        if flip > 0:
            out ^= u[:, 1] < flip
        return out
    return fail_fn


def _intervals(region, D):
    lo, hi = np.full(D, -np.inf), np.full(D, np.inf)
    for c in region:
        if c["op"] == ">=":
            lo[c["idx"]] = max(lo[c["idx"]], c["bound"])
        else:
            hi[c["idx"]] = min(hi[c["idx"]], c["bound"])
    return lo, hi


def intersect_mass(p, r1, r2):
    """Exact P_p(R1 intersect R2) for a uniform p (both are axis-aligned boxes)."""
    D = p.dim
    a1, b1 = _intervals(r1, D)
    a2, b2 = _intervals(r2, D)
    a, b = np.maximum(np.maximum(a1, a2), p.lo), np.minimum(np.minimum(b1, b2), p.hi)
    return float(np.prod(np.clip(b - a, 0, None) / (p.hi - p.lo)))


def representative(rng, pl, space, p, others=()):
    """A failing representative: cause dims beyond the threshold (out to the SEARCH bound),
    other dims uniform on the search bounds, re-drawn if they land in another region."""
    names = space.names()
    lower, upper = np.array(space.lower()), np.array(space.upper())
    for _ in range(1000):
        th = lower + rng.random(len(names)) * (upper - lower)
        for c in pl.region:
            j = c["idx"]
            end = upper[j] if c["op"] == ">=" else lower[j]
            th[j] = c["bound"] + rng.uniform(0.05, 1.0) * (end - c["bound"])
        if not any(fr.in_region(th[None], o.region)[0] for o in others):
            return th
    raise RuntimeError("no representative")


def cmc_region(rng, pl, space, p, fail_fn, seed, others=()):
    names = space.names()
    nominal = np.array(space.nominal_vector(), float)
    lower, upper = np.array(space.lower(), float), np.array(space.upper(), float)
    th = representative(rng, pl, space, p, others)
    res = certified_minimal_cause(th, nominal, lower, upper, fail_fn, k_seeds=8,
                                  max_seed_mult=4, seed0=seed)
    if not res.certified or not res.final_certified:
        return None, res
    cause = [{"dim": names[s], "value": float(res.theta_min[s]), "direction": d}
             for s, d in zip(res.cause, res.directions)]
    return fr.region_from_cause(cause, names, fr_int(space)), res


def region_used(p, space, reg):
    """The region famrisk.run_stage actually estimates on: the certified region, or its
    boundary-relaxed version when the certified one has zero mass (unless --no-relax)."""
    if not RELAX or fr.stratum_mass(p, reg)[0] > 0.0:
        return reg, False
    rel = fr.relax_region(reg, space, fr.relax_settings({})[1])
    return (rel, True) if fr.stratum_mass(p, rel)[0] > 0.0 else (reg, False)


def fr_int(space):
    from breakpoint_mjx.mincause import _integer_dims
    return _integer_dims(space)


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def est_mc(rng, p, region, fail_fn, n, seed):
    X = p.sample(n, rng)
    f = np.asarray(fail_fn(X, seed + np.arange(n)), bool) & fr.in_region(X, region)
    lo, hi = fr.cp_interval(int(f.sum()), n, DELTA)
    return lo, hi


def est_riskcert(rng, p, region, fail_fn, n, seed):
    if fr.stratum_mass(p, region)[0] == 0.0:
        return 0.0, 0.0          # nothing to find; give the baseline the zero for free
    D = p.dim
    a, b = _intervals(region, D)
    a, b = np.maximum(a, p.lo), np.minimum(b, p.hi)
    center = 0.5 * (a + b)
    q = build_proposal(p, center[None, :], alpha=0.3, fit="fixed", rng=rng)
    ctr = [seed]

    def ev(X):
        s = ctr[0] + np.arange(len(X))
        ctr[0] += len(X)
        return np.asarray(fail_fn(X, s), bool) & fr.in_region(X, region)
    o = operational_risk(ev, p, q, n_max=n, batch=500, target_halfwidth=None, delta=DELTA,
                         rng=rng)
    return float(o["lo"]), float(o["hi"])


def est_strat(rng, p, region, fail_fn, n, seed, delta=DELTA):
    o = fr.family_risk(fail_fn, p, region, n, delta_f=delta, rng=rng, seed0=seed)
    return o["risk_lo"], o["risk_hi"]


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def cell(space, p, mass, rate, reps, n, seed0):
    rows = []
    for i in range(reps):
        rng = np.random.default_rng([seed0, int(mass * 1e6), int(rate * 10), i])
        pl = plant(rng, space, p, mass, rate)
        salt = int(rng.integers(1 << 62))
        roll = oracle(pl, salt=salt)
        seed = 10_000_000 * (i + 1)
        row = {"true_mass": mass, "rate": rate}
        # given region (truth from the exact float mass, not the nominal planted mass)
        truth = rate * fr.stratum_mass(p, pl.region)[0]
        for name, fn in (("mc", est_mc), ("riskcert", est_riskcert), ("strat", est_strat)):
            lo, hi = fn(rng, p, pl.region, roll, n, seed + {"mc": 1, "riskcert": 2,
                                                            "strat": 3}[name] * 1_000_000)
            row[f"given_{name}"] = [lo, hi, bool(lo <= truth <= hi)]
        # CMC-recovered region (CMC sees the same failure process + label flips)
        reg, res = cmc_region(rng, pl, space, p, oracle(pl, FLIP, salt + 1), seed + 7_000_000)
        row["cmc_sims"] = int(res.sims)
        row["cmc_certified"] = reg is not None
        if reg is not None:
            reg, row["cmc_relaxed"] = region_used(p, space, reg)
            truth_r = rate * intersect_mass(p, reg, pl.region)
            row["cmc_truth"] = truth_r
            row["cmc_exact_dims"] = sorted(c["idx"] for c in reg) == sorted(pl.dims)
            for name, fn in (("mc", est_mc), ("riskcert", est_riskcert), ("strat", est_strat)):
                lo, hi = fn(rng, p, reg, roll, n, seed + {"mc": 4, "riskcert": 5,
                                                          "strat": 6}[name] * 1_000_000)
                row[f"cmc_{name}"] = [lo, hi, bool(lo <= truth_r <= hi)]
        rows.append(row)
    return rows


def summarize_cell(rows):
    out = {"reps": len(rows)}
    for src in ("given", "cmc"):
        rs = [r for r in rows if f"{src}_strat" in r]
        out[f"{src}_n"] = len(rs)
        for name in ("mc", "riskcert", "strat"):
            if not rs:
                continue
            iv = np.array([r[f"{src}_{name}"][:2] for r in rs], float)
            cov = np.mean([r[f"{src}_{name}"][2] for r in rs])
            out[f"{src}_{name}"] = {
                "coverage": float(cov),
                "median_width": float(np.median(iv[:, 1] - iv[:, 0])),
                "median_upper": float(np.median(iv[:, 1])),
            }
    rel = [r["cmc_relaxed"] for r in rows if "cmc_relaxed" in r]
    out["cmc_relaxed_rate"] = float(np.mean(rel)) if rel else None
    if any("cmc_certified" in r for r in rows):
        out["cmc_certified_rate"] = float(np.mean([r["cmc_certified"] for r in rows]))
        ex = [r["cmc_exact_dims"] for r in rows if r.get("cmc_certified")]
        out["cmc_exact_dims_rate"] = float(np.mean(ex)) if ex else None
        out["cmc_sims_median"] = float(np.median([r["cmc_sims"] for r in rows]))
    return out


def odd_experiment(space, reps, n, seed0):
    """Six planted regions per rep on disjoint dims (each mass twice, rates cycled); the
    declared ODD clips one cause dim of one region per mass so that region has zero mass
    under it. Families are built from the CMC-recovered regions and run through the REAL
    famrisk.run_stage with the ODD passed as a BP_RISK_DIST YAML."""
    import yaml

    p0 = OperationalDist.from_space(space)
    names = space.names()
    stats = {"excluded": 0, "excluded_certified": 0, "excluded_labelled_out": 0,
             "included": 0, "included_certified": 0, "included_labelled_out": 0,
             "class_counts": {c: 0 for c in fr.CLASSES}}
    tmpdir = tempfile.mkdtemp(prefix="famrisk_odd_")
    for i in range(reps):
        rng = np.random.default_rng([seed0, 99, i])
        planted, used = [], set()
        for k, mass in enumerate(MASSES * 2):
            pl = plant(rng, space, p0, mass, RATES[k % 3], avoid=used, sizes=(1, 2))
            used |= pl.dims
            planted.append(pl)
        excluded = [planted[0], planted[1], planted[2]]         # one per mass
        block = {}
        for pl in excluded:
            c = pl.region[0]
            nom = space.dims[c["dim"]].nominal
            lo, hi = p0.lo[c["idx"]], p0.hi[c["idx"]]
            mid = nom + 0.5 * (c["bound"] - nom)
            block[c["dim"]] = {"range": [float(lo), float(mid)] if c["op"] == ">="
                               else [float(mid), float(hi)]}
        path = os.path.join(tmpdir, f"odd_{i}.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump({"operational": block}, fh)
        salt = int(rng.integers(1 << 62))
        fail_fn = oracle(planted, salt=salt)
        fams = []
        for j, pl in enumerate(planted):
            reg, _ = cmc_region(rng, pl, space, p0, oracle(planted, FLIP, salt + 1),
                                seed=50_000_000 * (i + 1) + 1_000_003 * j,
                                others=[o for o in planted if o is not pl])
            f = SimpleNamespace(id=f"fam-{j}", cause_certified=reg is not None,
                                minimal_cause=None if reg is None else [
                                    {"dim": c["dim"], "value": float(c["value"]),
                                     "direction": "high" if c["op"] == ">=" else "low"}
                                    for c in reg], odd_risk=None)
            fams.append(f)
        fr.run_stage(fams, space, fail_fn, log=None, seed=i,
                     environ={"BP_RISK_DIST": path, "BP_FAMRISK_N": str(n),
                              "BP_FAMRISK_RELAX": "1" if RELAX else "0"})
        for j, (pl, f) in enumerate(zip(planted, fams)):
            key = "excluded" if j < 3 else "included"
            stats[key] += 1
            if f.odd_risk is not None:
                stats[f"{key}_certified"] += 1
                stats["class_counts"][f.odd_risk["class"]] += 1
                if f.odd_risk["class"] == "outside_odd":
                    stats[f"{key}_labelled_out"] += 1
    ec, ic = stats["excluded_certified"], stats["included_certified"]
    stats["accuracy_out_among_certified"] = stats["excluded_labelled_out"] / ec if ec else None
    stats["accuracy_out_end_to_end"] = stats["excluded_labelled_out"] / stats["excluded"]
    stats["in_odd_mislabelled_out"] = stats["included_labelled_out"]
    stats["included_certified_rate"] = ic / stats["included"]
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--out", default=str(ROOT / "docs" / "famrisk_results.json"))
    ap.add_argument("--skip-odd", action="store_true")
    ap.add_argument("--no-relax", action="store_true",
                    help="ablation: estimate and classify on the certified region only")
    a = ap.parse_args(argv)
    global RELAX
    RELAX = not a.no_relax
    space = load_theta_space(config_path("go1/theta_space.yaml"))
    p = OperationalDist.from_space(space)
    t0 = time.time()
    cells = {}
    print("SYNTHETIC benchmark: planted AND-regions on the Go1 ThetaSpace, p = uniform on "
          f"plausible bands, {a.reps} reps, N={a.n} rollouts per family per method",
          flush=True)
    for mass in MASSES:
        for rate in RATES:
            rows = cell(space, p, mass, rate, a.reps, a.n, a.seed)
            s = summarize_cell(rows)
            cells[f"m={mass:g},r={rate:g}"] = s
            g, c = s["given_strat"], s.get("cmc_strat") or {}
            print(f"  m={mass:g} r={rate:g}: given cov strat {g['coverage']:.3f} "
                  f"mc {s['given_mc']['coverage']:.3f} riskcert "
                  f"{s['given_riskcert']['coverage']:.3f}; width strat "
                  f"{g['median_width']:.3g} mc {s['given_mc']['median_width']:.3g} riskcert "
                  f"{s['given_riskcert']['median_width']:.3g} | CMC certified "
                  f"{s['cmc_certified_rate']:.2f}, cov strat {c.get('coverage', float('nan')):.3f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    # Bars
    bar1 = {}
    for mass in MASSES:
        ks = [k for k in cells if k.startswith(f"m={mass:g},")]
        given = min(cells[k]["given_strat"]["coverage"] for k in ks)
        cm = [cells[k]["cmc_strat"]["coverage"] for k in ks if "cmc_strat" in cells[k]]
        bar1[f"{mass:g}"] = {"given_min_coverage": given,
                             "cmc_min_coverage": min(cm) if cm else None,
                             "cmc_cells_with_certified_regions": len(cm)}
    bar1_pass = all(v["given_min_coverage"] >= 0.93 and
                    (v["cmc_min_coverage"] is None or v["cmc_min_coverage"] >= 0.93)
                    for v in bar1.values())
    bar2 = {}
    for mass in (1e-3, 1e-4):
        for src in ("given", "cmc"):
            ks = [k for k in cells if k.startswith(f"m={mass:g},") and f"{src}_strat" in cells[k]]
            for k in ks:
                s = cells[k]
                bar2[f"{src} {k}"] = {
                    "width_ratio_mc_over_strat": s[f"{src}_mc"]["median_width"] /
                    max(s[f"{src}_strat"]["median_width"], 1e-300),
                    "upper_ratio_mc_over_strat": s[f"{src}_mc"]["median_upper"] /
                    max(s[f"{src}_strat"]["median_upper"], 1e-300),
                    "width_ratio_riskcert_over_strat": s[f"{src}_riskcert"]["median_width"] /
                    max(s[f"{src}_strat"]["median_width"], 1e-300)}
    bar2_min = min(v["width_ratio_mc_over_strat"] for v in bar2.values())
    out = {"label": "SYNTHETIC (planted analytic regions; no physics, no policy)",
           "command": (f"PYTHONPATH=. python scripts/bench_famrisk.py --reps {a.reps} "
                       f"--n {a.n}" + (" --no-relax" if a.no_relax else "")),
           "boundary_relaxation": RELAX,
           "reps": a.reps, "n_per_family_per_method": a.n, "label_flip_for_cmc": FLIP,
           "delta": DELTA, "cells": cells,
           "bar1_coverage": {"per_mass": bar1, "pass": bool(bar1_pass)},
           "bar2_tightness": {"per_cell": bar2, "min_width_ratio": bar2_min,
                              "pass": bool(bar2_min >= 10.0)}}
    if not a.skip_odd:
        odd = odd_experiment(space, a.reps, a.n, a.seed)
        acc = odd["accuracy_out_among_certified"]
        odd["pass"] = bool(acc is not None and acc >= 0.95 and
                           odd["in_odd_mislabelled_out"] == 0)
        out["bar3_odd"] = odd
        print(f"  ODD: out-of-ODD accuracy {acc} among certified "
              f"(end-to-end {odd['accuracy_out_end_to_end']:.3f}); in-ODD mislabelled out "
              f"{odd['in_odd_mislabelled_out']} [{time.time() - t0:.0f}s]", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    print(f"bar1 coverage pass={bar1_pass}; bar2 min MC/strat width ratio {bar2_min:.1f}",
          flush=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
