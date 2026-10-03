"""Pre-registered SYNTHETIC benchmark for Certified Minimal Cause (docs/mincause.md).

Pure numpy on CPU. No physics, no policy, no robot: planted analytic failure functions over
the real Go1 ThetaSpace (16 dims). Every number this prints is SYNTHETIC.

    PYTHONPATH=. python scripts/bench_mincause.py            # 300 functions x p in {0,.1,.2}
    PYTHONPATH=. python scripts/bench_mincause.py --n 30     # quick look
    PYTHONPATH=. python scripts/bench_mincause.py --oc       # exact certifier table only

Writes docs/mincause_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breakpoint_mjx.config import load_theta_space  # noqa: E402
from breakpoint_mjx.margins import compute_attribution  # noqa: E402
from breakpoint_mjx.mincause import (  # noqa: E402
    ALPHA,
    certified_minimal_cause,
    check_label_consistency,
    look_schedule,
    operating_characteristics,
)
from breakpoint_mjx.naming import _ranked_dims  # noqa: E402
from breakpoint_mjx.paths import config_path  # noqa: E402

PS = (0.0, 0.1, 0.2)

# Round-1 numbers from the committed docs/mincause_results.json before the sequential
# certifier (K=8, fixed waves, n=300): median / mean rollouts per family and exact recovery.
ROUND1_REF = {
    "0.0": {"sims_median": 652.0, "sims_mean": 840.72, "exact": 1.0},
    "0.1": {"sims_median": 788.0, "sims_mean": 1024.21, "exact": 1.0},
    "0.2": {"sims_median": 1156.0, "sims_mean": 1463.04, "exact": 0.98},
}


class Planted:
    """AND of one-sided thresholds on 1-3 dims. ``fails(X)`` is the noiseless truth;
    ``score(X)`` is a continuous failure score in [0, 1] with fails <=> score >= 0.5."""

    def __init__(self, rng, nominal, lower, upper, size=None):
        D = len(nominal)
        self.size = int(size or rng.integers(1, 4))
        self.dims = [int(d) for d in rng.choice(D, self.size, replace=False)]
        self.sign, self.thr, self.bound = [], [], []
        for d in self.dims:
            can_hi = upper[d] - nominal[d] > 1e-9
            can_lo = nominal[d] - lower[d] > 1e-9
            hi = bool(rng.random() < 0.5) if (can_hi and can_lo) else can_hi
            bound = upper[d] if hi else lower[d]
            u = rng.uniform(0.2, 0.8)
            self.sign.append(1.0 if hi else -1.0)
            self.bound.append(bound)
            self.thr.append(nominal[d] + u * (bound - nominal[d]))
        self.sign = np.array(self.sign)
        self.thr = np.array(self.thr)
        self.bound = np.array(self.bound)

    def excess(self, X):
        X = np.atleast_2d(X)
        e = (X[:, self.dims] - self.thr) * self.sign / np.abs(self.bound - self.thr)
        return e.min(axis=1)                               # >= 0 iff every clause holds

    def fails(self, X):
        return self.excess(X) >= 0.0

    def score(self, X):
        return np.clip(0.5 + 0.5 * self.excess(X), 0.0, 1.0)

    def directions(self):
        return ["high" if s > 0 else "low" for s in self.sign]

    def sample_failing(self, rng, nominal, lower, upper, span):
        th = np.empty(len(nominal))
        for d in range(len(nominal)):
            while True:
                x = rng.uniform(lower[d], upper[d])
                if abs(x - nominal[d]) > 0.01 * span[d]:
                    break
            th[d] = x
        for j, d in enumerate(self.dims):
            u = rng.uniform(0.05, 1.0)
            th[d] = self.thr[j] + u * (self.bound[j] - self.thr[j])
        assert self.fails(th)[0]
        return th


def flaky_fail_fn(planted, p):
    def fail_fn(X, seeds):
        truth = planted.fails(X)
        if p <= 0:
            return truth
        r = np.random.default_rng(int(seeds[0]) + 7919)
        return truth ^ (r.random(len(truth)) < p)
    return fail_fn


def flaky_score_fn(planted, p, rng):
    def score_fn(X):
        s = planted.score(X)
        if p <= 0:
            return s
        flip = rng.random(len(s)) < p
        return np.where(flip, 1.0 - s, s)
    return score_fn


def _dir_consistent(dims, dirs, theta, nominal):
    return all(("high" if theta[d] > nominal[d] else "low") == dr for d, dr in zip(dims, dirs))


def run_trial(i, space, p, pop_n=4000):
    names = space.names()
    nominal = np.array(space.nominal_vector(), float)
    lower, upper = np.array(space.lower(), float), np.array(space.upper(), float)
    span = upper - lower
    frng = np.random.default_rng(10_000 + i)                      # function identity: p-free
    planted = Planted(frng, nominal, lower, upper)
    distractors = [Planted(frng, nominal, lower, upper) for _ in range(3)]
    theta = planted.sample_failing(frng, nominal, lower, upper, span)
    true_set = set(planted.dims)
    true_dirs = dict(zip(planted.dims, planted.directions()))
    rng = np.random.default_rng(20_000 + i + int(p * 1000) * 100_000)
    out = {"i": i, "p": p, "size": planted.size}

    # CMC (K=8, sequential certifier), the K=1 ablation, and the round-1 fixed-K certifier
    # (cmc_fixed: K seeds per wave, full alpha per wave) for the rollout comparison. The
    # fixed variant draws its seed from its own stream so cmc/cmc_k1 see the same seeds
    # as in round 1.
    rng_fixed = np.random.default_rng(30_000 + i + int(p * 1000) * 100_000)
    for tag, k, seq in (("cmc", 8, True), ("cmc_k1", 1, True), ("cmc_fixed", 8, False)):
        s0 = int((rng_fixed if tag == "cmc_fixed" else rng).integers(1 << 30))
        res = certified_minimal_cause(theta, nominal, lower, upper, flaky_fail_fn(planted, p),
                                      k_seeds=k, seed0=s0, sequential=seq)
        pred = set(res.cause)
        consistent = True
        if res.certified:
            try:
                check_label_consistency(res, names, nominal, dict(zip(names, theta)))
            except AssertionError:
                consistent = False
        shrink_err = [abs(res.theta_min[d] - planted.thr[j]) / span[d]
                      for j, d in enumerate(planted.dims)
                      if res.certified and d in pred]
        out[tag] = {
            "certified": bool(res.certified),
            "exact": bool(res.certified and pred == true_set),
            "exact_dir": bool(res.certified and pred == true_set and all(
                true_dirs[d] == dr for d, dr in zip(res.cause, res.directions))),
            "consistent": bool(consistent),
            "necessary": res.necessary,
            "sims": int(res.sims),
            "calls": int(res.calls),
            "pred_size": len(pred),
            "shrink_err": float(np.mean(shrink_err)) if shrink_err else None,
        }

    # Baseline A: current naming (family mean vs mean of all failures in the run).
    pop = rng.uniform(lower, upper, size=(pop_n, len(names)))

    # GENEROUS to the baseline: its search population is labeled with the NOISELESS truth
    # at every p (flaky labels would drown the family mean), and it is told the true |S|.
    def observed(pl):
        return pl.fails(pop)

    fam = pop[observed(planted)]
    fam = np.vstack([fam, theta[None, :]])
    allf = np.vstack([fam] + [pop[observed(d)] for d in distractors])

    def recs(X):
        return [SimpleNamespace(theta_named=dict(zip(names, x))) for x in X]

    ref = {n: float(allf[:, j].mean()) for j, n in enumerate(names)}
    ranked = _ranked_dims(recs(fam), ref, space)[: planted.size]
    a_dims = [names.index(n) for _, n, _ in ranked]
    a_dirs = [d for _, _, d in ranked]
    out["A"] = {
        "exact": set(a_dims) == true_set,
        "exact_dir": set(a_dims) == true_set and all(true_dirs[d] == dr
                                                      for d, dr in zip(a_dims, a_dirs)),
        "consistent": _dir_consistent(a_dims, a_dirs, theta, nominal),
        "sims": 0,
    }

    # Baseline B: margins.compute_attribution (finite-difference), top-|S|.
    # GENEROUS to the baseline: a NOISELESS continuous score at every p, and the true |S|.
    attr = compute_attribution(theta, lower, upper, names, flaky_score_fn(planted, 0.0, rng))
    top = sorted(range(len(names)), key=lambda j: -attr[names[j]])[: planted.size]
    b_dirs = ["high" if theta[d] > nominal[d] else "low" for d in top]
    out["B"] = {
        "exact": set(top) == true_set,
        "exact_dir": set(top) == true_set and all(true_dirs[d] == dr
                                                   for d, dr in zip(top, b_dirs)),
        "consistent": _dir_consistent(top, b_dirs, theta, nominal),
        "sims": 2 * len(names),
    }
    return out


def summarize(rows):
    summ = {}
    for p in sorted({r["p"] for r in rows}):
        rp = [r for r in rows if r["p"] == p]
        s = {}
        for m in ("cmc", "cmc_k1", "cmc_fixed", "A", "B"):
            v = [r[m] for r in rp]
            s[m] = {
                "exact": float(np.mean([x["exact"] for x in v])),
                "exact_dir": float(np.mean([x["exact_dir"] for x in v])),
                "consistent": float(np.mean([x["consistent"] for x in v])),
                "sims_mean": float(np.mean([x["sims"] for x in v])),
                "sims_median": float(np.median([x["sims"] for x in v])),
            }
            if m.startswith("cmc"):
                s[m]["certified"] = float(np.mean([x["certified"] for x in v]))
                s[m]["necessary_true"] = float(np.mean([x["necessary"] is True for x in v]))
                s[m]["calls_mean"] = float(np.mean([x["calls"] for x in v]))
                errs = [x["shrink_err"] for x in v if x["shrink_err"] is not None]
                s[m]["shrink_err_mean"] = float(np.mean(errs)) if errs else None
                s[m]["exact_by_size"] = {
                    str(k): float(np.mean([r[m]["exact"] for r in rp if r["size"] == k]))
                    for k in (1, 2, 3) if any(r["size"] == k for r in rp)}
        summ[str(p)] = s
    return summ


def oc_table(k=8, mult=4):
    """Exact operating characteristics (no sampling) of the round-1 fixed-K certifier and
    the round-2 sequential one, per configuration, at true failure rate q."""
    n_max = k * mult
    fixed = (list(range(k, n_max + 1, k)), None, None)
    seq = look_schedule(k, n_max, ALPHA)
    rows = {}
    for q in (1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1):
        rows[str(q)] = {
            "fixed": _fixed_oc(k, n_max, q),
            "sequential": list(operating_characteristics(*seq, q)),
        }
    return {"k": k, "max_n": n_max, "looks": seq[0], "fail_c": seq[1], "pass_d": seq[2],
            "fixed_looks": fixed[0], "rows": rows}


def _fixed_oc(k, n_max, q):
    """Round-1 certifier: CP test at full alpha after each K-seed wave (exact DP)."""
    from math import comb

    from breakpoint_mjx.mincause import decide
    state, prev = {0: 1.0}, 0
    pf = pp = en = 0.0
    for n in range(k, n_max + 1, k):
        dist = {}
        for kk, pk in state.items():
            for j in range(n - prev + 1):
                w = pk * comb(n - prev, j) * q ** j * (1 - q) ** (n - prev - j)
                dist[kk + j] = dist.get(kk + j, 0.0) + w
        state = {}
        for kk, pk in dist.items():
            d = decide(kk, n, k)
            if d == "fail":
                pf += pk
                en += pk * n
            elif d == "pass":
                pp += pk
                en += pk * n
            else:
                state[kk] = pk
        prev = n
    und = sum(state.values())
    return [pf, pp, und, en + und * prev]


def print_oc(t):
    print(f"exact per-configuration certifier characteristics, K={t['k']}, "
          f"max {t['max_n']} seeds (alpha={ALPHA})")
    print(f"  sequential looks {t['looks']}  FAIL if k>={t['fail_c']}  PASS if k<={t['pass_d']}")
    print(f"{'q':>5} | {'fixed P(FAIL)':>13} {'E[seeds]':>8} | {'seq P(FAIL)':>11} {'E[seeds]':>8}")
    for q, r in t["rows"].items():
        f, s_ = r["fixed"], r["sequential"]
        print(f"{q:>5} | {f[0]:13.3f} {f[3]:8.2f} | {s_[0]:11.3f} {s_[3]:8.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--oc", action="store_true", help="print the exact certifier table only")
    ap.add_argument("--out", default=str(ROOT / "docs" / "mincause_results.json"))
    args = ap.parse_args(argv)
    oc = oc_table()
    print_oc(oc)
    if args.oc:
        return
    space = load_theta_space(config_path("go1/theta_space.yaml"))
    t0 = time.time()
    rows = [run_trial(i, space, p) for p in PS for i in range(args.n)]
    summ = summarize(rows)
    print("SYNTHETIC analytic benchmark (planted AND failure sets on the Go1 ThetaSpace)")
    print(f"{'p':>4} {'method':>7} {'exact':>6} {'exact+dir':>9} {'dir-consist':>11} "
          f"{'sims/fam':>9}")
    for p, s in summ.items():
        for m, v in s.items():
            print(f"{p:>4} {m:>7} {v['exact']:6.3f} {v['exact_dir']:9.3f} "
                  f"{v['consistent']:11.3f} {v['sims_mean']:9.0f}")
    bars = {
        "cmc_exact_p0>=0.95": summ["0.0"]["cmc"]["exact"] >= 0.95,
        "cmc_exact_p0.1>=0.85": summ["0.1"]["cmc"]["exact"] >= 0.85,
        "cmc_consistency==1": all(summ[p]["cmc"]["consistent"] == 1.0 for p in summ),
        "cmc_minus_best_baseline_p0.1>=0.20": (
            summ["0.1"]["cmc"]["exact"]
            - max(summ["0.1"]["A"]["exact"], summ["0.1"]["B"]["exact"])) >= 0.20,
    }
    print("pre-registered bars:", json.dumps(bars))
    for p, s in summ.items():
        seq, fix = s["cmc"]["sims_median"], s["cmc_fixed"]["sims_median"]
        print(f"p={p}: median rollouts/family sequential {seq:.0f} vs fixed-K {fix:.0f} "
              f"({100 * (1 - seq / max(fix, 1)):.0f}% fewer)")
    print(f"elapsed {time.time() - t0:.1f}s")
    Path(args.out).write_text(json.dumps(
        {"label": "SYNTHETIC analytic benchmark", "n_functions": args.n, "ps": list(PS),
         "summary": summ, "bars": bars, "certifier_oc": oc,
         "round1_reference": ROUND1_REF, "rows": rows}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
