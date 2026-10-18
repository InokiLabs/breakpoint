"""False-alarm rate and power of the CI gate's two decision rules, by simulation.

    python scripts/gate_power.py [--trials 4000] [--out docs/gate_power.json]

No simulator: each (cell, seed) outcome is drawn from a model of two checkpoints run on
the same seeds. A seed has a latent difficulty shared by both checkpoints (correlation
``rho``), so a hard world tends to topple both, which is what pairing exploits. Per-cell
baseline fall rates are taken from the measured smoke rating of the bundled Go1 policy's
battery shape: most cells near zero, a few noisy ones.

Scenarios:
  null-*        candidate is exactly as robust as the baseline (retrained, new weights):
                any alarm is a false alarm;
  one-cell-*    one cell's fall rate rises (a real, local regression);
  spread-*      every cell's fall rate rises a little (a real, diffuse regression).

Decisions compared (breakpoint_mjx/gate.py):
  rule  the original: a cell regresses when the baseline passed it (<= 1/8 falls) and the
        candidate failed it;
  stat  the new default: exact paired sign tests, pooled + Holm per cell, alpha 0.05.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breakpoint_mjx import gate_stats  # noqa: E402

N_CELLS = 15
# baseline per-cell fall rates: 11 quiet cells, 3 noisy, 1 hard (battery-shaped)
BASE_P = np.array([0.0] * 6 + [0.02] * 5 + [0.10, 0.15, 0.20] + [0.40])
PASS_NUM, PASS_DEN = 1, 8


def _phi_inv(p):
    """Standard normal quantile, elementwise (rates of 0 or 1 are clipped, so a 0% cell
    never falls and a 100% cell always does)."""
    from statistics import NormalDist

    nd = NormalDist()
    return np.array([nd.inv_cdf(min(max(x, 1e-12), 1 - 1e-12)) for x in np.ravel(p)]
                    ).reshape(np.shape(p))


def draw(rng, p_base, p_cand, n_seeds, rho):
    """(base, cand) outcome arrays of shape (cells, seeds). Gaussian copula: both share a
    per-(cell, seed) difficulty z with weight sqrt(rho)."""
    shape = (len(p_base), n_seeds)
    z = rng.standard_normal(shape)
    eb = rng.standard_normal(shape)
    ec = rng.standard_normal(shape)
    s, t = math.sqrt(rho), math.sqrt(1.0 - rho)
    xb, xc = s * z + t * eb, s * z + t * ec
    tb = _phi_inv(np.asarray(p_base))[:, None]
    tc = _phi_inv(np.asarray(p_cand))[:, None]
    return (xb < tb).astype(int), (xc < tc).astype(int)


def rule_regressed(base, cand) -> bool:
    n = base.shape[1]
    fb, fc = base.sum(1), cand.sum(1)
    passed_b = fb * PASS_DEN <= PASS_NUM * n
    passed_c = fc * PASS_DEN <= PASS_NUM * n
    return bool(np.any(passed_b & ~passed_c & (fc > fb)))


def stat_regressed(base, cand, alpha=0.05) -> bool:
    pairs = []
    for i in range(base.shape[0]):
        b, c = base[i].tolist(), cand[i].tolist()
        pairs.append((str(i), {"falls": sum(b), "n": len(b), "outcomes": b},
                      {"falls": sum(c), "n": len(c), "outcomes": c}))
    return gate_stats.decide(pairs, alpha=alpha)["regressed"]


def escalate_regressed(rng, p_base, p_cand, base, cand, rho, escalate_to,
                       alpha=0.05) -> bool:
    """stat with escalation: suspect cells re-run on ``escalate_to`` fresh seeds."""
    def pairs_of(b, c, ids):
        return [(str(i), {"falls": int(b[k].sum()), "n": b.shape[1],
                          "outcomes": b[k].tolist()},
                 {"falls": int(c[k].sum()), "n": c.shape[1], "outcomes": c[k].tolist()})
                for k, i in enumerate(ids)]

    def stage2(ids):
        idx = [int(i) for i in ids]
        b2, c2 = draw(rng, p_base[idx], p_cand[idx], escalate_to, rho)
        return pairs_of(b2, c2, idx)

    return gate_stats.two_stage(pairs_of(base, cand, range(base.shape[0])), stage2,
                                alpha=alpha)["regressed"]


def scenarios():
    yield "null", BASE_P
    one = BASE_P.copy()
    one[7] = 0.5            # a quiet cell (0.02) becomes a coin flip
    yield "one-cell 0.02->0.50", one
    one_small = BASE_P.copy()
    one_small[7] = 0.25
    yield "one-cell 0.02->0.25", one_small
    yield "spread +0.10 everywhere", np.minimum(BASE_P + 0.10, 1.0)
    yield "spread +0.05 everywhere", np.minimum(BASE_P + 0.05, 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[4, 8, 32])
    ap.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--escalate-to", type=int, default=32,
                    help="fresh seeds per suspect cell for the 'stat+esc' column")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(0)
    rows = []
    for rho in args.rho:
        for n in args.seeds:
            for name, p_cand in scenarios():
                r = s = e = 0
                for _ in range(args.trials):
                    b, c = draw(rng, BASE_P, p_cand, n, rho)
                    r += rule_regressed(b, c)
                    s += stat_regressed(b, c)
                    e += escalate_regressed(rng, BASE_P, p_cand, b, c, rho,
                                            args.escalate_to)
                t = args.trials
                rows.append({"scenario": name, "seeds": n, "rho": rho,
                             "rule": r / t, "stat": s / t, "stat_escalated": e / t,
                             "escalate_to": args.escalate_to})
                print(f"rho={rho:.1f} seeds={n:>2} {name:<26} rule={r / t:6.1%} "
                      f"stat={s / t:6.1%} stat+esc{args.escalate_to}={e / t:6.1%}",
                      flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"trials": args.trials,
                                              "base_p": BASE_P.tolist(),
                                              "rows": rows}, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
