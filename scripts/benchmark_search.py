"""Benchmark search strategies against the STANDARD-PRACTICE baseline, on equal budgets.

Standard robustness evaluation is uniform random sampling of the DR space (that is what
"evaluate on N random seeds/configs" does); a hand-rolled adversarial sweep is roughly
CMA-ES. BreakPoint adds surrogate-active and LLM-guided search. This script scores each
strategy's ALREADY-SAVED records (failures.json) on identical per-eval metrics, so the
comparison is apples-to-apples per simulation spent -- no GPU needed, pure CPU analysis.

Usage (Colab or locally; seconds):
  python scripts/benchmark_search.py --run label=path/to/failures.json [label2=path2 ...] \
      [--target go1] [--min-transfer 0.6] [--json out.json]

  # e.g. against the shootout dirs the multi-policy notebook produced:
  python scripts/benchmark_search.py --target go1 \
      --run "random+cmaes=$BP_OUT/shootout_cmaes/go1/failures.json" \
      --run "active=$BP_OUT/shootout_active/go1/failures.json" \
      --run "llm=$BP_OUT/shootout_llm/go1/failures.json"

Within each file, records are ALSO split by their id prefix (random-* vs cmaes-*/llm-*/
active-*/atlas-*), so a single default run yields the random-vs-directed comparison on its
own. Metrics per strategy:

  * evals                  -- simulations spent (the budget; all rates are per-eval).
  * plausible-failure yield -- fraction of evals that found a PLAUSIBLE failure
                              (failure_score >= threshold AND transfer >= min-transfer).
                              Raw failure yield is padded by unsurvivable extremes, so the
                              plausible yield is the honest "useful finds per sim dollar".
  * top-20 weighted        -- mean of the 20 best transfer-weighted failure scores (the
                              engine's long-standing random-vs-directed comparison metric).
  * evals-to-first         -- how many evals until the first plausible failure (id order
                              within each strategy = evaluation order).
  * modes found            -- distinct dominant-deviating dims (direction-aware) among
                              plausible failures: a cheap fingerprint of failure VARIETY.
  * dispersion             -- mean pairwise normalized distance among plausible failures
                              (higher = the finds span the space, not one cluster).

Pure stdlib+numpy; unit-tested in tests/test_benchmark_search.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breakpoint_mjx.config import load_search_config, load_theta_space  # noqa: E402
from breakpoint_mjx.targets import get_target  # noqa: E402
from breakpoint_mjx.paths import config_path  # noqa: E402

# id prefixes -> strategy label (records carry e.g. "random-12", "cmaes-40", "llm-r2-...",
# "active-r3-...", "atlas-...")
_PREFIXES = ("random", "cmaes", "llm", "active", "atlas", "agentic")


def _strategy_of(rec_id):
    rid = rec_id or ""
    for p in _PREFIXES:
        if rid.startswith(p):
            return p
    return "other"


def _plausible_mask(records, threshold, min_transfer):
    return [
        (r.failure_score >= threshold and (r.transfer_score or 0.0) >= min_transfer)
        for r in records
    ]


def _dominant_mode(rec, space, lo, span):
    """Direction-aware dominant deviating dim of one record vs nominal: 'high friction_floor'.

    Deviation is |theta - nominal| normalized by the bound span (scale-free), the same idea
    the family naming uses (vs the nominal reference)."""
    v = np.asarray(rec.theta_vector, dtype=float)
    nom = np.asarray([space.dims[n].nominal for n in space.names()], dtype=float)
    dev = np.abs(v - nom) / span
    j = int(np.argmax(dev))
    name = space.names()[j]
    return ("high " if v[j] > nom[j] else "low ") + name


def score_strategy(records, space, threshold, min_transfer):
    """All metrics for one strategy's record list (evaluation order). Pure."""
    n = len(records)
    if n == 0:
        return {"evals": 0}
    lo = np.asarray(space.lower(), dtype=float)
    hi = np.asarray(space.upper(), dtype=float)
    span = np.where(hi > lo, hi - lo, 1.0)

    mask = _plausible_mask(records, threshold, min_transfer)
    plaus = [r for r, m in zip(records, mask) if m]
    weighted = sorted(
        (r.failure_score * (r.transfer_score or 0.0) for r in records), reverse=True
    )[:20]
    first = next((i + 1 for i, m in enumerate(mask) if m), None)

    modes = sorted({_dominant_mode(r, space, lo, span) for r in plaus})
    if len(plaus) >= 2:
        X = (np.asarray([r.theta_vector for r in plaus], dtype=float) - lo) / span
        m = min(len(X), 512)  # cap the O(m^2) pairwise work
        Xs = X[np.linspace(0, len(X) - 1, m).astype(int)]
        d = np.sqrt(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1))
        dispersion = float(d[np.triu_indices(m, k=1)].mean()) / np.sqrt(X.shape[1])
    else:
        dispersion = 0.0

    return {
        "evals": n,
        "plausible_failures": len(plaus),
        "plausible_yield": round(len(plaus) / n, 4),
        "top20_weighted": round(float(np.mean(weighted)) if weighted else 0.0, 4),
        "evals_to_first_plausible": first,
        "modes_found": len(modes),
        "modes": modes[:12],
        "dispersion": round(dispersion, 4),
    }


def benchmark(runs, space, threshold, min_transfer):
    """runs: {label: [FailureRecord,...]}. Splits each by id prefix, scores everything.
    Returns {strategy_label: metrics} with the file label prefixed when a file mixes
    strategies (e.g. 'cmaes-run/random')."""
    out = {}
    for label, records in runs.items():
        by_strategy = {}
        for r in records:
            by_strategy.setdefault(_strategy_of(r.id), []).append(r)
        for strat, recs in sorted(by_strategy.items()):
            key = f"{label}/{strat}" if len(by_strategy) > 1 else f"{label}/{strat}"
            out[key] = score_strategy(recs, space, threshold, min_transfer)
    return out


def _fmt_table(results):
    cols = ["evals", "plausible_yield", "top20_weighted", "evals_to_first_plausible",
            "modes_found", "dispersion"]
    head = f"{'strategy':38s} " + " ".join(f"{c:>24s}" if c == "evals_to_first_plausible"
                                           else f"{c:>16s}" for c in cols)
    lines = [head, "-" * len(head)]
    for k in sorted(results):
        r = results[k]
        row = f"{k:38s} "
        for c in cols:
            v = r.get(c, "-")
            w = 24 if c == "evals_to_first_plausible" else 16
            row += f"{str(v):>{w}s} "
        lines.append(row)
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Search-strategy benchmark vs the random baseline.")
    ap.add_argument("--run", action="append", required=True,
                    help="label=path/to/failures.json (repeatable)")
    ap.add_argument("--target", default="go1")
    ap.add_argument("--min-transfer", type=float, default=0.6,
                    help="plausibility cutoff for the 'useful find' metrics (default 0.6)")
    ap.add_argument("--json", default=None, help="also write the results dict to this path")
    a = ap.parse_args(argv)

    from breakpoint_mjx.config import merge_target_thresholds
    from breakpoint_mjx.search import load_failures

    target = get_target(a.target)
    cfg = merge_target_thresholds(load_search_config(config_path("search.yaml")), target)
    space = load_theta_space(config_path(target.theta_space_path))

    runs = {}
    for spec in a.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"[skip] {label}: {path} not found", flush=True)
            continue
        runs[label] = load_failures(path)
    if not runs:
        raise SystemExit("no readable --run files")

    results = benchmark(runs, space, cfg.failure_threshold, a.min_transfer)
    print("=== BEGIN BREAKPOINT SEARCH BENCHMARK ===")
    print(f"target={target.name} threshold={cfg.failure_threshold} "
          f"min_transfer={a.min_transfer} ('random' rows = the standard-practice baseline)")
    print(_fmt_table(results))
    print("=== END BREAKPOINT SEARCH BENCHMARK ===")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
