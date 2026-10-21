"""Supplementary analyses added after the full run (post hoc, labelled as such in the report).

* cluster-radius sensitivity: directed / baseline ratio of distinct failure clusters at every
  eps in the config (the predefined metric uses eps 0.10; the others were predefined as
  sensitivity values but were not in the generated tables)
* failure-mode concentration: share of failures per direction-aware dominant dimension and
  the effective number of modes exp(entropy)
* wall-clock medians (the mean is distorted by host stalls; see the timing of each run)

Reads only raw run files; writes new files under <results>/supplementary/ and refuses to
overwrite.

    python -m benchmark.analysis.supplementary --results benchmark/results/full
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark.analysis.metrics import bootstrap_ratio
from benchmark.evaluation.setup import regime_space
from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.equal_budget import dominant_modes
from breakpoint_mjx.paths import config_path

BASELINES = ("uniform", "sobol", "lhs_boundary")


def _runs(results: Path):
    out = {}
    for p in sorted((results / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        out.setdefault((r["policy"], r["regime"], r["method"]), []).append(r)
    return out


def _write(path: Path, rows: list[dict]):
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args(argv)
    results = Path(a.results)
    outdir = results / "supplementary"
    outdir.mkdir(exist_ok=True)
    runs = _runs(results)
    space = load_theta_space(config_path("go1/theta_space.yaml"))

    eps_rows, mode_rows, wall_rows = [], [], []
    for (pol, reg, m), rs in sorted(runs.items()):
        walls = [r["timing"]["method_wall_s"] for r in rs]
        wall_rows.append({"policy": pol, "regime": reg, "method": m, "reps": len(rs),
                          "wall_s_median": round(float(np.median(walls)), 1),
                          "wall_s_mean": round(float(np.mean(walls)), 1),
                          "wall_s_max": round(float(np.max(walls)), 1)})
        box = regime_space(space, reg)
        c = Counter()
        for r in rs:
            X = np.asarray(r["rows"]["theta"], dtype=float)
            f = np.asarray(r["rows"]["failure"], dtype=bool)
            c.update(dominant_modes(X[f], box))
        tot = sum(c.values())
        if tot:
            p = np.array(list(c.values()), dtype=float) / tot
            eff = float(np.exp(-(p * np.log(p)).sum()))
            top = "; ".join(f"{k} {v / tot:.2f}" for k, v in c.most_common(4))
        else:
            eff, top = 0.0, ""
        mode_rows.append({"policy": pol, "regime": reg, "method": m, "failures": tot,
                          "effective_modes": round(eff, 2), "top_modes": top})
        if m != "directed":
            continue
        for b in BASELINES:
            base = runs.get((pol, reg, b))
            if not base:
                continue
            for e in sorted(rs[0]["analysis"]["clusters"], key=float):
                d = [r["analysis"]["clusters"][e]["n"] for r in rs]
                bb = [r["analysis"]["clusters"][e]["n"] for r in base]
                br = bootstrap_ratio(d, bb)
                fmt = lambda x: "" if x is None else round(x, 3)  # noqa: E731
                eps_rows.append({"policy": pol, "regime": reg, "eps": e, "vs": b,
                                 "directed_mean": round(float(np.mean(d)), 1),
                                 "baseline_mean": round(float(np.mean(bb)), 1),
                                 "ratio": fmt(br["ratio"]), "lo": fmt(br["lo"]),
                                 "hi": fmt(br["hi"])})
    _write(outdir / "eps_sensitivity.csv", eps_rows)
    _write(outdir / "mode_concentration.csv", mode_rows)
    _write(outdir / "wall_clock.csv", wall_rows)
    print(f"[supplementary] wrote {outdir}")


if __name__ == "__main__":
    main()
