"""Apply the pre-registered selection rule (BENCHMARK_PLAN.md §12) to the tuning runs.

    python -m benchmark.analysis.select_variant --results benchmark/results/tune \
        --config benchmark/configs/tune.yaml

Rule: among candidates whose eps-0.1 estimated validated distinct failures in conservative
is at least uniform's, pick the highest mean coarse estimated validated distinct (eps-0.3
clusters x leader validation rate) summed over conservative and broad. Also list every
candidate that validated at least one near-nominal failure. Writes selection.json (refuses
to overwrite) and prints the table it decided from.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.evaluation.setup import load_config


def per_run(r):
    v = r["validation"]
    rate = v["leaders_validated"] / v["leaders_checked"] if v["leaders_checked"] else 0.0
    cl = r["analysis"]["clusters"]
    return {"fine": cl["0.1"]["n"] * rate, "coarse": cl["0.3"]["n"] * rate,
            "validated_any": sum(1 for x in v["results"].values() if x["validated"]),
            "failure_rate": r["analysis"]["n_failures"] / r["plan"]["budget"]}


def select(results: Path, bc):
    cells = defaultdict(list)
    for p in sorted((results / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        cells[(r["method"], r["regime"])].append(per_run(r))
    mean = lambda m, reg, k: float(np.mean([x[k] for x in cells[(m, reg)]]))  # noqa: E731
    candidates = list(bc.method_variants)
    table = {}
    for m in ["uniform", "directed", *candidates]:
        table[m] = {f"{k}_{reg}": round(mean(m, reg, k), 1)
                    for reg in ("near_nominal", "conservative", "broad")
                    for k in ("fine", "coarse", "validated_any", "failure_rate")
                    if cells.get((m, reg))}
        table[m]["coarse_sum"] = round(table[m]["coarse_conservative"]
                                       + table[m]["coarse_broad"], 1)
    floor = table["uniform"]["fine_conservative"]
    eligible = [m for m in candidates if table[m]["fine_conservative"] >= floor]
    chosen = max(eligible, key=lambda m: table[m]["coarse_sum"]) if eligible else None
    near = [m for m in candidates if table[m]["validated_any_near_nominal"] > 0]
    return {"rule": "BENCHMARK_PLAN.md §12", "table": table,
            "uniform_fine_conservative_floor": floor, "eligible": eligible,
            "selected": chosen, "near_nominal_validated": near}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    out = select(Path(a.results), load_config(a.config))
    path = Path(a.results) / "selection.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite")
    path.write_text(json.dumps(out, indent=2) + "\n")
    for m, row in out["table"].items():
        print(m, row)
    print("eligible:", out["eligible"], "selected:", out["selected"],
          "near-nominal validated:", out["near_nominal_validated"])


if __name__ == "__main__":
    main()
