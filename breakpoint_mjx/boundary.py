"""Boundary battery for the CI gate: conditions where the BASELINE falls sometimes.

The shipped battery (legged-quadruped-v1) is mostly conditions a good policy survives, so a
small regression shows up there as a handful of new falls the gate cannot tell from noise.
A condition where the baseline falls on 10-60% of seeds is different: the policy is at its
edge there, and a slightly worse candidate tips over measurably more often. The benchmark
measured this (benchmark/reports/TECHNICAL_REPORT.md section 9.3): on 24 such conditions x 8
paired seeds the gate caught a synthetic sigma-0.1 weight perturbation the fixed battery
passed (p = 0.006), caught every larger one, and raised no alarm on an identical pair.

    breakpoint gate-boundary --robot go1 --baseline prod.pkl --out boundary.yaml
    breakpoint gate --robot go1 --baseline prod.pkl --candidate new.pkl \\
        --battery boundary.yaml

HOW IT IS BUILT, baseline only. Conditions are sampled uniformly inside the near-nominal box
(each plausible band shrunk toward nominal to 25% of its width on each side) and inside the
plausible band itself, the same boxes the benchmark used. Every condition runs on
``select_seeds`` selection seeds; the ones whose fall rate lies in ``p_range`` are kept (up
to ``max_cells``, a seeded random subset when more qualify). The candidate is never looked
at, and the written battery runs on fresh seeds disjoint from the selection seeds (and from
the gate's escalation seeds), so the gate's false-alarm bound holds unchanged: the cells are
fixed before either checkpoint is tested, and each (cell, seed) is still a paired outcome.

The output is an ordinary battery YAML (fall rule, pass rule, horizon and robot class
copied from the robot's default battery) plus a ``built_from`` block naming the baseline's
sha256. It is specific to that baseline: when the baseline changes, rebuild it; the gate
notes a mismatch. Every value lies inside the plausible band, so ``breakpoint rate`` and the
gate accept it as they accept a shipped battery.

Sampling, selection and rendering are pure and unit-tested; only the default evaluator
(gate.make_sim_evaluator) touches the simulator.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from breakpoint_mjx import gate, ratings

REGIMES = ("near_nominal", "conservative")
REGIME_PREFIX = {"near_nominal": "near", "conservative": "plaus"}
NEAR_NOMINAL_FRAC = 0.25

# Selection seeds and battery seeds never overlap each other or the gate's escalation seeds
# (gate.ESCALATION_SEED_BASE + i), so the battery's own runs are fresh evidence.
SELECT_SEED_BASE = 4_000_000
BATTERY_SEED_BASE = 5_000_000

DEFAULTS = {"n_conditions": 128, "select_seeds": 20, "p_range": (0.10, 0.60),
            "max_cells": 24, "seeds": 8, "sample_seed": 2025}


# --------------------------------------------------------------------------------------
# pure: sampling, selection, rendering
# --------------------------------------------------------------------------------------

def regime_box(space, regime: str):
    """(lo, hi) in the space's vector order: the plausible band, or for near_nominal that
    band shrunk toward nominal to NEAR_NOMINAL_FRAC of its distance on each side."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; choose from {REGIMES}")
    lo = np.array([d.plausible[0] for d in space.dims.values()], dtype=float)
    hi = np.array([d.plausible[1] for d in space.dims.values()], dtype=float)
    if regime == "near_nominal":
        nom = np.array(space.nominal_vector(), dtype=float)
        lo = nom - NEAR_NOMINAL_FRAC * (nom - lo)
        hi = nom + NEAR_NOMINAL_FRAC * (hi - nom)
    return lo, hi


def _clean(space, dim: str, v: float) -> float:
    d = space.dims[dim]
    if "latency" in dim:
        v = float(round(v))
    v = round(float(v), 5)
    return float(min(max(v, d.plausible[0]), d.plausible[1]))


def sample_conditions(space, n: int, regimes=REGIMES, rng=None) -> list:
    """``n`` conditions split evenly over ``regimes`` -> [(cell_id, {dim: value})]. Dims
    that land on nominal are left out (the battery convention); latency is whole steps."""
    rng = rng if rng is not None else np.random.default_rng(DEFAULTS["sample_seed"])
    if n < len(regimes):
        raise ValueError(f"need at least one condition per regime ({len(regimes)})")
    names, nominal = space.names(), space.nominal_vector()
    out = []
    for k, regime in enumerate(regimes):
        m = n // len(regimes) + (1 if k < n % len(regimes) else 0)
        lo, hi = regime_box(space, regime)
        x = lo + rng.random((m, len(lo))) * (hi - lo)
        for i, row in enumerate(x):
            theta = {}
            for dim, v, nom in zip(names, row, nominal):
                v = _clean(space, dim, v)
                if abs(v - nom) > 1e-12:
                    theta[dim] = v
            out.append((f"{REGIME_PREFIX[regime]}-{i:03d}", theta))
    return out


def select_cells(conditions, outcomes: dict, p_range, max_cells: int, rng=None) -> list:
    """Keep conditions whose selection fall rate is in ``p_range`` (inclusive), at most
    ``max_cells`` (a seeded random subset, file order kept) -> [(id, theta, falls, n)]."""
    lo, hi = map(float, p_range)
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"p_range must satisfy 0 <= lo <= hi <= 1, got {p_range}")
    keep = []
    for cid, theta in conditions:
        o = outcomes[cid]
        falls, n = int(sum(o)), len(o)
        if n and lo <= falls / n <= hi:
            keep.append((cid, theta, falls, n))
    if len(keep) > max_cells:
        rng = rng if rng is not None else np.random.default_rng(DEFAULTS["sample_seed"])
        idx = np.sort(rng.choice(len(keep), max_cells, replace=False))
        keep = [keep[i] for i in idx]
    return keep


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_battery(base, cells, *, battery_id: str, seeds, built_from: dict) -> str:
    """The battery YAML: rules copied from ``base``, one cell per selected condition."""
    if not cells:
        raise ValueError("no cells to write")
    lines = [
        f"# BreakPoint boundary battery {battery_id} (generated by `breakpoint gate-boundary`).",
        "#",
        "# Conditions where the BASELINE named in built_from fell on some, not all, selection",
        "# seeds; each cell's comment gives that selection rate. Run it with",
        "# `breakpoint gate --battery <this file>` (budget standard: smoke caps the horizon,",
        "# which changes what these conditions mean). It is specific to that baseline;",
        "# rebuild it when the baseline changes. Do not hand-edit: the sha256 is reported.",
        f"battery: {battery_id}",
        "version: 1",
        f"robot_class: {base.robot_class}",
        f"horizon: {int(base.horizon)}",
        f"seeds: [{', '.join(str(int(s)) for s in seeds)}]",
        f"fall_rule: {_q(base.fall_rule)}",
        f"pass_rule: {_q(base.pass_rule)}",
        "built_from:",
    ]
    for k, v in built_from.items():
        lines.append(f"  {k}: {_q(v) if isinstance(v, str) else json.dumps(v)}")
    lines.append("cells:")
    for cid, theta, falls, n in cells:
        body = ", ".join(f"{k}: {v!r}" for k, v in theta.items())
        lines += [f"  # baseline fell {falls}/{n} on the selection seeds",
                  f"  - id: {cid}", f"    theta: {{{body}}}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------

def build_boundary_battery(*, robot, baseline, battery=None, space=None,
                           n_conditions=DEFAULTS["n_conditions"],
                           select_seeds=DEFAULTS["select_seeds"],
                           p_range=DEFAULTS["p_range"], max_cells=DEFAULTS["max_cells"],
                           seeds=DEFAULTS["seeds"], sample_seed=DEFAULTS["sample_seed"],
                           regimes=REGIMES, evaluate_fn=None, opts=None, log=print):
    """Sample, run the baseline on the selection seeds, keep the boundary conditions.
    -> (yaml_text or None, info). ``evaluate_fn(spec, cells, seeds, horizon) ->
    {cell_id: [0/1 per seed]}`` is the gate's evaluator interface (injected in tests)."""
    if select_seeds < 1 or seeds < 1 or max_cells < 1:
        raise ValueError("--select-seeds, --seeds and --max-cells must be >= 1")
    if not ratings.policy_file(baseline).is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: "
                                f"{ratings.policy_file(baseline)}")
    base = gate.resolve_battery(robot, battery)
    if space is None:
        from breakpoint_mjx.config import load_theta_space
        from breakpoint_mjx.paths import config_path
        from breakpoint_mjx.targets import get_target

        space = load_theta_space(config_path(get_target(robot).theta_space_path))
    rng = np.random.default_rng(int(sample_seed))
    conditions = sample_conditions(space, int(n_conditions), regimes, rng)
    # the candidates as a battery, so the gate's evaluator (and its theta checks) apply
    probe = dataclasses.replace(base, cells=tuple(conditions))
    ratings.cell_thetas(probe, space)
    if evaluate_fn is None:
        evaluate_fn, _backend = gate.make_sim_evaluator(robot, probe, opts=opts, log=log)
    sel = [SELECT_SEED_BASE + i for i in range(int(select_seeds))]
    log(f"gate-boundary: baseline on {len(conditions)} conditions x {len(sel)} selection "
        f"seeds x {base.horizon} steps ...")
    outcomes = evaluate_fn(baseline, gate.battery_cells(probe), sel, int(base.horizon))
    cells = select_cells(conditions, outcomes, p_range, int(max_cells), rng)
    rates = [sum(outcomes[c]) / len(sel) for c, _ in conditions]
    info = {"conditions": len(conditions), "selected": len(cells),
            "in_range": sum(1 for r in rates if p_range[0] <= r <= p_range[1]),
            "never_fell": sum(1 for r in rates if r == 0.0),
            "always_fell": sum(1 for r in rates if r == 1.0)}
    if not cells:
        return None, info
    sha = ratings.sha256_file(ratings.policy_file(baseline))
    built_from = {"baseline_sha256": sha, "base_battery": base.id,
                  "regimes": list(regimes), "conditions_sampled": len(conditions),
                  "sample_seed": int(sample_seed), "selection_seeds": [sel[0], sel[-1]],
                  "p_range": [float(p_range[0]), float(p_range[1])]}
    text = render_battery(base, cells, battery_id=f"boundary-{sha[:12]}",
                          seeds=[BATTERY_SEED_BASE + i for i in range(int(seeds))],
                          built_from=built_from)
    return text, info


def add_args(ap):
    ap.add_argument("--robot", default="go1", help="registered target (default: go1)")
    ap.add_argument("--baseline", required=True,
                    help="the baseline checkpoint the gate will compare against")
    ap.add_argument("--battery", default=None,
                    help="battery whose rules (fall rule, horizon, pass rule) to copy "
                         "(default: the robot class's battery)")
    ap.add_argument("--out", default="boundary-battery.yaml",
                    help="battery YAML to write (default: boundary-battery.yaml)")
    ap.add_argument("--conditions", type=int, default=DEFAULTS["n_conditions"],
                    help=f"conditions to sample, split over near-nominal and plausible "
                         f"(default {DEFAULTS['n_conditions']})")
    ap.add_argument("--select-seeds", type=int, default=DEFAULTS["select_seeds"],
                    help=f"baseline seeds per condition for selection "
                         f"(default {DEFAULTS['select_seeds']})")
    ap.add_argument("--p-min", type=float, default=DEFAULTS["p_range"][0],
                    help="lowest baseline fall rate kept (default 0.10)")
    ap.add_argument("--p-max", type=float, default=DEFAULTS["p_range"][1],
                    help="highest baseline fall rate kept (default 0.60)")
    ap.add_argument("--max-cells", type=int, default=DEFAULTS["max_cells"],
                    help=f"most cells in the battery (default {DEFAULTS['max_cells']})")
    ap.add_argument("--seeds", type=int, default=DEFAULTS["seeds"],
                    help=f"seeds the battery runs per cell (default {DEFAULTS['seeds']})")
    ap.add_argument("--sample-seed", type=int, default=DEFAULTS["sample_seed"],
                    help="sampling seed (the build is deterministic given it)")
    for flag in ("--obs-spec", "--obs-norm", "--squash", "--activation"):
        ap.add_argument(flag, default=None, help="as `rate`")
    ap.add_argument("--action-scale", type=float, default=None, help="as `rate`")
    ap.add_argument("--mean-dim", type=int, default=None, help="as `rate`")
    return ap


def build(args, evaluate_fn=None, space=None) -> int:
    """Exit 0 with the battery written, 1 when no condition qualified, 2 on an error."""
    try:
        text, info = build_boundary_battery(
            robot=args.robot, baseline=args.baseline, battery=args.battery, space=space,
            n_conditions=args.conditions, select_seeds=args.select_seeds,
            p_range=(args.p_min, args.p_max), max_cells=args.max_cells, seeds=args.seeds,
            sample_seed=args.sample_seed, evaluate_fn=evaluate_fn,
            opts=gate._adapter_opts(args))
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"gate-boundary: error: {e.args[0] if e.args else e}", file=sys.stderr)
        return 2
    print(f"gate-boundary: {info['conditions']} conditions; baseline never fell in "
          f"{info['never_fell']}, always fell in {info['always_fell']}, "
          f"{info['in_range']} in [{args.p_min:g}, {args.p_max:g}]")
    if text is None:
        print("gate-boundary: no condition in range; nothing written. Sample more "
              "(--conditions) or widen --p-min/--p-max.", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    if info["selected"] < args.max_cells:
        print(f"gate-boundary: only {info['selected']} cells (wanted {args.max_cells}); "
              f"the gate has less power. Sample more with --conditions.")
    print(f"wrote {out} ({info['selected']} cells x {args.seeds} seeds). Gate with:\n"
          f"  breakpoint gate --robot {args.robot} --baseline {args.baseline} "
          f"--candidate <ckpt> --battery {out}")
    return 0


def main(argv=None) -> int:
    ap = add_args(argparse.ArgumentParser(
        description="Build a boundary battery for `breakpoint gate` from the baseline."))
    return build(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
