"""CI gate for robot policies: "pytest for robot policies".

On every new checkpoint, the team's OWN runner evaluates the candidate and the baseline
checkpoint on the SAME fixed conditions (battery cells) and the SAME seeds, then asks one
statistical question: is there evidence the candidate falls more than the baseline?
Weights never leave the runner: the report holds per-cell fall counts and the checkpoints'
sha256 hashes, nothing else.

    breakpoint gate --robot go1 --baseline ckpt_old.pkl --candidate ckpt_new.pkl \\
        --budget smoke --out gate.json

Verdict (``--decision stat``, the default). Each (cell, seed) is a pair of outcomes, so the
gate uses exact paired tests (breakpoint_mjx/gate_stats.py): a pooled sign test over every
discordant pair plus a Holm-adjusted sign test per cell, each at alpha/2. The build's
false-alarm probability is at most ``--alpha`` (default 0.05) however many cells the
battery has, and a cell the baseline already fails is tested too, so "2/8 falls became 8/8"
is caught.

Escalation (``--escalate-to``, on by default for the standard budget). Eight seeds cannot
tell a real one-cell regression from noise, so cells where the candidate fell more but not
significantly are re-run, alone, on fresh seeds and decided again. Each stage gets alpha/2
and stage 2 sees only new seeds, so the false-alarm bound still holds
(gate_stats.two_stage). scripts/gate_power.py measures what this buys.

``--decision rule`` keeps the original pass-rule comparison (a cell regresses when the
baseline passed it, the candidate failed it, and the fall rate rose by more than
``--tolerance``). It is simple to explain but raises frequent false alarms on noisy
policies: with 15 cells and 8 seeds, two identical policies with a 10% fall rate trip it
in most runs (docs/ci-gate.md has the measured numbers).

Exit codes: 0 pass, 1 regression, 2 anything else (usage error, a policy that will not
load, an internal error). A crash is never reported as a regression.

Battery loading, fall thresholds and sim-artifact handling are shared with
``breakpoint rate`` (breakpoint_mjx/ratings.py), so the gate and a rating judge the same
battery identically. When BREAKPOINT_API_KEY and BREAKPOINT_PROJECT are set the report
(contract fields only) is POSTed to BreakPoint Web; an upload error only warns.

Relation to scripts/regression.py: that gate diffs FAILURE FAMILIES found by a search
between two stored versions. This gate compares FIXED CONDITIONS run on both checkpoints
in the same job.

The comparison, report and markdown functions are pure (no JAX) and unit-tested on a
mocked evaluator; only ``make_sim_evaluator`` touches the simulator.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from breakpoint_mjx import gate_stats, ratings
from breakpoint_mjx.obs_adversary import OBS_PERTURBATION_MODEL

SCHEMA = ratings.SCHEMA
GATE_FAILED_MSG = "CI GATE FAILED (regression detected)"
DEFAULT_API_URL = "https://breakpoint-inoki.vercel.app"
UPLOAD_TIMEOUT_S = 15
DECISIONS = ("stat", "rule")

# Budget presets, the same as `breakpoint rate`: smoke fits a CPU-only GitHub-hosted runner;
# standard is the battery as declared (all its seeds, its horizon) and wants a GPU.
BUDGETS = {
    "smoke": {"n_seeds": ratings.SMOKE_SEEDS, "horizon_cap": ratings.SMOKE_HORIZON},
    "standard": {"n_seeds": None, "horizon_cap": None},
}

# Escalation seeds are outside every battery's seed list, so stage 2 never re-uses a world
# stage 1 saw; seed i of an escalation is ESCALATION_SEED_BASE + i (deterministic).
ESCALATION_SEED_BASE = 10_000
# Smoke escalates too: with its 4 seeds a single cell can never reach significance on
# stage 1 alone (best one-sided p is 1/16 before Holm), so without escalation a
# regression confined to one cell could not fail the build at all.
DEFAULT_ESCALATE_TO = {"standard": 32, "smoke": 16}

ADAPTER_OPTS = ("obs_norm", "action_scale", "squash", "activation", "mean_dim", "obs_spec")


# --------------------------------------------------------------------------------------
# battery
# --------------------------------------------------------------------------------------

def resolve_battery(robot: str, battery=None) -> "ratings.Battery":
    """The battery for this robot: ``battery`` (an id or a YAML path) or the robot class's
    default, refused when it was written for a different robot class."""
    bat = ratings.load_battery(battery or ratings.default_battery(robot))
    ratings.check_robot(bat, robot)
    return bat


def plan_run(bat, budget: str = "standard"):
    """(seeds, horizon) for a budget preset applied to a battery."""
    if budget not in BUDGETS:
        raise ValueError(f"unknown budget {budget!r}; choose from {sorted(BUDGETS)}")
    b = BUDGETS[budget]
    seeds = [int(s) for s in bat.seeds]
    if b["n_seeds"] is not None:
        seeds = seeds[: b["n_seeds"]]
    horizon = int(bat.horizon)
    if b["horizon_cap"] is not None:
        horizon = min(horizon, int(b["horizon_cap"]))
    return seeds, horizon


def battery_cells(bat) -> list:
    """The battery's cells as [{"id", "theta"}] (the evaluator interface)."""
    return [{"id": cid, "theta": dict(theta)} for cid, theta in bat.cells]


# --------------------------------------------------------------------------------------
# pure: cells, comparison, report, markdown
# --------------------------------------------------------------------------------------

def cell_result(cell_id, outcome, n: int, pass_num: int, pass_den: int) -> dict:
    """One contract cell row. ``outcome`` is the fall count (int), the per-seed 0/1 list,
    or ``{"outcomes": [0/1 per seed], "tracking_error": [float per seed]}``; the lists are
    kept so the verdict can use the seed pairing (and the degradation test the errors)."""
    tracking = None
    if isinstance(outcome, dict):
        tracking = outcome.get("tracking_error")
        outcome = outcome["outcomes"]
        if tracking is not None and len(tracking) != len(outcome):
            raise ValueError(f"cell {cell_id!r}: {len(tracking)} tracking errors for "
                             f"{len(outcome)} seeds")
    if isinstance(outcome, (list, tuple)):
        outcomes = [int(bool(x)) for x in outcome]
        if len(outcomes) != n:
            raise ValueError(f"cell {cell_id!r}: {len(outcomes)} outcomes for {n} seeds")
        falls = sum(outcomes)
    else:
        outcomes, falls = None, int(outcome)
    if not 0 <= falls <= n:
        raise ValueError(f"cell {cell_id!r}: {falls} falls out of {n} seeds")
    lo, hi = ratings.clopper_pearson(falls, n)
    row = {"id": str(cell_id), "n": n, "falls": falls,
           "fall_rate": round(falls / n, 6) if n else 0.0,
           # widened outward to 4 decimals so rounding never narrows the interval
           "ci95": [math.floor(lo * 1e4) / 1e4, min(1.0, math.ceil(hi * 1e4) / 1e4)],
           "passed": ratings.cell_passed(falls, n, pass_num, pass_den)}
    if outcomes is not None:
        row["outcomes"] = outcomes
    if tracking is not None:
        row["tracking_error"] = [float(x) for x in tracking]
    return row


def degradation_pairs(pairs) -> list:
    """(baseline, candidate) tracking errors for every (cell, seed) where neither fell."""
    out = []
    for _cid, b, c in pairs:
        bo, co = b.get("outcomes"), c.get("outcomes")
        bt, ct = b.get("tracking_error"), c.get("tracking_error")
        if None in (bo, co, bt, ct) or len(bo) != len(co):
            continue
        out += [(bt[i], ct[i]) for i in range(len(bo)) if not bo[i] and not co[i]]
    return out


def _rule_status(bc, c, delta, tolerance) -> str:
    if bc["passed"] and not c["passed"]:
        return "REGRESSED" if delta > tolerance + 1e-12 else "within tolerance"
    if not bc["passed"] and c["passed"]:
        return "fixed"
    if not c["passed"]:
        return "fails (baseline too)"
    return "pass"


def compare_cells(baseline: list, candidate: list, tolerance: float = 0.0,
                  decision: str = "stat", alpha: float = gate_stats.DEFAULT_ALPHA,
                  run_stage2=None, degradation_margin=None, degradation_alpha=None,
                  fail_on_degradation: bool = False) -> dict:
    """Compare the two runs cell by cell and decide the gate.

    Every row carries both the statistical result (discordant counts, p, Holm p) and the
    pass-rule status, whichever ``decision`` is used; ``new_failing_cells`` lists the
    cells that decided a failure, ``warnings`` the cells where the candidate fell more
    without that being enough evidence.

    ``run_stage2(cell_ids) -> pairs`` turns on escalation (``--decision stat`` only): the
    suspect cells are re-run on fresh seeds and decided again, each stage at alpha/2
    (gate_stats.two_stage).

    ``degradation_margin`` (not None) adds the paired tracking-error test over seeds where
    neither fell, at ``degradation_alpha``. It is advisory unless ``fail_on_degradation``;
    the caller then splits alpha between it and the fall tests."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    if tolerance < 0:
        raise ValueError("--tolerance must be >= 0")
    b = {c["id"]: c for c in baseline}
    pairs = []
    for c in candidate:
        bc = b.get(c["id"])
        if bc is None:
            raise ValueError(f"cell {c['id']!r} missing from the baseline run")
        pairs.append((c["id"], bc, c))
    if decision == "stat" and run_stage2 is not None:
        two = gate_stats.two_stage(pairs, run_stage2, alpha=alpha, tolerance=tolerance)
        stat, stage2 = two["stage1"], two["stage2"]
    else:
        two = None
        stat, stage2 = gate_stats.decide(pairs, alpha=alpha, tolerance=tolerance), None
    s2 = {t["id"]: t for t in (stage2["cells"] if stage2 else [])}

    rows = []
    for (cid, bc, c), t in zip(pairs, stat["cells"]):
        delta = float(c["fall_rate"]) - float(bc["fall_rate"])
        rule = _rule_status(bc, c, delta, tolerance)
        e = s2.get(cid)
        if decision == "stat":
            # a cell that was escalated is attributed on its fresh-seed evidence: the
            # stage-1 pooled verdict says "the battery got worse", not which cells did
            if t["significant"]:
                status = "REGRESSED"
            elif e is not None and e["significant"]:
                status = "REGRESSED (escalated)"
            elif (e is not None and stage2["pooled"]["significant"]
                  and e["delta"] > tolerance + 1e-12):
                status = "REGRESSED (escalated, pooled)"
            elif e is not None and e["delta"] <= tolerance + 1e-12:
                status = "cleared on more seeds"
            elif stat["pooled"]["significant"] and delta > tolerance + 1e-12:
                status = "REGRESSED (pooled)"
            elif delta > tolerance + 1e-12:
                status = "worse, not significant"
            elif delta < 0:
                status = "better"
            else:
                status = "pass" if c["passed"] else "fails (baseline too)"
        else:
            status = rule
        rows.append({"id": cid, "baseline": bc, "candidate": c, "delta": round(delta, 6),
                     "status": status, "rule_status": rule,
                     "up": t["up"], "down": t["down"], "p": t["p"],
                     "p_holm": t["p_holm"], "escalated": e})
    new_failing = [r["id"] for r in rows if r["status"].startswith("REGRESSED")]
    warnings = [r["id"] for r in rows if r["status"] == "worse, not significant"
                or (decision == "rule" and r["delta"] > tolerance + 1e-12
                    and r["id"] not in new_failing)]
    if decision == "stat":
        regressed = two["regressed"] if two else stat["regressed"]
    else:
        regressed = bool(new_failing)
    degradation = None
    if degradation_margin is not None:
        dp = degradation_pairs(pairs)
        if dp:
            degradation = gate_stats.degradation_test(
                dp, margin=degradation_margin,
                alpha=alpha if degradation_alpha is None else degradation_alpha)
            degradation["fails_build"] = bool(fail_on_degradation)
            if fail_on_degradation and degradation["significant"]:
                regressed = True
    return {"rows": rows, "new_failing_cells": new_failing, "warnings": warnings,
            "regressed": bool(regressed), "decision": decision, "stat": stat,
            "alpha": alpha, "stage2": stage2, "escalated": two["escalated"] if two else [],
            "degradation": degradation}


def summary_block(cells: list, budget: str, seeds, horizon) -> dict:
    budget = "smoke" if budget == "smoke" else "full"
    return {"cells_passed": sum(1 for c in cells if c["passed"]),
            "cells_total": len(cells),
            "label": ratings.summary_label(cells, budget, seeds, horizon)}


def tolerance_label(tolerance: float) -> str:
    return f"fall_rate_delta<={tolerance:g}"


def _public_cell(c: dict) -> dict:
    return {k: v for k, v in c.items() if k not in ("outcomes", "tracking_error")}


def _degradation_block(deg):
    if deg is None:
        return None
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in deg.items()}


def _escalation_block(comparison, escalation_seeds):
    s2 = comparison.get("stage2")
    if not comparison.get("escalated"):
        return {"seeds": list(escalation_seeds or []), "cells": [], "pooled": None,
                "cell_tests": []}
    return {"seeds": list(escalation_seeds), "cells": list(comparison["escalated"]),
            "pooled": {k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in s2["pooled"].items()},
            "cell_tests": [{"id": t["id"], "up": t["up"], "down": t["down"],
                            "p": round(t["p"], 6), "p_holm": round(t["p_holm"], 6),
                            "significant": t["significant"]} for t in s2["cells"]]}


def build_report(*, robot, bat, policy_name, policy_sha256, baseline_sha256,
                 candidate_cells, comparison, seeds, horizon, budget, backend, tolerance,
                 engine_version, commit, date=None, source="local",
                 escalation_seeds=()) -> dict:
    """The kind="gate" report. ``cells``/``summary`` describe the CANDIDATE; the baseline
    and the verdict enter through ``gate``."""
    st = comparison["stat"]
    return {
        "schema": SCHEMA,
        "kind": "gate",
        "battery": {"id": bat.id, "sha256": bat.sha256},
        "engine": {"version": engine_version, "commit": commit,
                   "obs_model": OBS_PERTURBATION_MODEL},
        "robot": robot,
        "policy": {"name": policy_name, "source": source, "sha256": policy_sha256},
        "run": {"date": date or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "backend": backend, "seeds": [int(s) for s in seeds], "runner": "self",
                "budget": budget, "horizon": int(horizon)},
        "cells": [_public_cell(c) for c in candidate_cells],
        "summary": summary_block(candidate_cells, budget, seeds, horizon),
        "gate": {
            "baseline_sha256": baseline_sha256, "candidate_sha256": policy_sha256,
            "new_failing_cells": list(comparison["new_failing_cells"]),
            "regressed": bool(comparison["regressed"]),
            "tolerance": tolerance_label(tolerance),
            "decision": comparison["decision"],
            "method": st["method"] if comparison["decision"] == "stat" else
                      "pass-rule comparison (no significance test)",
            "alpha": comparison["alpha"],
            "pooled": {k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in st["pooled"].items()},
            "baseline_falls": {r["id"]: r["baseline"]["falls"] for r in comparison["rows"]},
            "cell_tests": [{"id": r["id"], "up": r["up"], "down": r["down"],
                            "p": round(r["p"], 6), "p_holm": round(r["p_holm"], 6),
                            "status": r["status"]} for r in comparison["rows"]],
            "warnings": list(comparison["warnings"]),
            "min_detectable": {"per_cell": st["min_up_per_cell"],
                               "pooled": st["min_up_pooled"]},
            "escalation": _escalation_block(comparison, escalation_seeds),
            "degradation": _degradation_block(comparison.get("degradation")),
        },
        "attestation": None,
    }


CONTRACT_KEYS = ("schema", "kind", "battery", "engine", "robot", "policy", "run", "cells",
                 "summary", "gate", "attestation")


def contract_payload(report: dict) -> dict:
    """Only the contract's top-level fields (no weights, paths, or extras)."""
    return {k: report.get(k) for k in CONTRACT_KEYS}


def _fmt_p(p) -> str:
    return "-" if p is None else (f"{p:.3f}" if p >= 0.001 else f"{p:.1e}")


def to_markdown(report: dict, comparison: dict, *, baseline_name: str, notes=()) -> str:
    g, run = report["gate"], report["run"]
    verdict = "FAILED: regression detected" if g["regressed"] else "PASSED"
    lines = [f"## BreakPoint gate: {verdict}", ""]
    deg = g.get("degradation")
    if g["regressed"] and g["new_failing_cells"]:
        lines.append(f"The candidate falls more than the baseline in "
                     f"{len(g['new_failing_cells'])} condition(s): "
                     + ", ".join(f"`{c}`" for c in g["new_failing_cells"]) + ".")
    elif g["regressed"]:
        lines.append("The candidate does not fall more, but it tracks its command "
                     "significantly worse on seeds where neither checkpoint fell.")
    elif comparison["warnings"]:
        lines.append("No significant regression, but the candidate fell more often in "
                     + ", ".join(f"`{c}`" for c in comparison["warnings"])
                     + ". With this many seeds that is within noise; rerun with more "
                       "seeds (`--budget standard`) to settle it.")
    else:
        lines.append("No condition shows the candidate falling more than the baseline.")
    pooled = g["pooled"]
    lines += [
        "",
        f"- robot: `{report['robot']}`; battery: `{report['battery']['id']}` "
        f"(sha256 `{report['battery']['sha256'][:12]}`)",
        f"- baseline: `{baseline_name}` (sha256 `{g['baseline_sha256'][:12]}`)",
        f"- candidate: `{report['policy']['name']}` (sha256 `{g['candidate_sha256'][:12]}`)",
        f"- budget: `{run['budget']}`; seeds {run['seeds']}; horizon {run['horizon']} steps; "
        f"backend `{run['backend']}`; tolerance `{g['tolerance']}`",
        f"- decision: `{g['decision']}` ({g['method']}, alpha {g['alpha']:g})",
    ]
    if pooled.get("up") is not None:
        lines.append(f"- pooled: candidate fell alone in {pooled['up']} seed-pairs, the "
                     f"baseline alone in {pooled['down']} (p = {_fmt_p(pooled['p'])})")
    md = g["min_detectable"]
    if g["decision"] == "stat":
        per = (f"one cell alone fails the build once {md['per_cell']} of "
               f"{len(run['seeds'])} seeds newly fall" if md["per_cell"] else
               f"no single cell can fail the build on {len(run['seeds'])} seeds alone")
        pool = (f"the battery does at {md['pooled']} net seed-pairs" if md["pooled"]
                else "nor can the battery")
        lines.append(f"- sensitivity: {per}; {pool}")
    if deg:
        med = ("-" if deg["median_ratio"] is None else f"x{deg['median_ratio']:.2f}")
        verdict_d = ("significant" if deg["significant"] else "not significant")
        role = "fails the build" if deg["fails_build"] else "advisory"
        lines.append(f"- tracking ({role}): over {deg['n_pairs']} seed-pairs where neither "
                     f"fell, the candidate tracked worse by >{deg['margin']:.0%} in "
                     f"{deg['up']}, better in {deg['down']} (median error {med}, "
                     f"p = {_fmt_p(deg['p'])}, {verdict_d})")
    esc = g["escalation"]
    if esc["cells"]:
        lines.append(f"- escalated: {', '.join(f'`{c}`' for c in esc['cells'])} re-run on "
                     f"{len(esc['seeds'])} fresh seeds (each stage at alpha/2)")
    elif g["decision"] == "stat" and esc["seeds"]:
        lines.append(f"- escalation: armed ({len(esc['seeds'])} fresh seeds), nothing "
                     f"suspect to re-run")
    lines += [f"- candidate: {report['summary']['label']}", "",
              "| cell | baseline falls | candidate falls | candidate 95% CI | p (Holm) "
              "| status |", "|---|---|---|---|---|---|"]
    for r in comparison["rows"]:
        lo, hi = r["candidate"]["ci95"]
        status = f"**{r['status']}**" if r["status"].startswith("REGRESSED") else r["status"]
        if r.get("escalated"):
            e = r["escalated"]
            status += f" (+{len(esc['seeds'])} seeds: {e['up']} vs {e['down']})"
        lines.append(f"| `{r['id']}` | {r['baseline']['falls']}/{r['baseline']['n']} | "
                     f"{r['candidate']['falls']}/{r['candidate']['n']} | "
                     f"[{lo:.2f}, {hi:.2f}] | {_fmt_p(r['p_holm'])} | {status} |")
    lines.append("")
    for n in notes:
        lines.append(f"> note: {n}")
    if notes:
        lines.append("")
    lines.append("Simulation only. Weights never left this runner.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# upload (optional; never fails the build)
# --------------------------------------------------------------------------------------

def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def maybe_upload(report: dict, environ=None, opener=None, log=None) -> str:
    """POST the contract payload to <BREAKPOINT_API_URL>/api/projects/<project>/gate with
    the project api key. Returns "skipped", "uploaded" or "failed"; never raises."""
    env = os.environ if environ is None else environ
    log = log or (lambda m: print(m, file=sys.stderr))
    key = (env.get("BREAKPOINT_API_KEY") or "").strip()
    project = (env.get("BREAKPOINT_PROJECT") or "").strip()
    if not key or not project:
        return "skipped"
    base = (env.get("BREAKPOINT_API_URL") or DEFAULT_API_URL).rstrip("/")
    url = f"{base}/api/projects/{urllib.parse.quote(project, safe='')}/gate"
    try:
        req = urllib.request.Request(
            url, method="POST", data=_canonical(contract_payload(report)),
            headers={"authorization": f"Bearer {key}",
                     "content-type": "application/json"})
        open_fn = opener or urllib.request.urlopen
        with open_fn(req, timeout=UPLOAD_TIMEOUT_S) as r:
            status = getattr(r, "status", 200)
        if not 200 <= int(status) < 300:
            log(f"WARNING: gate upload to {url} returned HTTP {status} "
                f"(gate result is unaffected)")
            return "failed"
        return "uploaded"
    except urllib.error.HTTPError as e:
        log(f"WARNING: gate upload to {url} failed: HTTP {e.code} "
            f"(gate result is unaffected)")
    except Exception as e:  # network, DNS, timeout, anything: warn only
        log(f"WARNING: gate upload to {url} failed: {type(e).__name__}: {e} "
            f"(gate result is unaffected)")
    return "failed"


# --------------------------------------------------------------------------------------
# simulator evaluator
# --------------------------------------------------------------------------------------

def make_sim_evaluator(robot: str, bat, opts=None, log=print):
    """Return (evaluate_fn, backend). ``evaluate_fn(spec, cells, seeds, horizon)`` returns
    {cell_id: {"outcomes": [0/1 per seed], "tracking_error": [per seed]}}. It is the ratings runner (same thresholds from the
    battery's fall_rule, same theta validation, a sim_invalid rollout counted as a fall),
    so a gate and a rating of the same checkpoint agree cell for cell."""
    from breakpoint_mjx.config import (load_search_config, load_theta_space,
                                       merge_target_thresholds)
    from breakpoint_mjx.paths import config_path
    from breakpoint_mjx.targets import get_target

    from breakpoint_mjx.accel import tune_for_gpu

    tune_for_gpu()
    from breakpoint_mjx.check_policy import _load_env, load_policy

    opts = opts or {}
    target = get_target(robot)
    space = load_theta_space(config_path(target.theta_space_path))
    thetas = ratings.cell_thetas(bat, space)
    cfg = merge_target_thresholds(load_search_config(config_path("search.yaml"), env={}),
                                  target)
    thresholds = dict(cfg.thresholds, trunk_height_min=bat.fall_height_m,
                      up_proj_min=bat.up_proj_min)
    env = _load_env(target, opts.get("obs_spec"))
    backend = ratings.backend_name()

    battery_ids = [cid for cid, _ in bat.cells]

    def evaluate_fn(spec, cells, seeds, horizon):
        # Always the full battery batch (one compiled shape); a subset (escalation) just
        # keeps the rows it asked for.
        unknown = [c["id"] for c in cells if c["id"] not in battery_ids]
        if unknown:
            raise ValueError(f"cells not in the battery: {unknown}")
        policy_fn, _norm, _map, label = load_policy(target, env, spec, opts)
        log(f"gate: {label}: {len(cells)} cells x {len(seeds)} seeds x {horizon} steps "
            f"(first call compiles, ~1-3 min on CPU) ...")
        run = ratings.make_runner(target, env, policy_fn, space, int(horizon), thresholds)
        want = {c["id"] for c in cells}
        out = {cid: {"outcomes": [], "tracking_error": []}
               for cid in battery_ids if cid in want}
        for s in seeds:
            sigs = run(thetas, int(s))
            for cid, sig in zip(battery_ids, sigs):
                if cid in want:
                    out[cid]["outcomes"].append(int(bool(float(sig.get("fall", 0.0))
                                                         or float(sig.get("sim_invalid", 0.0)))))
                    out[cid]["tracking_error"].append(float(sig.get("tracking_error", 0.0)))
        return out

    return evaluate_fn, backend


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------

def run_gate(*, robot, baseline, candidate, battery=None, budget="standard",
             tolerance=0.0, decision="stat", alpha=gate_stats.DEFAULT_ALPHA,
             escalate_to=None, evaluate_fn=None, backend=None, opts=None,
             candidate_name=None, baseline_name=None, log=print,
             degradation_margin=gate_stats.DEFAULT_DEGRADATION_MARGIN,
             fail_on_degradation=False):
    """Evaluate both checkpoints on the same cells/seeds; return (report, comparison,
    markdown). ``evaluate_fn(spec, cells, seeds, horizon) -> {cell_id: falls | [0/1]}``
    is injected in tests; the default builds the simulator evaluator.

    ``fail_on_degradation`` lets a significant tracking-error degradation fail the build;
    alpha is then split in half between it and the fall tests, so false alarms stay at or
    below ``alpha`` overall. Off, the tracking test is reported at full alpha as advisory.
    ``degradation_margin=None`` turns the tracking test off."""
    from breakpoint_mjx import __version__

    if tolerance < 0:
        raise ValueError("--tolerance must be >= 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("--alpha must be in (0, 1)")
    if decision not in DECISIONS:
        raise ValueError(f"--decision must be one of {DECISIONS}")
    if budget not in BUDGETS:
        raise ValueError(f"unknown budget {budget!r}; choose from {sorted(BUDGETS)}")
    if escalate_to is None:
        escalate_to = DEFAULT_ESCALATE_TO[budget]
    if escalate_to < 0:
        raise ValueError("--escalate-to must be >= 0")
    if fail_on_degradation and degradation_margin is None:
        raise ValueError("--fail-on-degradation needs the tracking test (a margin)")
    fall_alpha = alpha / 2.0 if fail_on_degradation else alpha
    deg_alpha = alpha / 2.0 if fail_on_degradation else alpha
    for label, spec in (("baseline", baseline), ("candidate", candidate)):
        if not ratings.policy_file(spec).is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: "
                                    f"{ratings.policy_file(spec)}")
    bat = resolve_battery(robot, battery)
    seeds, horizon = plan_run(bat, budget)
    notes = []
    if budget != "standard":
        notes.append(f"budget {budget}: {len(seeds)} of {len(bat.seeds)} battery seeds, "
                     f"horizon {horizon} (battery declares {bat.horizon}); a smoke gate "
                     f"screens for gross regressions only")
    cells = battery_cells(bat)
    if evaluate_fn is None:
        evaluate_fn, backend = make_sim_evaluator(robot, bat, opts=opts, log=log)

    def run_one(spec, which=cells, on=seeds):
        res = evaluate_fn(spec, which, on, horizon)
        missing = [c["id"] for c in which if c["id"] not in res]
        if missing:
            raise RuntimeError(f"evaluator returned no result for cells {missing}")
        return [cell_result(c["id"], res[c["id"]], len(on), bat.pass_num, bat.pass_den)
                for c in which]

    esc_seeds = [ESCALATION_SEED_BASE + i for i in range(escalate_to)]

    def run_stage2(cell_ids):
        which = [c for c in cells if c["id"] in set(cell_ids)]
        log(f"gate: escalating {len(which)} suspect cell(s) to {len(esc_seeds)} fresh "
            f"seeds: {', '.join(cell_ids)}")
        b2, c2 = run_one(baseline, which, esc_seeds), run_one(candidate, which, esc_seeds)
        return [(b["id"], b, c) for b, c in zip(b2, c2)]

    base_cells = run_one(baseline)
    cand_cells = run_one(candidate)
    comparison = compare_cells(
        base_cells, cand_cells, tolerance, decision, fall_alpha,
        run_stage2=run_stage2 if (decision == "stat" and esc_seeds) else None,
        degradation_margin=degradation_margin, degradation_alpha=deg_alpha,
        fail_on_degradation=fail_on_degradation)
    if fail_on_degradation:
        notes.append(f"alpha {alpha:g} is split: {fall_alpha:g} for the fall tests, "
                     f"{deg_alpha:g} for the tracking test")
    b_nom = next((c for c in base_cells if c["id"] == "nominal"), None)
    if b_nom is not None and not b_nom["passed"]:
        notes.append("the BASELINE fails the nominal cell; check it with "
                     "`breakpoint check-policy` before trusting any comparison")
    if comparison["stat"]["method"].startswith("one-sided Fisher"):
        notes.append("the evaluator returned fall counts only, so the seed pairing was "
                     "not used (less power)")
    cand_sha = ratings.sha256_file(ratings.policy_file(candidate))
    base_sha = ratings.sha256_file(ratings.policy_file(baseline))
    if cand_sha == base_sha:
        notes.append("baseline and candidate are the same file (identical sha256)")
    if bat.baseline_sha256 and bat.baseline_sha256 != base_sha:
        notes.append(f"battery {bat.id} was built from a different baseline (sha256 "
                     f"{bat.baseline_sha256[:12]}); its cells may no longer sit at this "
                     f"baseline's edge. Rebuild it with `breakpoint gate-boundary`")
    if bat.baseline_sha256 and budget != "standard":
        notes.append("a boundary battery's fall rates were measured at the full horizon; "
                     "a capped-horizon budget weakens it")
    report = build_report(
        robot=robot, bat=bat,
        policy_name=candidate_name or ratings.policy_file(candidate).name,
        policy_sha256=cand_sha, baseline_sha256=base_sha, candidate_cells=cand_cells,
        comparison=comparison, seeds=seeds, horizon=horizon, budget=budget,
        backend=backend or "mjx-cpu", tolerance=tolerance, engine_version=__version__,
        commit=ratings.engine_commit(),
        escalation_seeds=esc_seeds if decision == "stat" else ())
    md = to_markdown(report, comparison,
                     baseline_name=baseline_name or ratings.policy_file(baseline).name,
                     notes=notes)
    return report, comparison, md


def add_args(ap):
    ap.add_argument("--robot", default="go1", help="registered target (default: go1)")
    ap.add_argument("--baseline", required=True, help="baseline checkpoint (the one in prod)")
    ap.add_argument("--candidate", required=True, help="candidate checkpoint under test")
    ap.add_argument("--battery", default=None,
                    help="battery id under breakpoint_mjx/batteries/ or a YAML path "
                         "(default: the robot class's battery, as `breakpoint rate`)")
    ap.add_argument("--budget", default="standard", choices=sorted(BUDGETS),
                    help=f"smoke: {ratings.SMOKE_SEEDS} seeds, horizon <= "
                         f"{ratings.SMOKE_HORIZON} (CPU runners); standard: the battery "
                         f"as declared (GPU runner)")
    ap.add_argument("--decision", default="stat", choices=DECISIONS,
                    help="stat (default): exact paired tests, false alarms <= --alpha; "
                         "rule: the original pass-rule comparison, no significance test")
    ap.add_argument("--alpha", type=float, default=gate_stats.DEFAULT_ALPHA,
                    help="false-alarm budget for --decision stat (default 0.05)")
    ap.add_argument("--escalate-to", type=int, default=None,
                    help="re-run cells where the candidate fell more, but not significantly, "
                         "on this many fresh seeds (default: 32 for standard, 16 for "
                         "smoke; 0 = off). Costs only the suspect cells; false alarms stay "
                         "<= --alpha")
    ap.add_argument("--degradation-margin", type=float,
                    default=gate_stats.DEFAULT_DEGRADATION_MARGIN,
                    help="tracking-error deadband for the paired degradation test over "
                         "seeds where neither checkpoint fell (default 0.10 = 10%%; "
                         "negative = off)")
    ap.add_argument("--fail-on-degradation", action="store_true",
                    help="fail the build when the candidate tracks its command "
                         "significantly worse (alpha is then split with the fall tests)")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="fall-rate margin: an increase at or below this never fails "
                         "the build, however significant")
    ap.add_argument("--out", default="gate.json", help="report path (default: gate.json)")
    ap.add_argument("--summary-out", default=None,
                    help="also write the markdown summary here (e.g. $GITHUB_STEP_SUMMARY; "
                         "appended)")
    ap.add_argument("--candidate-name", default=None, help="name in the report (default: file name)")
    ap.add_argument("--baseline-name", default=None, help="name in the summary (default: file name)")
    ap.add_argument("--no-upload", action="store_true",
                    help="never upload, even when BREAKPOINT_API_KEY/BREAKPOINT_PROJECT are set")
    # the same policy-adapter flags as `breakpoint rate`, so a contract/BYO policy gates
    ap.add_argument("--obs-spec", default=None, help="policy contract YAML (as `rate`)")
    ap.add_argument("--obs-norm", default=None, help="observation normalizer (as `rate`)")
    ap.add_argument("--action-scale", type=float, default=None, help="as `rate`")
    ap.add_argument("--squash", default=None, help="as `rate`")
    ap.add_argument("--activation", default=None, help="as `rate`")
    ap.add_argument("--mean-dim", type=int, default=None, help="as `rate`")
    return ap


def _adapter_opts(args) -> dict:
    opts = {k: getattr(args, k, None) for k in ADAPTER_OPTS}
    if opts["obs_spec"] is None and os.environ.get("BP_OBS_SPEC"):
        opts["obs_spec"] = os.environ["BP_OBS_SPEC"]
    return opts


def _margin(m):
    return None if m is None or m < 0 else float(m)


def gate(args, evaluate_fn=None, backend=None, environ=None, opener=None) -> int:
    try:
        report, comparison, md = run_gate(
            robot=args.robot, baseline=args.baseline, candidate=args.candidate,
            battery=args.battery, budget=args.budget, tolerance=args.tolerance,
            decision=getattr(args, "decision", "stat"),
            alpha=getattr(args, "alpha", gate_stats.DEFAULT_ALPHA),
            escalate_to=getattr(args, "escalate_to", None),
            evaluate_fn=evaluate_fn, backend=backend, opts=_adapter_opts(args),
            candidate_name=args.candidate_name, baseline_name=args.baseline_name,
            degradation_margin=_margin(getattr(args, "degradation_margin",
                                               gate_stats.DEFAULT_DEGRADATION_MARGIN)),
            fail_on_degradation=bool(getattr(args, "fail_on_degradation", False)))
    except (FileNotFoundError, ValueError, KeyError) as e:   # BatteryError is a ValueError
        print(f"gate: error: {e.args[0] if e.args else e}", file=sys.stderr)
        return 2
    except SystemExit as e:   # the BYO adapter raises SystemExit(str) on a dim mismatch
        if isinstance(e.code, str):
            print(f"gate: cannot load a policy: {e.code}", file=sys.stderr)
            return 2
        raise
    except Exception as e:   # noqa: BLE001 - a crash must never read as "regression"
        traceback.print_exc(file=sys.stderr)
        print(f"gate: internal error ({type(e).__name__}: {e}); this is NOT a regression "
              f"verdict", file=sys.stderr)
        return 2
    problems = ratings.validate_report(report)
    if problems:
        print("gate: internal error, report violates the contract:\n  "
              + "\n  ".join(problems), file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(md)
    if args.summary_out:
        with open(args.summary_out, "a", encoding="utf-8") as fh:
            fh.write(md)
    print(f"wrote {out}")
    if not args.no_upload:
        res = maybe_upload(report, environ=environ, opener=opener)
        if res == "uploaded":
            print("uploaded the gate report to BreakPoint Web")
    if report["gate"]["regressed"]:
        print(f"\n{GATE_FAILED_MSG}:")
        for r in comparison["rows"]:
            if r["id"] in report["gate"]["new_failing_cells"]:
                print(f"  - cell {r['id']}: {r['baseline']['falls']}/{r['baseline']['n']} "
                      f"-> {r['candidate']['falls']}/{r['candidate']['n']} falls")
        deg = report["gate"].get("degradation")
        if deg and deg["fails_build"] and deg["significant"]:
            print(f"  - tracking: worse in {deg['up']} of {deg['n_pairs']} seed-pairs "
                  f"where neither fell, better in {deg['down']} (p = {_fmt_p(deg['p'])})")
        return 1
    print("\nCI gate passed: no evidence the candidate falls more than the baseline.")
    return 0


def main(argv=None) -> int:
    ap = add_args(argparse.ArgumentParser(
        description="BreakPoint CI gate: candidate vs baseline checkpoint on fixed conditions."))
    return gate(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
