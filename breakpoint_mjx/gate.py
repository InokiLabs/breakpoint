"""CI gate for robot policies: "pytest for robot policies".

On every new checkpoint, the team's OWN runner evaluates the candidate and the baseline
checkpoint on the SAME fixed conditions (battery cells) and the SAME seeds, compares them
cell by cell, and fails the build when the candidate falls in a condition the baseline
survived. Weights never leave the runner: the report holds per-cell fall counts and the
checkpoints' sha256 hashes, nothing else.

    breakpoint gate --robot go1 --baseline ckpt_old.pkl --candidate ckpt_new.pkl \\
        --budget smoke --out gate.json

Writes a ``kind: "gate"`` report per the shared contract (breakpoint.report/v1), prints a
markdown summary, and exits 1 on a regression (0 on pass, 2 on a usage error). When
BREAKPOINT_API_KEY and BREAKPOINT_PROJECT are set, the report (contract fields only) is
POSTed to BreakPoint Web; an upload error only warns and never changes the exit code.

Relation to scripts/regression.py: that gate diffs FAILURE FAMILIES found by a search
between two stored versions (fingerprint matching, severity/validity thresholds). This
gate compares FIXED CONDITIONS run on both checkpoints in the same job, so the diff is a
per-cell comparison rather than family matching. It keeps regression.py's CI contract:
the same "CI GATE FAILED (regression detected)" line and exit 1 on a regression, and a
push failure never changes the gate result.

Battery: when ``breakpoint_mjx/batteries/<battery>.yaml`` exists it is loaded (contract
format: cells, seeds, horizon, pass_rule). Until that directory lands (a parallel branch
adds the full battery), the small fixed condition set ``BUILTIN_BATTERY`` below is used
and the report names it honestly (``builtin-quadruped-gate-v0``), never as the full
battery. Swapping loaders later is the single assignment ``BATTERY_LOADER = ...``.

The comparison, report and markdown functions are pure (no JAX) and unit-tested on a
mocked evaluator; only ``make_sim_evaluator`` touches the simulator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "breakpoint.report/v1"
GATE_FAILED_MSG = "CI GATE FAILED (regression detected)"
DEFAULT_BATTERY = "legged-quadruped-v1"
DEFAULT_PASS_FALL_RATE = 1.0 / 8.0      # contract pass_rule: fall rate <= 1/8 across seeds
DEFAULT_API_URL = "https://breakpoint-inoki.vercel.app"
UPLOAD_TIMEOUT_S = 15
BATTERIES_DIR = Path(__file__).resolve().parent / "batteries"

# Budget presets. smoke fits a CPU-only GitHub-hosted runner (few seeds, short horizon);
# standard is the battery as declared (all its seeds, its horizon) and wants a GPU.
BUDGETS = {
    "smoke": {"n_seeds": 2, "horizon_cap": 150},
    "standard": {"n_seeds": None, "horizon_cap": None},
}

# Small fixed condition set used until breakpoint_mjx/batteries/ exists on this branch.
# Theta dims come from configs/go1/theta_space.yaml; values sit at or near the plausible
# band edges so a healthy policy is expected to survive most of them.
BUILTIN_BATTERY = {
    "battery": "builtin-quadruped-gate-v0",
    "version": 0,
    "robot_class": "legged-quadruped",
    "horizon": 500,
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7],
    "fall_rule": "trunk height above the floor (floor-relative) < 0.18 m",
    "cells": [
        {"id": "nominal", "theta": {}},
        {"id": "low-friction", "theta": {"friction_floor": 0.40}},
        {"id": "payload", "theta": {"torso_mass_add_kg": 2.0}},
        {"id": "lag-dropout", "theta": {"latency_steps": 2, "obs_dropout_prob": 0.2}},
        {"id": "sensor-noise", "theta": {"obs_noise_scale": 0.04, "obs_bias_scale": 0.04}},
        {"id": "push", "theta": {"push_mag": 1.0, "push_step_frac": 0.5}},
        {"id": "soft-gains", "theta": {"kp_scale": 0.85, "kd_scale": 0.85}},
        {"id": "slope", "theta": {"floor_slope_rad": 0.10}},
    ],
    "pass_rule": "a cell passes if fall rate <= 1/8 across seeds",
}


# --------------------------------------------------------------------------------------
# battery
# --------------------------------------------------------------------------------------

def _canonical(obj) -> bytes:
    """Contract canonical JSON: UTF-8, keys sorted, separators (",", ":")."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def load_battery(name: str = DEFAULT_BATTERY, batteries_dir: Path | None = None):
    """Return (spec dict, sha256 hex, note). Loads ``<batteries_dir>/<name>.yaml`` when it
    exists (sha256 of the file bytes, per contract); otherwise falls back to
    BUILTIN_BATTERY (sha256 of its canonical JSON) and says so in ``note``."""
    d = Path(batteries_dir) if batteries_dir is not None else BATTERIES_DIR
    path = d / f"{name}.yaml"
    if path.exists():
        import yaml

        raw = path.read_bytes()
        spec = yaml.safe_load(raw.decode("utf-8"))
        if not isinstance(spec, dict) or not spec.get("cells"):
            raise ValueError(f"battery {path} has no cells")
        return spec, hashlib.sha256(raw).hexdigest(), None
    spec = BUILTIN_BATTERY
    where = "breakpoint_mjx/batteries/" if batteries_dir is None else d.name + "/"
    note = (f"battery {name!r} not found under {where}; using the built-in "
            f"{spec['battery']} ({len(spec['cells'])} cells). The full battery lands "
            f"with breakpoint_mjx/batteries/.")
    return spec, hashlib.sha256(_canonical(spec)).hexdigest(), note


# The one line to change when ratings.py / batteries/ provides the shared loader.
BATTERY_LOADER = load_battery


def plan_run(spec: dict, budget: str = "standard"):
    """(seeds, horizon) for a budget preset applied to a battery spec."""
    if budget not in BUDGETS:
        raise ValueError(f"unknown budget {budget!r}; choose from {sorted(BUDGETS)}")
    b = BUDGETS[budget]
    seeds = [int(s) for s in spec.get("seeds") or [0]]
    if b["n_seeds"] is not None:
        seeds = seeds[: b["n_seeds"]]
    horizon = int(spec.get("horizon") or 500)
    if b["horizon_cap"] is not None:
        horizon = min(horizon, int(b["horizon_cap"]))
    return seeds, horizon


def parse_pass_fall_rate(spec: dict) -> float:
    """Read 'fall rate <= a/b' (or a decimal) from the battery's pass_rule; default 1/8."""
    import re

    rule = str(spec.get("pass_rule") or "")
    m = re.search(r"<=\s*([0-9.]+)\s*/\s*([0-9.]+)", rule)
    if m and float(m.group(2)) > 0:
        return float(m.group(1)) / float(m.group(2))
    m = re.search(r"<=\s*([0-9.]+)", rule)
    if m:
        return float(m.group(1))
    return DEFAULT_PASS_FALL_RATE


# --------------------------------------------------------------------------------------
# pure: cells, comparison, report, markdown
# --------------------------------------------------------------------------------------

def cell_result(cell_id: str, falls: int, n: int, pass_fall_rate: float) -> dict:
    """One contract cell row with an exact (Clopper-Pearson) 95% interval."""
    from breakpoint_mjx.coverage import clopper_pearson_interval

    falls, n = int(falls), int(n)
    rate = falls / n if n else 0.0
    lo, hi = clopper_pearson_interval(falls, n, alpha=0.05)
    return {"id": str(cell_id), "n": n, "falls": falls, "fall_rate": round(rate, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "passed": bool(n > 0 and rate <= pass_fall_rate + 1e-12)}


def compare_cells(baseline: list, candidate: list, tolerance: float = 0.0) -> dict:
    """Per-cell comparison. A cell REGRESSES when the baseline passed it, the candidate
    failed it, and the candidate's fall rate exceeds the baseline's by more than
    ``tolerance`` (a fall-rate margin; 0.0 means any newly failing cell regresses)."""
    b = {c["id"]: c for c in baseline}
    rows, new_failing = [], []
    for c in candidate:
        bc = b.get(c["id"])
        if bc is None:
            raise ValueError(f"cell {c['id']!r} missing from the baseline run")
        delta = float(c["fall_rate"]) - float(bc["fall_rate"])
        regressed = bool(bc["passed"] and not c["passed"] and delta > tolerance + 1e-12)
        if regressed:
            status = "REGRESSED"
            new_failing.append(c["id"])
        elif bc["passed"] and not c["passed"]:
            status = "within tolerance"
        elif not bc["passed"] and c["passed"]:
            status = "fixed"
        elif not c["passed"]:
            status = "fails (baseline too)"
        else:
            status = "pass"
        rows.append({"id": c["id"], "baseline": bc, "candidate": c,
                     "delta": round(delta, 4), "status": status})
    return {"rows": rows, "new_failing_cells": new_failing,
            "regressed": bool(new_failing)}


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spec_path(spec: str) -> str:
    """Strip an adapter prefix (onnx:/params:/brax:/...) to get the checkpoint file."""
    s = str(spec)
    head, sep, tail = s.partition(":")
    if sep and len(head) > 1 and head.lower() in ("onnx", "params", "npz", "weights", "brax"):
        return tail
    return s


def engine_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).parent),
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and len(sha) == 40 else None
    except Exception:
        return None


def summary_block(cells: list) -> dict:
    passed = sum(1 for c in cells if c["passed"])
    return {"cells_passed": passed, "cells_total": len(cells),
            "label": f"{passed}/{len(cells)} cells passed"}


def tolerance_label(tolerance: float) -> str:
    return f"fall_rate_delta<={tolerance:g}"


def build_report(*, robot, battery_id, battery_sha256, policy_name, policy_sha256,
                 baseline_sha256, candidate_cells, comparison, seeds, backend,
                 tolerance, engine_version, commit, date=None, source="local") -> dict:
    """The kind="gate" report, exactly the contract's fields. ``cells``/``summary``
    describe the CANDIDATE; the baseline enters through ``gate``."""
    return {
        "schema": SCHEMA,
        "kind": "gate",
        "battery": {"id": battery_id, "sha256": battery_sha256},
        "engine": {"version": engine_version, "commit": commit},
        "robot": robot,
        "policy": {"name": policy_name, "source": source, "sha256": policy_sha256},
        "run": {"date": date or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "backend": backend, "seeds": [int(s) for s in seeds], "runner": "self"},
        "cells": candidate_cells,
        "summary": summary_block(candidate_cells),
        "gate": {"baseline_sha256": baseline_sha256, "candidate_sha256": policy_sha256,
                 "new_failing_cells": list(comparison["new_failing_cells"]),
                 "regressed": bool(comparison["regressed"]),
                 "tolerance": tolerance_label(tolerance)},
        "attestation": None,
    }


CONTRACT_KEYS = ("schema", "kind", "battery", "engine", "robot", "policy", "run", "cells",
                 "summary", "gate", "attestation")


def contract_payload(report: dict) -> dict:
    """Only the contract's top-level fields (no weights, paths, or extras)."""
    return {k: report.get(k) for k in CONTRACT_KEYS}


def _fmt_cell(c: dict) -> str:
    return f"{c['falls']}/{c['n']}"


def to_markdown(report: dict, comparison: dict, *, baseline_name: str, horizon: int,
                budget: str, notes=()) -> str:
    g = report["gate"]
    verdict = ("FAILED: regression detected" if g["regressed"] else "PASSED")
    lines = [f"## BreakPoint gate: {verdict}", ""]
    if g["regressed"]:
        lines.append(f"The candidate falls in {len(g['new_failing_cells'])} condition(s) "
                     f"the baseline survived: " + ", ".join(f"`{c}`" for c in
                                                         g["new_failing_cells"]) + ".")
    else:
        lines.append("The candidate survives every condition the baseline survived.")
    lines += [
        "",
        f"- robot: `{report['robot']}`; battery: `{report['battery']['id']}` "
        f"(sha256 `{report['battery']['sha256'][:12]}`)",
        f"- baseline: `{baseline_name}` (sha256 `{g['baseline_sha256'][:12]}`)",
        f"- candidate: `{report['policy']['name']}` (sha256 `{g['candidate_sha256'][:12]}`)",
        f"- budget: `{budget}`; seeds {report['run']['seeds']}; horizon {horizon} steps; "
        f"backend `{report['run']['backend']}`; tolerance `{g['tolerance']}`",
        f"- candidate: {report['summary']['label']}",
        "",
        "| cell | baseline falls | candidate falls | candidate 95% CI | status |",
        "|---|---|---|---|---|",
    ]
    for r in comparison["rows"]:
        lo, hi = r["candidate"]["ci95"]
        status = f"**{r['status']}**" if r["status"] == "REGRESSED" else r["status"]
        lines.append(f"| `{r['id']}` | {_fmt_cell(r['baseline'])} | "
                     f"{_fmt_cell(r['candidate'])} | [{lo:.2f}, {hi:.2f}] | {status} |")
    lines.append("")
    for n in notes:
        lines.append(f"> note: {n}")
    if notes:
        lines.append("")
    lines.append("Simulation only. A cell passes when its fall rate is at most the "
                 "battery's pass rule; weights never left this runner.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# upload (optional; never fails the build)
# --------------------------------------------------------------------------------------

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

def cell_thetas(space, cells: list):
    """(C, D) theta matrix: nominal everywhere except each cell's overrides."""
    import numpy as np

    names = space.names()
    nom = np.asarray(space.nominal_vector(), dtype=np.float32)
    X = np.tile(nom[None, :], (len(cells), 1))
    for i, c in enumerate(cells):
        for k, v in (c.get("theta") or {}).items():
            if k not in names:
                raise ValueError(f"cell {c['id']!r}: theta dim {k!r} is not in this "
                                 f"robot's theta space ({', '.join(names)})")
            X[i, names.index(k)] = float(v)
    return X


def make_sim_evaluator(robot: str, opts=None, log=print):
    """Return (evaluate_fn, backend). ``evaluate_fn(spec, cells, seeds, horizon)`` returns
    {cell_id: falls} with n = len(seeds) per cell. Seed s runs every cell in one batch
    under PRNGKey(s), row i keyed fold_in(PRNGKey(s), i), so baseline and candidate see
    identical worlds and identical rollout keys."""
    from breakpoint_mjx.accel import tune_for_gpu

    tune_for_gpu()
    import dataclasses

    import jax
    import jax.numpy as jnp
    import numpy as np

    from breakpoint_mjx.check_policy import _load_env, load_policy
    from breakpoint_mjx.config import (load_search_config, load_theta_space,
                                       merge_target_thresholds)
    from breakpoint_mjx.evaluator import make_evaluator
    from breakpoint_mjx.paths import config_path
    from breakpoint_mjx.search import scores_from_trajectories
    from breakpoint_mjx.targets import get_target

    opts = opts or {}
    target = get_target(robot)
    cfg = merge_target_thresholds(load_search_config(config_path("search.yaml")), target)
    space = load_theta_space(config_path(target.theta_space_path))
    env = _load_env(target, opts.get("obs_spec"))
    consts = target.resolve_constants(env)
    backend = "mjx-gpu" if jax.default_backend() == "gpu" else "mjx-cpu"

    def evaluate_fn(spec, cells, seeds, horizon):
        policy_fn, _norm, _map, label = load_policy(target, env, spec, opts)
        log(f"gate: {label}: {len(cells)} cells x {len(seeds)} seeds x {horizon} steps "
            f"(first call compiles, ~1-3 min on CPU) ...")
        run_cfg = dataclasses.replace(cfg, horizon=int(horizon))
        evaluate = make_evaluator(env, policy_fn, space, int(horizon), target, consts)
        X = jnp.asarray(cell_thetas(space, cells))
        falls = np.zeros(len(cells), dtype=int)
        base_model = env._mjx_model
        try:
            for s in seeds:
                traj = evaluate(X, jax.random.PRNGKey(int(s)))
                _, _, comps = scores_from_trajectories(traj, run_cfg.thresholds,
                                                       run_cfg.horizon, target.objective)
                falls += np.array([int(float(c.get("fall", 0.0)) > 0) for c in comps])
        finally:
            env._mjx_model = base_model  # see check_policy.run_check: no leaked tracer
        return {c["id"]: int(f) for c, f in zip(cells, falls)}

    return evaluate_fn, backend


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------

def run_gate(*, robot, baseline, candidate, battery=DEFAULT_BATTERY, budget="standard",
             tolerance=0.0, evaluate_fn=None, backend=None, batteries_dir=None,
             candidate_name=None, baseline_name=None, log=print):
    """Evaluate both checkpoints on the same cells/seeds; return (report, comparison,
    markdown). ``evaluate_fn(spec, cells, seeds, horizon) -> {cell_id: falls}`` is
    injected in tests; the default builds the simulator evaluator."""
    from breakpoint_mjx import __version__

    if tolerance < 0:
        raise ValueError("--tolerance must be >= 0")
    for label, spec in (("baseline", baseline), ("candidate", candidate)):
        if not Path(spec_path(spec)).is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {spec_path(spec)}")
    spec_doc, battery_sha, note = BATTERY_LOADER(battery, batteries_dir)
    notes = [note] if note else []
    seeds, horizon = plan_run(spec_doc, budget)
    if budget != "standard":
        notes.append(f"budget {budget}: {len(seeds)} of {len(spec_doc.get('seeds') or [])} "
                     f"battery seeds, horizon {horizon} (battery declares "
                     f"{spec_doc.get('horizon')}); a smoke gate is a screen, not a rating")
    pass_rate = parse_pass_fall_rate(spec_doc)
    cells = spec_doc["cells"]
    if evaluate_fn is None:
        evaluate_fn, backend = make_sim_evaluator(robot, log=log)

    def run_one(spec):
        falls = evaluate_fn(spec, cells, seeds, horizon)
        return [cell_result(c["id"], falls[c["id"]], len(seeds), pass_rate) for c in cells]

    base_cells = run_one(baseline)
    cand_cells = run_one(candidate)
    comparison = compare_cells(base_cells, cand_cells, tolerance)
    b_nom = next((c for c in base_cells if c["id"] == "nominal"), None)
    if b_nom is not None and not b_nom["passed"]:
        notes.append("the BASELINE fails the nominal cell; the gate cannot detect "
                     "regressions in conditions it never survived (check the policy "
                     "with `breakpoint check-policy`)")
    cand_sha = file_sha256(spec_path(candidate))
    base_sha = file_sha256(spec_path(baseline))
    report = build_report(
        robot=robot, battery_id=spec_doc.get("battery", battery), battery_sha256=battery_sha,
        policy_name=candidate_name or Path(spec_path(candidate)).name,
        policy_sha256=cand_sha, baseline_sha256=base_sha, candidate_cells=cand_cells,
        comparison=comparison, seeds=seeds, backend=backend or "mjx-cpu",
        tolerance=tolerance, engine_version=__version__, commit=engine_commit())
    md = to_markdown(report, comparison,
                     baseline_name=baseline_name or Path(spec_path(baseline)).name,
                     horizon=horizon, budget=budget, notes=notes)
    return report, comparison, md


def add_args(ap):
    ap.add_argument("--robot", default="go1", help="registered target (default: go1)")
    ap.add_argument("--baseline", required=True, help="baseline checkpoint (the one in prod)")
    ap.add_argument("--candidate", required=True, help="candidate checkpoint under test")
    ap.add_argument("--battery", default=DEFAULT_BATTERY,
                    help=f"battery id under breakpoint_mjx/batteries/ (default: "
                         f"{DEFAULT_BATTERY}; built-in condition set if absent)")
    ap.add_argument("--budget", default="standard", choices=sorted(BUDGETS),
                    help="smoke: 2 seeds, horizon <= 150 (CPU runners); standard: the "
                         "battery as declared (GPU runner)")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="fall-rate margin: a newly failing cell regresses only if the "
                         "candidate's fall rate exceeds the baseline's by more than this")
    ap.add_argument("--out", default="gate.json", help="report path (default: gate.json)")
    ap.add_argument("--summary-out", default=None,
                    help="also write the markdown summary here (e.g. $GITHUB_STEP_SUMMARY; "
                         "appended)")
    ap.add_argument("--candidate-name", default=None, help="name in the report (default: file name)")
    ap.add_argument("--baseline-name", default=None, help="name in the summary (default: file name)")
    ap.add_argument("--no-upload", action="store_true",
                    help="never upload, even when BREAKPOINT_API_KEY/BREAKPOINT_PROJECT are set")
    return ap


def gate(args, evaluate_fn=None, backend=None, environ=None, opener=None) -> int:
    try:
        report, comparison, md = run_gate(
            robot=args.robot, baseline=args.baseline, candidate=args.candidate,
            battery=args.battery, budget=args.budget, tolerance=args.tolerance,
            evaluate_fn=evaluate_fn, backend=backend, candidate_name=args.candidate_name,
            baseline_name=args.baseline_name)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    if out.parent and not out.parent.exists():
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
        for c in report["gate"]["new_failing_cells"]:
            print(f"  - cell {c}: candidate falls where the baseline survived")
        return 1
    print("\nCI gate passed: no condition the baseline survived is failed by the candidate.")
    return 0


def main(argv=None) -> int:
    ap = add_args(argparse.ArgumentParser(
        description="BreakPoint CI gate: candidate vs baseline checkpoint on fixed conditions."))
    return gate(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
