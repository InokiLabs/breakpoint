"""Workstream 2a (T3): per-checkpoint failure-regression CI entrypoint.

Run this once per policy version (checkpoint). It produces the version's failure
families, DIFFS them against the previous stored version's atlas, prints a readable
new/resolved/worsened/unchanged report, saves the new version into the per-target
history store, and EXITS NON-ZERO when the CI gate trips (a new critical family
appeared, or an existing family worsened past a threshold) -- i.e. failure families
behave like failing unit tests.

Usage (Colab or local):

    python scripts/regression.py --target go1 --version ckpt-47

Family source (pick one; default is the last-produced failures.json):
    --families PATH   use an already-produced families/atlas JSON or a failures.json
                      (CPU-only; no GPU needed -- this is the CI-friendly default path).
    --run             run the full search+cluster pipeline for this checkpoint first
                      (Colab/GPU; delegates to scripts/run_pipeline.py).
    (default)         use <BP_OUT>/<target>/failures.json if it exists.

Baseline: --baseline VERSION, or (default) the most recent prior stored version. On the
very first run there is no baseline: everything is "new", the atlas is saved, and it
exits 0 with a "baseline established" message.

The diff/fingerprint/gate logic lives in breakpoint_mjx.regression (pure, CPU-tested).
This script is just the CLI + the (Colab-only) --run wiring.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breakpoint_mjx import regression  # noqa: E402
from breakpoint_mjx.config import (  # noqa: E402
    load_search_config,
    load_theta_space,
    merge_target_thresholds,
)
from breakpoint_mjx.targets import get_target  # noqa: E402
from breakpoint_mjx.paths import config_path  # noqa: E402


def _families_from_file(path, space, cfg, n_families):
    """Load families from either an atlas JSON (has "families") or a failures.json.

    A failures.json is the raw record list save_failures writes (ALL evaluated records,
    not just failures), so we threshold + cluster it exactly as run_pipeline does before
    diffing -- keeping the regression families identical to the report's families.

    build_families / load_failures are imported LAZILY (only the failures.json branch needs
    them) so a pure atlas-JSON CI run stays sklearn-free -- ``regression`` itself never
    imports scikit-learn, and neither does this path.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict) and "families" in payload:
        families, _ = regression.load_atlas(path)
        return families
    from breakpoint_mjx.clustering import build_families
    from breakpoint_mjx.search import load_failures

    records = load_failures(path)
    failures = [r for r in records if r.failure_score >= cfg.failure_threshold]
    if not failures:
        return []
    families, _ = build_families(failures, space, n_families)
    return families


def _run_pipeline_and_load(args, target, space, cfg, n_families, bp_out):
    """(Colab/GPU) run the real pipeline for this checkpoint, then load its families.

    Delegates to scripts/run_pipeline.py rather than duplicating search+cluster. The
    pipeline output is isolated under <bp_out>/runs/<version>/ so per-checkpoint searches
    never collide via the shared per-target failures.json (each checkpoint is a fresh
    search). The atlas history stays under the top-level bp_out.

    # VERIFY (Colab): needs the frozen policy checkpoint on Drive + a GPU (imports jax via
    # run_pipeline). Untested on the CPU-only dev box by design.
    """
    run_root = os.path.join(bp_out, "runs", regression._sanitize_version(args.version))
    os.makedirs(run_root, exist_ok=True)
    prev_bp_out = os.environ.get("BP_OUT")
    os.environ["BP_OUT"] = run_root
    try:
        import importlib

        run_pipeline = importlib.import_module("scripts.run_pipeline")
        old_argv = sys.argv
        try:
            sys.argv = ["run_pipeline.py", "--target", target.name]
            run_pipeline.main()
        finally:
            sys.argv = old_argv
    finally:
        if prev_bp_out is None:
            os.environ.pop("BP_OUT", None)
        else:
            os.environ["BP_OUT"] = prev_bp_out
    fpath = os.path.join(run_root, target.name, "failures.json")
    return _families_from_file(fpath, space, cfg, n_families)


def _resolve_new_families(args, target, space, cfg, n_families, out_dir, bp_out):
    if args.families:
        return _families_from_file(args.families, space, cfg, n_families)
    if args.run:
        return _run_pipeline_and_load(args, target, space, cfg, n_families, bp_out)
    fpath = os.path.join(out_dir, "failures.json")
    if not os.path.exists(fpath):
        raise SystemExit(
            f"no families source found. Pass --families PATH, or --run to search "
            f"(Colab/GPU), or produce {fpath} first."
        )
    return _families_from_file(fpath, space, cfg, n_families)


def build_parser():
    ap = argparse.ArgumentParser(
        description="BreakPoint per-checkpoint failure-regression CI gate."
    )
    ap.add_argument("--target", default="go1", help="registered target name (default: go1)")
    ap.add_argument("--version", required=True, help="policy version / checkpoint id")
    ap.add_argument(
        "--baseline",
        default=None,
        help="baseline version id (default: most recent prior stored version)",
    )
    ap.add_argument(
        "--families",
        default=None,
        help="path to an atlas JSON or a failures.json to diff (CPU; no GPU)",
    )
    ap.add_argument(
        "--run",
        action="store_true",
        help="run the full search+cluster pipeline for this checkpoint first (Colab/GPU)",
    )
    ap.add_argument("--bp-out", default=None, help="override BP_OUT root (default: env BP_OUT or 'out')")
    ap.add_argument("--n-families", type=int, default=None, help="KMeans family count (default: search.yaml)")
    ap.add_argument("--tol", type=float, default=regression.DEFAULT_TOL, help="fingerprint match tolerance")
    ap.add_argument(
        "--critical-severity",
        default=regression.DEFAULT_CRITICAL_SEVERITY,
        choices=["low", "medium", "high", "critical"],
        help="a NEW family at/above this severity fails the gate",
    )
    ap.add_argument("--critical-validity", type=float, default=regression.DEFAULT_CRITICAL_VALIDITY,
                    help="a NEW family with measured validity >= this fails the gate")
    ap.add_argument("--worsen-failure-delta", type=float, default=regression.DEFAULT_WORSEN_FAILURE_DELTA,
                    help="matched family whose mean_failure_score rose more than this fails the gate")
    ap.add_argument("--worsen-validity-delta", type=float, default=regression.DEFAULT_WORSEN_VALIDITY_DELTA,
                    help="matched family whose validity dropped more than this fails the gate")
    ap.add_argument("--no-gate", action="store_true", help="report + save only; always exit 0")
    ap.add_argument("--push", action="store_true",
                    help="also push this run + the regression verdict to BreakPoint Web "
                         "(needs BP_WEB_URL + BP_API_KEY and a families.json from run_pipeline)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    bp_out = args.bp_out or os.environ.get("BP_OUT", "out")
    target = get_target(args.target)
    space = load_theta_space(config_path(target.theta_space_path))
    cfg = merge_target_thresholds(load_search_config(config_path("search.yaml")), target)
    n_families = args.n_families or cfg.n_families

    out_dir = os.path.join(bp_out, target.name)
    history_dir = regression.atlas_history_dir(bp_out, target.name)

    new_families = _resolve_new_families(args, target, space, cfg, n_families, out_dir, bp_out)

    meta = {
        "version": str(args.version),
        "target": target.name,
        "n_families": n_families,
        "failure_threshold": cfg.failure_threshold,
        "n_families_found": len(new_families),
    }

    # Baseline resolution. Default prefers the most recent GREEN (gate_passed) version, not
    # merely the most recent one, so a regressed checkpoint does not poison the baseline and
    # silence a sustained regression after its first alarm (resolve_baseline falls back to
    # the most recent of any status when none is green -- e.g. right after the first run).
    if args.baseline is not None:
        baseline_version = args.baseline
        if not os.path.exists(regression.version_path(history_dir, baseline_version)):
            raise SystemExit(
                f"baseline version {baseline_version!r} not found under {history_dir}"
            )
    else:
        prev = regression.resolve_baseline(history_dir, exclude=args.version)
        baseline_version = prev["version"] if prev else None

    # First run: no baseline yet -> everything is "new"; establish it (green) and exit clean.
    if baseline_version is None:
        for w in regression.broken_search_warnings(new_families, []):
            print(f"WARNING: {w}")
        meta["gate_passed"] = True
        path = regression.save_version(history_dir, args.version, new_families, meta)
        print(
            f"No baseline found under {history_dir}.\n"
            f"Baseline ESTABLISHED for version {args.version!r} with "
            f"{len(new_families)} families (all new) -> {path}"
        )
        for f in new_families:
            fp = regression.family_fingerprint(f, space).key
            print(f"  [NEW] {f.name}  (severity {regression.family_severity(f)}, "
                  f"score {f.mean_failure_score:.3f})  fp={fp}")
        return 0

    old_families, _old_meta = regression.load_version(history_dir, baseline_version)
    diff = regression.diff_atlases(old_families, new_families, space, tol=args.tol)
    print(regression.format_diff(diff, baseline_version=baseline_version, version=args.version))

    # Broken-search guard: a silently-empty/collapsed atlas must not read green as "all
    # resolved". Warn loudly, and escalate an empty new atlas to a gate violation.
    warnings = regression.broken_search_warnings(new_families, old_families)
    for w in warnings:
        print(f"WARNING: {w}")

    exit_code, violations = regression.evaluate_gate(
        diff,
        critical_severity=args.critical_severity,
        critical_validity=args.critical_validity,
        worsen_failure_delta=args.worsen_failure_delta,
        worsen_validity_delta=args.worsen_validity_delta,
    )
    if len(new_families) == 0 and len(old_families) > 0:
        violations.append("BROKEN SEARCH: new atlas has 0 families (see warning above)")

    # Record the true gate outcome in this version's meta BEFORE returning, so a regressed
    # checkpoint is stored as gate_passed=False and the next run diffs against the last GREEN
    # baseline instead of this one. Every run is still saved for the record.
    gate_passed = not violations
    meta["gate_passed"] = gate_passed
    meta["gate_violations"] = violations
    path = regression.save_version(history_dir, args.version, new_families, meta)
    print(f"\nSaved version {args.version!r} ({len(new_families)} families, "
          f"gate_passed={gate_passed}) -> {path}")

    if violations:
        print("\nCI GATE FAILED (regression detected):")
        for v in violations:
            print(f"  - {v}")
    else:
        print("\nCI gate passed: no new critical or worsened-past-threshold families.")

    # Optional: push this run + the regression verdict to BreakPoint Web. The push NEVER
    # changes the gate exit code (a failed push warns; CI still fails locally on a violation).
    if args.push:
        _push_run_with_regression(target, bp_out, args.version, baseline_version,
                                  diff, gate_passed, violations)

    # --no-gate reports + records the true outcome but never fails the build.
    if args.no_gate:
        return 0
    return 1 if violations else 0


def _serializable_diff(diff: dict) -> dict:
    """Strip the FailureFamily object refs so the diff is JSON-safe (name/severity/deltas)."""
    def clean(e: dict) -> dict:
        return {
            k: v for k, v in e.items()
            if k not in ("family", "old", "new") and not hasattr(v, "member_ids")
        }
    return {k: [clean(e) for e in diff.get(k, [])]
            for k in ("new", "resolved", "worsened", "unchanged")}


def _push_run_with_regression(target, bp_out, version, baseline_version, diff, gate_passed, violations):
    """Write regression.json next to the run's families.json, then push via scripts/push_run."""
    import json
    out_dir = os.path.join(bp_out, target.name)
    fam_json = os.path.join(out_dir, "families.json")
    if not os.path.exists(fam_json):
        print(f"WARNING: --push skipped, no families.json in {out_dir} "
              f"(run scripts/run_pipeline.py --target {target.name} first)")
        return
    with open(os.path.join(out_dir, "regression.json"), "w") as fh:
        json.dump({
            "baseline_label": baseline_version,
            "diff": _serializable_diff(diff),
            "gate_passed": bool(gate_passed),
            "violations": list(violations),
        }, fh)
    web_url = os.environ.get("BP_WEB_URL", "").rstrip("/")
    key = os.environ.get("BP_API_KEY", "")
    if not web_url or not key:
        print("WARNING: --push set but BP_WEB_URL / BP_API_KEY missing; wrote regression.json only")
        return
    import uuid
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import push_run  # noqa: E402
    rc = push_run.push(out_dir, web_url, key, str(version), str(uuid.uuid4()))
    if rc != 0:
        print("WARNING: push failed (gate result is unaffected)")


if __name__ == "__main__":
    sys.exit(main())
