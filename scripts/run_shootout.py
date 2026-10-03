"""One-command search-strategy shootout: cmaes vs active vs llm, then the benchmark table.

Runs the full pipeline once per search mode against the SAME frozen checkpoint (each in
its own sandbox dir so failures.json never cross-contaminate), then scores all runs with
scripts/benchmark_search.py on identical per-eval metrics. Resumable: a mode whose
failures.json already exists is skipped unless --force.

The llm mode REQUIRES an API key (ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY);
without one it silently degrades to uniform random, which would fake a "LLM does nothing"
ablation. This script refuses to run llm keyless unless --allow-degraded.

Usage (Colab, after the bootstrap cell of the multi-policy notebook):
  python scripts/run_shootout.py --target go1
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breakpoint_mjx.targets import get_target  # noqa: E402

_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="go1")
    ap.add_argument("--modes", default="cmaes,multi,active,llm,agentic")
    ap.add_argument("--force", action="store_true", help="re-run modes that already have results")
    ap.add_argument("--allow-degraded", action="store_true",
                    help="let the llm mode run without an API key (NOT a real ablation)")
    args = ap.parse_args(argv)

    base_out = os.environ.get("BP_OUT")
    if not base_out:
        sys.exit("STOP: BP_OUT is not set. Run the notebook bootstrap cell first.")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    if ({"llm", "agentic"} & set(modes)) and not any(os.environ.get(k) for k in _KEYS):
        if args.allow_degraded:
            print("WARNING: no API key -- llm mode will degrade to random. "
                  "The llm column will NOT measure the LLM.")
        else:
            sys.exit("STOP: llm/agentic modes need ANTHROPIC_API_KEY (or OPENAI/OPENROUTER) set, "
                     "otherwise it degrades to random and the ablation is meaningless. "
                     "Set the key or pass --allow-degraded.")

    tgt = get_target(args.target)
    src_ckpt = os.path.join(base_out, args.target, tgt.checkpoint_name)
    if not os.path.exists(src_ckpt):
        sys.exit(f"STOP: no checkpoint at {src_ckpt}. Train {args.target} first "
                 "(cell 5 of the multi-policy notebook).")

    run_args = []
    for mode in modes:
        sdir = os.path.join(base_out, f"shootout_{mode}", args.target)
        failures = os.path.join(sdir, "failures.json")
        run_args += ["--run", f"{mode}={failures}"]
        if os.path.exists(failures) and not args.force:
            print(f"[shootout {mode}] already done ({failures}); skipping (--force to redo)")
            continue
        os.makedirs(sdir, exist_ok=True)
        shutil.copy(src_ckpt, os.path.join(sdir, tgt.checkpoint_name))
        env = dict(os.environ)
        env["BP_OUT"] = os.path.join(base_out, f"shootout_{mode}")
        env["BP_SEARCH"] = mode
        env.pop("BP_VALIDATE", None)  # keep the shootout lean
        env.pop("BP_MARGINS", None)
        print(f"\n======== shootout: {mode} ========", flush=True)
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--target", args.target],
            env=env, cwd=str(ROOT),
        )
        print(f"[shootout {mode}] {round(time.time() - t0, 1)}s "
              f"({'ok' if r.returncode == 0 else f'EXIT {r.returncode}'})")
        if r.returncode != 0:
            sys.exit(f"STOP: pipeline failed for mode {mode}; not benchmarking a partial set.")

    print("\n======== benchmark (per-eval, equal-budget metrics) ========", flush=True)
    out_json = os.path.join(base_out, f"shootout_benchmark_{args.target}.json")
    from benchmark_search import main as bench_main
    print("<<<SHOOTOUT RESULTS>>>")
    bench_main(run_args + ["--target", args.target, "--json", out_json])
    print("<<<END SHOOTOUT RESULTS>>>")
    print("saved:", out_json)


if __name__ == "__main__":
    main()
