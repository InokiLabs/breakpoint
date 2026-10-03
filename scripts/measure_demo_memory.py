"""Measure the peak memory and wall clock of a BreakPoint command (default: the CPU demo).

    python scripts/measure_demo_memory.py --json peak.json -- breakpoint demo --out d

Launches the command, samples the WHOLE process tree every ``--interval`` seconds with
psutil, and records the peak of the summed resident set (RSS) and of the summed private
commit (``memory_info().private`` on Windows, ``vms`` elsewhere), plus free system RAM
and commit at the start. Each child output line is echoed with its elapsed time, so
stage boundaries (``families (CMC)``, ``wrote ... report.html``) get timestamps too.
The numbers in stagecache.DEMO_PEAK_* and the README come from this script.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="write the summary here")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("give a command after --")
    import psutil

    from breakpoint_mjx.stagecache import free_memory

    mb = 1 << 20
    start_mem = free_memory()
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONUNBUFFERED="1"), text=True,
                            encoding="utf-8", errors="replace")
    peak = {"rss_mb": 0, "commit_mb": 0, "t_rss_s": 0.0, "t_commit_s": 0.0}
    stop = threading.Event()

    def sample():
        root = psutil.Process(proc.pid)
        while not stop.is_set():
            rss = com = 0
            try:
                procs = [root] + root.children(recursive=True)
            except psutil.Error:
                break
            for p in procs:
                try:
                    mi = p.memory_info()
                    rss += mi.rss
                    com += getattr(mi, "private", 0) or mi.vms
                except psutil.Error:
                    pass
            now = time.time() - t0
            if rss // mb > peak["rss_mb"]:
                peak.update(rss_mb=rss // mb, t_rss_s=round(now, 1))
            if com // mb > peak["commit_mb"]:
                peak.update(commit_mb=com // mb, t_commit_s=round(now, 1))
            stop.wait(a.interval)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    marks = []
    for line in proc.stdout:
        el = time.time() - t0
        sys.stdout.write(f"[{el:7.1f}s] {line}")
        sys.stdout.flush()
        low = line.lower()
        if any(k in low for k in ("families", "resuming from", "wrote", "rendered",
                                  "independent replay", "exited", "low-memory")):
            marks.append([round(el, 1), line.strip()[:160]])
    rc = proc.wait()
    stop.set()
    th.join(timeout=5)
    summary = {"cmd": cmd, "returncode": rc, "wall_s": round(time.time() - t0, 1),
               "peak_tree_rss_mb": peak["rss_mb"], "peak_tree_commit_mb": peak["commit_mb"],
               "t_peak_rss_s": peak["t_rss_s"], "t_peak_commit_s": peak["t_commit_s"],
               "interval_s": a.interval, "start_free": start_mem,
               "date": time.strftime("%Y-%m-%d %H:%M"), "marks": marks}
    print(json.dumps({k: v for k, v in summary.items() if k != "marks"}, indent=1))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(summary, fh, indent=1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
