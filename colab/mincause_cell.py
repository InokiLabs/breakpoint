# BreakPoint Certified Minimal Cause (CMC) on GPU -- UNVERIFIED (never run; written on a
# Windows machine with no JAX GPU). Paste into a Colab A100 cell AFTER the bootstrap
# (cell 1 of colab/breakpoint_validation.ipynb); it runs on whatever revision the bootstrap checked out.
#
# What it measures on go1 (trained checkpoint from the bootstrap):
#   1. a fresh search with CMC OFF (BP_MINCAUSE=0): the old clustering names, then
#   2. the SAME failures.json re-read with CMC ON (default, K=8): certified minimal causes,
#      necessity, min distance from nominal, merges, rollouts spent.
# Read out: per family, old name vs CMC name, |S|, necessary, distance; median |S|; whether
# every clause direction agrees with the displayed representative values.
import json, os, subprocess, time

REPO = "/content/breakpoint-mjx"
os.chdir(REPO)

BASE = os.environ.get("BP_OUT", "/content/drive/MyDrive/breakpoint_mjx_out")
ROOT = os.path.join(BASE, "mincause")
out_dir = os.path.join(ROOT, "go1")
os.makedirs(out_dir, exist_ok=True)
for fn in ("failures.json",):
    p = os.path.join(out_dir, fn)
    if os.path.exists(p):
        os.remove(p)                                   # fresh search for this run


def run(extra):
    env = dict(os.environ, BP_OUT=ROOT, BP_EMBED_VIDEO="0", **extra)
    t0 = time.time()
    r = subprocess.run(["python", "scripts/run_pipeline.py", "--target", "go1"], env=env,
                       capture_output=True, text=True)
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise SystemExit("pipeline failed")
    return json.load(open(os.path.join(out_dir, "families.json"))), time.time() - t0


off, t_off = run({"BP_MINCAUSE": "0"})
on, t_on = run({"BP_MINCAUSE": "1", "BP_MINCAUSE_K": "8"})   # reuses failures.json
from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.paths import config_path
_space = load_theta_space(config_path("go1/theta_space.yaml"))
nominal = dict(zip(_space.names(), _space.nominal_vector()))
sizes, agree = [], True
for f in on["families"]:
    for c in f.get("minimal_cause") or []:
        shown = f["representative_theta"][c["dim"]]
        side = "high" if shown > nominal[c["dim"]] else "low"
        vside = "high" if c["value"] > nominal[c["dim"]] else "low"
        agree &= side == c["direction"] == vside
    if f.get("minimal_cause") is not None:
        sizes.append(len(f["minimal_cause"]))
res = {
    "names_off": [f["name"] for f in off["families"]],
    "names_on": [f["name"] for f in on["families"]],
    "margins_on": [f["margin"] for f in on["families"]],
    "cause_sizes": sizes,
    "directions_agree_with_displayed": bool(agree),
    "median_cause_size": sorted(sizes)[len(sizes) // 2] if sizes else None,
    "necessary": [f.get("cause_necessary") for f in on["families"]],
    "min_distance": [f.get("min_distance") for f in on["families"]],
    "wallclock_s": {"off": round(t_off), "on_reusing_search": round(t_on)},
}
print("<<<MINCAUSE RESULT (UNVERIFIED until this prints on an A100)>>>")
print(json.dumps(res, indent=1))
