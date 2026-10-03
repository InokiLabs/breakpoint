# BreakPoint skeptic-grade evidence on GPU -- UNVERIFIED (never run; written on a Windows
# machine with no JAX GPU). Paste into a Colab A100 cell AFTER the bootstrap
# (cell 1 of colab/breakpoint_validation.ipynb); it runs on whatever revision the bootstrap checked out.
#
# What it measures on go1 (trained checkpoint from the bootstrap), standard budget:
#   1. a fresh search -> families -> Certified Minimal Cause (K=8), then
#   2. the independent replay (default on): each certified minimal counterexample re-run on
#      8 seeds in C MuJoCo on Playground's FULL-collision Go1 scene + a same-seed nominal
#      twin, both rendered and clipped 1 s after the fall.
# Read out: per family, status (confirmed / not_confirmed_outside_mjx / unconfirmed), the badge
# k/n + CP lower bound, the lowest torso height above the floor in the replay (must stay
# >= -0.02 m) vs in the MJX search rollout, the replay wall time, and the nominal check.
# The C replay runs on the Colab CPU (host-side mj_step), so its seconds are CPU seconds.
import json, os, subprocess, time

REPO = "/content/breakpoint-mjx"
os.chdir(REPO)
subprocess.run(["pip", "-q", "install", "-e", "."], check=True)          # pick up new modules

BASE = os.environ.get("BP_OUT", "/content/drive/MyDrive/breakpoint_mjx_out")
ROOT = os.path.join(BASE, "evidence")
out_dir = os.path.join(ROOT, "go1")
os.makedirs(out_dir, exist_ok=True)
p = os.path.join(out_dir, "failures.json")
if os.path.exists(p):
    os.remove(p)                                       # fresh search for this run

# The pipeline looks for <BP_OUT>/go1/go1_policy.pkl, and BP_OUT is the evidence folder
# here, which is empty on a fresh Drive. Pass the policy explicitly: the trained checkpoint
# from the bootstrap if it exists, else the bundled demo policy (weaker; say so in any write-up).
ckpt = os.path.join(BASE, "go1", "go1_policy.pkl")
if not os.path.exists(ckpt):
    ckpt = os.path.join(REPO, "breakpoint_mjx", "assets", "demo", "go1_policy.pkl")
print("policy:", ckpt)

env = dict(os.environ, BP_OUT=ROOT, BP_BUDGET="standard", BP_EMBED_VIDEO="0", BP_POLICY=ckpt,
           BP_REPLAY="1", BP_REPLAY_K="8", BP_REPLAY_MAXMULT="4")   # pool like CMC
t0 = time.time()
r = subprocess.run(["python", "scripts/run_pipeline.py", "--target", "go1"], env=env,
                   capture_output=True, text=True)
print(r.stdout[-6000:])
if r.returncode != 0:
    print(r.stderr[-4000:])
    raise SystemExit("pipeline failed")
fj = json.load(open(os.path.join(out_dir, "families.json")))
ev = fj.get("evidence") or {}
rp = ev.get("replay") or {}
res = {
    "wall_s": round(time.time() - t0, 1),
    "replay_status": rp.get("status"), "replay_reason": rp.get("reason"),
    "replay_seconds": rp.get("seconds"),
    "collision_model": (rp.get("collision_model") or {}).get("source"),
    "n_reproduced": rp.get("n_reproduced"), "n_demoted": rp.get("n_demoted"),
    "nominal_check": fj.get("nominal_check"),
    "surrogate_fit": fj.get("surrogate_fit"),
    "envelope": (fj.get("envelope") or {}).get("summary_long"),
    "families": [
        {"name": f["name"], "status": f.get("status"),
         "badge": (f"{f['replay']['verdict']} {f['replay']['k']}/{f['replay']['n']} "
                   f"(CP lower {f['replay']['cp_lo']})"
                   if f.get("replay") else None),
         "replay_min_floor_m": (f.get("replay") or {}).get("min_floor_height"),
         "mjx_min_floor_m": f.get("mjx_min_floor_height"),
         "videos": f.get("evidence_videos")}
        for f in fj["families"]
    ],
}
print(json.dumps(res, indent=2))
bad = [f for f in res["families"]
       if f["replay_min_floor_m"] is not None and f["replay_min_floor_m"] < -0.02]
print("torso below floor in the full-collision replay:", bad or "none")
