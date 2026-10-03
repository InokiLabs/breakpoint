# BreakPoint Crash Ratings v1 on GPU -- UNVERIFIED (never run; written on a Windows machine
# with no JAX GPU). Paste into a Colab A100 cell AFTER the bootstrap
# (cell 1 of colab/breakpoint_validation.ipynb); it runs on whatever revision the bootstrap checked out.
#
# What it runs: the FULL legged-quadruped-v1 battery (15 cells x 8 seeds x 500 steps, no
# --budget smoke) through `breakpoint rate`, on:
#   1. the bundled demo Go1 policy (breakpoint_mjx/assets/demo/go1_policy.pkl), if this
#      install ships it;
#   2. the Go1 checkpoint the bootstrap trained on your Drive (<BP_OUT>/go1/go1_policy.pkl),
#      if it exists. That is YOUR OWN training run, not a third-party policy.
# Third-party checkpoints: none included. No public Go1 checkpoint with a clear licence was
# found (2026-10-03): MuJoCo Playground publishes training notebooks, not weights;
# unitree_rl_gym ships only g1 / h1 / h1_2 humanoid policies (checked 2026-09-27, see
# scripts/fetch_thirdparty.py); Isaac Lab's pretrained Go2 checkpoint has no licence file
# next to it (terms unverified) and is a Go2, not a Go1, so it is left out of public ratings.
#
# Each report is printed in full and saved under <BP_OUT>/ratings/. Expect run.backend
# "mjx-gpu" and run.budget "full"; anything else means this is not a real rating.
import json, os, subprocess, time

REPO = "/content/breakpoint-mjx"
os.chdir(REPO)
subprocess.run(["pip", "-q", "install", "-e", "."], check=True)          # pick up new modules

BASE = os.environ.get("BP_OUT", "/content/drive/MyDrive/breakpoint_mjx_out")
OUT = os.path.join(BASE, "ratings")
os.makedirs(OUT, exist_ok=True)

policies = []
demo = os.path.join(REPO, "breakpoint_mjx", "assets", "demo", "go1_policy.pkl")
if os.path.exists(demo):
    policies.append(("breakpoint-demo-go1", demo, "bundled with breakpoint-mjx"))
own = os.path.join(BASE, "go1", "go1_policy.pkl")
if os.path.exists(own):
    policies.append(("bootstrap-trained-go1", own, "local"))
else:
    print("no bootstrap-trained checkpoint at", own)
if not policies:
    raise SystemExit("no policy to rate: train one first (python scripts/train.py --target go1)")

for name, ckpt, source in policies:
    out = os.path.join(OUT, f"legged-quadruped-v1--{name}.json")
    t0 = time.time()
    r = subprocess.run(["breakpoint", "rate", "--robot", "go1", "--policy", ckpt,
                        "--battery", "legged-quadruped-v1", "--policy-name", name,
                        "--policy-source", source, "--out", out],
                       capture_output=True, text=True)
    print(f"===== {name}  ({time.time() - t0:.0f}s, exit {r.returncode})")
    print(r.stdout[-6000:])
    if r.returncode != 0:
        print(r.stderr[-4000:])
        continue
    rep = json.load(open(out))
    assert rep["run"]["budget"] == "full", "not a full rating"
    if rep["run"]["backend"] != "mjx-gpu":
        print("WARNING: backend is", rep["run"]["backend"], "- JAX did not see the GPU")
    print(json.dumps(rep, indent=2))
