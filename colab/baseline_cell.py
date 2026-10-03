# BreakPoint equal-budget baseline on GPU -- UNVERIFIED (never run; written on a Windows
# machine with no JAX GPU). Paste into a Colab A100 cell AFTER the bootstrap
# (cell 1 of colab/breakpoint_validation.ipynb); it runs on whatever revision the bootstrap checked out.
#
# What it measures on go1 (trained checkpoint from the bootstrap), standard preset
# (horizon 300, popsize 128, 4 restarts -> 512 rollouts per batched call):
#   three arms at the SAME budget B per seed: uniform random over the theta box, the
#   engine's multi-restart CMA-ES (restarts*popsize*generations == B exactly), and the
#   declared training-DR distribution if BP_RISK_DIST names one (skipped otherwise),
#   3 seeds, scored on metrics the search does not optimize (failures, closest failure
#   to nominal in box units, rule-only in-band failures, modes), plus the C-MuJoCo
#   full-collision replay confirmation rate of the 4 closest failures per arm/seed.
# The C replay runs on the Colab CPU (host-side mj_step); its seconds are CPU seconds.
import os, subprocess, time

REPO = "/content/breakpoint-mjx"
os.chdir(REPO)
subprocess.run(["pip", "-q", "install", "-e", "."], check=True)          # pick up new modules

BASE = os.environ.get("BP_OUT", "/content/drive/MyDrive/breakpoint_mjx_out")
out_dir = os.path.join(BASE, "equal_budget", "go1")
os.makedirs(out_dir, exist_ok=True)

# The bootstrap trains go1 into <BP_OUT>/go1/<checkpoint>. Point --policy at that pickle;
# fall back to the bundled demo policy (weaker; say so in any write-up) if it is missing.
ckpt = os.path.join(BASE, "go1", "go1_policy.pkl")
policy_args = ["--policy", ckpt] if os.path.exists(ckpt) else []
print("policy:", ckpt if policy_args else "bundled demo policy (no trained checkpoint found)")

B = 5120                          # 4 restarts x 128 popsize x 10 generations (standard)
env = dict(os.environ, BP_BASELINE_REPLAY="1", BP_REPLAY_K="8")
t0 = time.time()
r = subprocess.run(["python", "-m", "breakpoint_mjx.equal_budget", "--target", "go1",
                    "--budget", str(B), "--seeds", "3", "--preset", "standard",
                    "--replay-k", "4", "--label", "Colab A100 standard",
                    "--out", out_dir, *policy_args],
                   env=env, capture_output=True, text=True)
print(r.stdout[-8000:])
if r.returncode != 0:
    print(r.stderr[-4000:])
    raise SystemExit("equal-budget baseline failed")
print(f"wall {time.time() - t0:.0f} s")
print(open(os.path.join(out_dir, "equal_budget.md"), encoding="utf-8").read())
