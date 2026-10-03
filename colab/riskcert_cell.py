# BreakPoint operational-risk validation on GPU -- UNVERIFIED (never run; written on a
# Windows machine with no JAX GPU). Paste into a Colab A100 cell AFTER the bootstrap
# (cell 1 of colab/breakpoint_validation.ipynb); it runs on whatever revision the bootstrap checked out.
#
# What it measures, per target:
#   1. a 16x128 multi-start CMA-ES search (BP_SEARCH=multi), then
#   2. BP_RISK: sequential doubly-robust IS + betting CS, 512-rollout batches, up to
#      8,192 fresh rollouts, stopping early at +/-0.005, alpha = 0.3, operational
#      p = uniform on each ThetaDim's plausible band, and
#   3. a brute-force plain-MC reference on 65,536 p-draws (BP_RISK_MC_N) as ground truth.
# Read out: does the anytime CI cover the brute-force rate, how wide is it, how many
# rollouts did it use, and how does that compare with plain MC's own CP interval at the
# same N. The CPU analysis predicts NO width win unless N * P_fail >> 1/alpha, so a
# "no gain" outcome here is the expected result, not a bug.
import json, os, subprocess

REPO = "/content/breakpoint-mjx"
os.chdir(REPO)

BASE = os.environ.get("BP_OUT", "/content/drive/MyDrive/breakpoint_mjx_out")
ROOT = os.path.join(BASE, "riskcert")
TARGETS = ["go1", "barkour"]          # rate other checkpoints via BP_POLICY
results = {}
for t in TARGETS:
    env = dict(os.environ, BP_OUT=ROOT, BP_SEARCH="multi", BP_RESTARTS="16",
               BP_RISK="1", BP_RISK_N="8192", BP_RISK_BATCH="512", BP_RISK_TOL="0.005",
               BP_RISK_ALPHA="0.3", BP_RISK_MC_N="65536", BP_EMBED_VIDEO="0")
    out_dir = os.path.join(ROOT, t)
    fj = os.path.join(out_dir, "failures.json")
    if os.path.exists(fj):
        os.remove(fj)                  # force a fresh search so the proposal is this run's
    r = subprocess.run(["python", "scripts/run_pipeline.py", "--target", t], env=env,
                       capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        continue
    fam = json.load(open(os.path.join(out_dir, "families.json")))
    o = fam.get("operational_risk")
    if not o:
        print(t, "no operational_risk block (see 'skipped' line above)")
        continue
    ref = o.get("plain_mc_reference", {})
    results[t] = {
        "estimate": o["estimate"], "ci": [o["lo"], o["hi"]], "n_rollouts": o["n_rollouts"],
        "stopped": o["stopped"], "ess": o["ess"], "w_max_observed": o["w_max_observed"],
        "plain_mc_on_defensive_draws": [o["plain_mc_baseline"]["lo"],
                                        o["plain_mc_baseline"]["hi"],
                                        o["plain_mc_baseline"]["n"]],
        "reference_rate_65536": ref.get("rate"), "reference_ci": [ref.get("lo"), ref.get("hi")],
        "ci_covers_reference": ref.get("ci_covers_reference_rate"),
        "search_selected_rate": fam["stats"]["failure_rate"],
        "envelope_certificate": (fam.get("envelope") or {}).get("simulator_certificate"),
    }
print("<<<RISKCERT RESULT (UNVERIFIED until this prints on an A100)>>>")
print(json.dumps(results, indent=1))
