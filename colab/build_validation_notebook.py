"""Generate colab/breakpoint_validation.ipynb: the one notebook that produces BreakPoint's
real-budget evidence on a Colab GPU.

Run with any Python 3 (stdlib only): `python colab/build_validation_notebook.py`.
The two measurement cells are NOT duplicated here: the notebook exec()s
colab/evidence_cell.py and colab/baseline_cell.py from the checked-out branch, so the
notebook and the repo cannot drift apart.
"""
import json
import os

BRANCH = "main"

CELLS = [
    ("markdown", """# BreakPoint validation run (Colab GPU)

Produces a **standard-budget run on a GPU** and an **equal-budget comparison against
uniform random sampling**.

**Before you start**
1. Runtime -> Change runtime type -> **A100** (L4 works, slower).
2. Only if you install from a private repo or fork: add a GitHub token as a Colab secret
   named **`GH_TOKEN`** (key icon in the left sidebar; read access to that repo) and turn on
   "Notebook access" for it. A public repo needs no token.
3. Run the cells top to bottom. Outputs go to your Google Drive under
   `MyDrive/breakpoint_mjx_out/`, so a disconnect does not lose them.

**Time**: expect roughly 1-2 hours in total on an A100. If a cell fails, keep its error
output; it says which stage broke.

**What to paste back**: the output of the last cell.
"""),

    ("code", f"""# 1) Bootstrap: install, clone the repo, mount Drive (once per session)
import os, subprocess, sys
os.environ["MUJOCO_GL"] = "egl"          # headless rendering

def sh(cmd):
    tok = os.environ.get("GH_TOKEN")
    print("$", cmd.replace(tok, "***") if tok else cmd)
    subprocess.run(cmd, shell=True, check=True)

# JAX with CUDA 12, pinned <0.9 (brax 0.14 still calls jax.device_put_replicated).
sh('pip -q install -U "jax[cuda12]<0.9"')

from google.colab import userdata
try:
    os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN") or ""
except Exception:
    os.environ["GH_TOKEN"] = ""          # no secret: fine for a public repo

REPO = "/content/breakpoint-mjx"
auth = ("x-access-token:" + os.environ["GH_TOKEN"] + "@") if os.environ["GH_TOKEN"] else ""
url = "https://" + auth + "github.com/InokiLabs/breakpoint.git"
if not os.path.exists(REPO):
    sh(f"git clone -q {{url}} InokiLabs/breakpoint")
os.chdir(REPO)
sh("git fetch -q origin {BRANCH}")
sh("git checkout -q {BRANCH}")
sh("git pull -q origin {BRANCH}")
sh("pip -q install -r requirements.txt")
sh("pip -q install -e .")

from google.colab import drive
drive.mount("/content/drive")
os.environ["BP_OUT"] = "/content/drive/MyDrive/breakpoint_mjx_out"
os.makedirs(os.environ["BP_OUT"], exist_ok=True)

import jax
print("JAX devices:", jax.devices())
assert any(d.platform == "gpu" for d in jax.devices()), "No GPU: set the runtime to A100/L4"
print("commit:", subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip())
"""),

    ("code", """# 2) Doctor + which policy will be tested
import os, subprocess
subprocess.run(["python", "-m", "breakpoint_mjx.cli", "doctor"], check=False)
trained = os.path.join(os.environ["BP_OUT"], "go1", "go1_policy.pkl")
print("\\npolicy under test:",
      trained if os.path.exists(trained)
      else "bundled demo policy (no trained go1 checkpoint on Drive; weaker, say so in any write-up)")
"""),

    ("markdown", """## 3) Evidence run (standard budget)
Fresh search -> failure families -> Certified Minimal Cause -> each certified minimal
counterexample re-run on 8 seeds in C MuJoCo with Playground's full-collision Go1 scene.
Read out per family: confirmed / not confirmed / unconfirmed, the k/n badge with its
Clopper-Pearson lower bound, and the torso height above the floor in the replay."""),

    ("code", """# 3) Evidence run
import os
os.chdir("/content/breakpoint-mjx")
exec(open("colab/evidence_cell.py", encoding="utf-8").read())
evidence_result = res
"""),

    ("markdown", """## 4) Equal-budget baseline
Directed search vs uniform random (and vs the training DR distribution when one is
declared) at the SAME number of rollouts, 3 seeds, scored on metrics the search does not
optimize, plus the C MuJoCo replay confirmation rate of the closest failures per arm."""),

    ("code", """# 4) Equal-budget baseline
import os
os.chdir("/content/breakpoint-mjx")
exec(open("colab/baseline_cell.py", encoding="utf-8").read())
"""),

    ("code", """# 5) PASTE THE OUTPUT OF THIS CELL BACK
import json, os, subprocess
base = os.environ["BP_OUT"]
print("commit:", subprocess.run(["git", "-C", "/content/breakpoint-mjx", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip())
print("\\n=== EVIDENCE ===")
try:
    print(json.dumps(evidence_result, indent=1)[:6000])
except NameError:
    print("evidence cell did not finish")
print("\\n=== EQUAL-BUDGET BASELINE ===")
p = os.path.join(base, "equal_budget", "go1", "equal_budget.md")
print(open(p, encoding="utf-8").read() if os.path.exists(p) else "baseline cell did not finish")
"""),
]


def build():
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [],
    }
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "breakpoint_validation.ipynb")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return out


if __name__ == "__main__":
    print(build())
