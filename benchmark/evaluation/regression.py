"""Experiment C: policy regression testing.

Two batteries, kept separate:

* **Fixed battery**: the shipped ``legged-quadruped-v1`` conditions and seeds, judged by
  the release gate exactly as CI runs it (``python -m breakpoint_mjx.gate``). The
  acceptance criterion is fixed in BENCHMARK_PLAN.md: the gate's statistical decision.
* **Adversarial battery**: validated failure conditions discovered in Experiment A,
  evaluated on both checkpoints with paired seeds. Reported separately; it never feeds the
  gate verdict (its conditions were selected because the baseline fails them).

A real checkpoint pair is required for a real comparison. Without one, only calibration
controls run: the identical pair (should pass) and explicitly SYNTHETIC weight-noise
candidates (not real policies; they test the gate's sensitivity, nothing more).
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ADVERSARIAL_KEY_BASE = 5_000_000


def make_synthetic_candidate(src: Path, dst: Path, sigma: float, seed: int) -> dict:
    """Write a SYNTHETIC degraded checkpoint: Gaussian noise of ``sigma`` x each policy
    weight tensor's own standard deviation, added to the policy network only (element 1 of
    the brax (normalizer, policy, value) tuple). Seeded and recorded; never a real policy."""
    import jax

    params = pickle.load(open(src, "rb"))
    if not (isinstance(params, (tuple, list)) and len(params) >= 2):
        raise ValueError("expected a brax (normalizer, policy, value) params tuple")
    rng = np.random.default_rng([int(seed), int(round(sigma * 1000))])

    def noisy(x):
        a = np.asarray(x)
        sd = float(a.std()) if a.size > 1 else 0.0
        return (a + sigma * sd * rng.standard_normal(a.shape)).astype(a.dtype)

    pol = jax.tree_util.tree_map(noisy, params[1])
    out = (params[0], pol, *params[2:])
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        pickle.dump(out, f)
    return {"path": str(dst), "sigma": sigma, "seed": seed, "kind": "synthetic"}


def run_gate(robot, baseline, candidate, out_json, budget="standard", alpha=0.05,
             escalate_to=None, log=print) -> dict:
    """Run the release gate as CI does; return its report plus the exit code."""
    cmd = [sys.executable, "-m", "breakpoint_mjx.gate", "--robot", robot,
           "--baseline", str(baseline), "--candidate", str(candidate),
           "--budget", budget, "--decision", "stat", "--alpha", str(alpha),
           "--tolerance", "0", "--out", str(out_json), "--no-upload"]
    if escalate_to is not None:
        cmd += ["--escalate-to", str(int(escalate_to))]
    log("[regression] " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    rep = json.loads(Path(out_json).read_text()) if Path(out_json).exists() else None
    return {"exit_code": p.returncode, "report": rep,
            "stdout_tail": p.stdout[-4000:], "stderr_tail": p.stderr[-2000:]}


def adversarial_battery(recorder_base, recorder_cand, thetas, n_seeds, batch):
    """Paired evaluation of both checkpoints on the same conditions and the SAME rollout
    keys. -> per-condition {"base": [0/1...], "cand": [0/1...]} (valid falls)."""
    import jax

    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    rows = np.repeat(thetas, n_seeds, axis=0)
    total = rows.shape[0]
    pad = (-total) % batch
    if pad:
        rows = np.concatenate([rows, np.repeat(rows[-1:], pad, axis=0)])
    for rec in (recorder_base, recorder_cand):
        rec.reset()
        for b, s in enumerate(range(0, rows.shape[0], batch)):
            key = jax.random.fold_in(jax.random.PRNGKey(ADVERSARIAL_KEY_BASE), b)
            rec(rows[s:s + batch].astype(np.float32), key)
    fb = recorder_base.table()["failure"][:total]
    fc = recorder_cand.table()["failure"][:total]
    out = []
    for c in range(thetas.shape[0]):
        sl = slice(c * n_seeds, (c + 1) * n_seeds)
        out.append({"base": fb[sl].astype(int).tolist(), "cand": fc[sl].astype(int).tolist()})
    return out


def decide_adversarial(per_condition, alpha=0.05):
    """The gate's own statistics on the adversarial battery (reported, not gating)."""
    from breakpoint_mjx.gate_stats import decide

    pairs = []
    for i, c in enumerate(per_condition):
        b = {"falls": int(sum(c["base"])), "n": len(c["base"]), "outcomes": c["base"]}
        k = {"falls": int(sum(c["cand"])), "n": len(c["cand"]), "outcomes": c["cand"]}
        pairs.append((f"adv-{i}", b, k))
    return decide(pairs, alpha=alpha, tolerance=0.0)
