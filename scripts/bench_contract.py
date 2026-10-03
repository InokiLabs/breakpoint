"""Benchmark Behavioral Contract Identification on SYNTHETIC scrambles of a REAL policy.

The bundled Go1 policy (media/dl/frozen_go1_policy.pkl, a brax PPO checkpoint trained in
Playground) is re-expressed behind N seeded scrambled contracts. Each scramble:

* reorders the policy's obs terms (7 terms, any order),
* lays the joints out in an order from the leg-symmetry group (leg-major or Isaac Lab's
  joint-type-major, any leg order; every 4th scramble uses Isaac Lab's exact order),
* rescales 1-3 obs terms by x0.5 or x2 (never last_action: a policy sees its own raw
  output there),
* sets the action scale to x0.5, x1 or x2 of the target's.

``adapters.contract_id.fold_contract`` folds each scramble into the weights exactly
(first-layer rows, normalizer statistics, head columns), so the scrambled policy is a
real trained network that only walks when driven through its true contract. Four
methods then try to recover the contract under the same rollout budget:

  (a) identity   the target's native layout, i.e. what BreakPoint does without a contract
  (b) random     uniform hypotheses from the same space, same racing + acceptance rule
  (c) stageA     Stage A's top hypothesis, zero policy rollouts
  (d) bci        full BCI (Stage A + closed-loop re-estimation + CP racing)

A method RECOVERS a scramble when its chosen contract's 16-rollout nominal fall rate is
<= the true contract's + 1/16 (for random and bci the contract must also be ACCEPTED).
A FALSE ACCEPT is an accepted contract with fall rate >= 0.5. Everything runs on CPU.

    PYTHONPATH=. .venv-cpu/Scripts/python.exe scripts/bench_contract.py --n 40

Writes docs/contract_results.json (resumable: finished scrambles are skipped).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from breakpoint_mjx.adapters.contract_id import (  # noqa: E402
    Hypothesis,
    Racer,
    fold_contract,
    identify_contract,
    policy_core,
    reset_reference,
    stage_a,
    input_stats,
    symmetry_orders,
)
from breakpoint_mjx.adapters.importers import (  # noqa: E402
    ISAACLAB_QUADRUPED_ORDER,
    from_brax_pkl,
)
from breakpoint_mjx.adapters.obs_spec import GO1_INTERFACE as G  # noqa: E402

EVAL_N = 16
EVAL_SEED = 424242
LABEL = "SYNTHETIC scrambles of a real trained policy, CPU"
BARS = {
    "bci_recovery_min": 0.80,
    "false_accept_max": 0,
    "bci_minus_random_min_pp": 20.0,
    "identity_recovery_max": 0.10,
    "median_wallclock_max_s": 600.0,
}


def make_scramble(i, seed_base=5000):
    rng = np.random.default_rng(seed_base + i)
    terms = list(G.layout)
    order = [terms[k] for k in rng.permutation(len(terms))]
    orders = symmetry_orders(G)
    joint = (list(ISAACLAB_QUADRUPED_ORDER) if i % 4 == 0
             else list(orders[int(rng.integers(len(orders)))]))
    cand = [t for t in terms if t != "last_action"]
    k = int(rng.integers(1, 4))
    picked = [cand[j] for j in rng.choice(len(cand), size=k, replace=False)]
    scales = {t: float(rng.choice([0.5, 2.0])) for t in picked}
    ratio = float(rng.choice([0.5, 1.0, 2.0]))
    return Hypothesis(terms=tuple(order), scales=scales, order=tuple(joint),
                      action_scale=G.action_scale * ratio, origin="truth")


def _identity_contract():
    from breakpoint_mjx.adapters.obs_spec import build_contract

    return build_contract({"robot": "go1", "obs": list(G.layout)}, robot=G)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--budget", type=int, default=1024,
                    help="racing rollouts per identification (bci and random alike)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "contract_results.json"))
    ap.add_argument("--policy", default=str(ROOT / "media" / "dl" / "frozen_go1_policy.pkl"))
    ap.add_argument("--methods", default="identity,random,stageA,bci")
    ap.add_argument("--batch", type=int, default=32, help="rollouts per jitted call (RAM)")
    ap.add_argument("--seed-base", dest="seed_base", type=int, default=5000,
                    help="scramble i uses seed seed_base+i; development used 1000+0..7")
    args = ap.parse_args(argv)
    methods = args.methods.split(",")

    from breakpoint_mjx.targets import get_target

    out = Path(args.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    rows = {r["scramble"]: r for r in data.get("scrambles", [])}

    target = get_target("go1")
    env = target.load_env()
    native = from_brax_pkl(args.policy)
    apply_fn, w_native, meta = policy_core(native)
    t0 = time.monotonic()
    racer = Racer(env, target, apply_fn, horizon=150, batch=args.batch,
                  signature=meta["signature"])
    racer.race([_identity_contract()], n=1, seed=0, weights=w_native)
    compile_s = time.monotonic() - t0
    ref = reset_reference(env, n=256, seed=991)
    print(f"racer compiled in {compile_s:.0f}s", flush=True)

    ident = _identity_contract()
    for i in range(args.n):
        if i in rows and all(m in rows[i]["methods"] for m in methods):
            continue
        truth = make_scramble(i, args.seed_base)
        tc = truth.contract(G)
        pol = fold_contract(native, tc)
        _, w, pmeta = policy_core(pol)

        def evaluate(c):
            r = racer.race([c], n=EVAL_N, seed=EVAL_SEED, weights=w)[0]
            return {"falls": r["falls"], "n": r["n"], "success": r["success"],
                    "fall_rate": r["falls"] / r["n"]}

        row = rows.get(i) or {"scramble": i, "truth": truth.spec("go1"),
                              "true_eval": evaluate(tc), "methods": {}}
        bar = row["true_eval"]["fall_rate"] + 1.0 / EVAL_N
        print(f"\n[{i}] true fall {row['true_eval']['falls']}/{EVAL_N}; terms "
              f"{truth.terms}; scales {truth.scales}; action {truth.action_scale}", flush=True)
        for m in methods:
            if m in row["methods"]:
                continue
            t1 = time.monotonic()
            if m == "identity":
                ev = evaluate(ident)
                rec = {"accepted": None, "rollouts": 0, "eval": ev,
                       "recovered": ev["fall_rate"] <= bar + 1e-9}
            elif m == "stageA":
                pmu, psd, _ = input_stats(pmeta["in_dim"], pmeta["norm"], pmeta["first_w"])
                top = stage_a(G, pmeta["in_dim"], pmu, psd, ref, top_k=1)[0]
                ev = evaluate(top.contract(G))
                rec = {"accepted": None, "rollouts": 0, "eval": ev,
                       "structure_correct": top.key() == truth.key(),
                       "chosen": top.spec("go1"),
                       "recovered": ev["fall_rate"] <= bar + 1e-9}
            else:
                res = identify_contract(pol, target, env, budget=args.budget, racer=racer,
                                        reference=ref, method=m, seed=17 * i + 1)
                ev = evaluate(res.contract)
                acc = res.status == "accepted"
                rec = {"status": res.status, "accepted": acc, "rollouts": res.rollouts,
                       "eval": ev, "structure_correct": res.hypothesis.key() == truth.key(),
                       "chosen": res.hypothesis.spec("go1"),
                       "evidence": res.evidence, "log": res.log,
                       "recovered": bool(acc and ev["fall_rate"] <= bar + 1e-9),
                       "false_accept": bool(acc and ev["fall_rate"] >= 0.5)}
            rec["seconds"] = time.monotonic() - t1
            row["methods"][m] = rec
            print(f"  {m:8s} {rec.get('status', '-'):9s} recovered={rec['recovered']!s:5s} "
                  f"eval falls {rec['eval']['falls']}/{EVAL_N} rollouts {rec['rollouts']} "
                  f"({rec['seconds']:.0f}s)", flush=True)
        rows[i] = row
        _write(out, rows, args, compile_s)
    summary = _write(out, rows, args, compile_s)
    print(json.dumps(summary, indent=2))


def _write(out, rows, args, compile_s):
    scr = [rows[k] for k in sorted(rows)]
    summ = {}
    for m in ("identity", "random", "stageA", "bci"):
        rs = [r["methods"][m] for r in scr if m in r["methods"]]
        if not rs:
            continue
        s = {"n": len(rs), "recovered": sum(bool(r["recovered"]) for r in rs)}
        s["recovery_rate"] = s["recovered"] / len(rs)
        if m in ("random", "bci"):
            s["accepted"] = sum(bool(r["accepted"]) for r in rs)
            s["ambiguous"] = sum(r.get("status") == "ambiguous" for r in rs)
            s["failed"] = sum(r.get("status") == "failed" for r in rs)
            s["false_accepts"] = sum(bool(r.get("false_accept")) for r in rs)
            s["structure_correct"] = sum(bool(r.get("structure_correct")) for r in rs)
            s["median_rollouts"] = statistics.median(r["rollouts"] for r in rs)
            s["median_seconds"] = statistics.median(r["seconds"] for r in rs)
        if m == "stageA":
            s["structure_correct"] = sum(bool(r.get("structure_correct")) for r in rs)
        summ[m] = s
    verdict = {}
    if "bci" in summ:
        b = summ["bci"]
        verdict["bci_recovery"] = [b["recovery_rate"], b["recovery_rate"] >= BARS["bci_recovery_min"]]
        verdict["false_accepts"] = [b["false_accepts"], b["false_accepts"] <= BARS["false_accept_max"]]
        verdict["median_wallclock_s"] = [b["median_seconds"],
                                         b["median_seconds"] <= BARS["median_wallclock_max_s"]]
    if "bci" in summ and "random" in summ:
        d = 100 * (summ["bci"]["recovery_rate"] - summ["random"]["recovery_rate"])
        verdict["bci_minus_random_pp"] = [d, d >= BARS["bci_minus_random_min_pp"]]
    if "identity" in summ:
        r = summ["identity"]["recovery_rate"]
        verdict["identity_recovery"] = [r, r <= BARS["identity_recovery_max"]]
    doc = {
        "label": LABEL,
        "what": "Behavioral Contract Identification on seeded scrambles of the bundled "
                "Go1 brax policy (media/dl/frozen_go1_policy.pkl) folded exactly into the "
                "weights; nominal Go1JoystickFlatTerrain, horizon 150, CPU MJX.",
        "command": "PYTHONPATH=. .venv-cpu/Scripts/python.exe scripts/bench_contract.py "
                   f"--n {args.n} --budget {args.budget} --seed-base {args.seed_base}",
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "cpu_count": os.cpu_count()},
        "racer_compile_seconds": compile_s,
        "budget_rollouts": args.budget,
        "scramble_seed_base": args.seed_base,
        "development_note": "BCI was developed and tuned on scrambles with seed_base 1000, "
                            "i = 0..7. The reported run uses a seed_base never used "
                            "during development (default 5000).",
        "eval": {"rollouts": EVAL_N, "seed": EVAL_SEED, "horizon": 150},
        "recovery_rule": "chosen contract's 16-rollout fall rate <= true contract's + 1/16; "
                         "random and bci must also ACCEPT (identity and stageA have no "
                         "accept rule and are counted on the fall-rate bar alone)",
        "bars_preregistered": BARS,
        "summary": summ,
        "verdict": verdict,
        "scrambles": scr,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, default=_np))
    return {"summary": summ, "verdict": verdict}


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    main()
