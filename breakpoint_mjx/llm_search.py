"""True LLM-guided adversarial search (Phase 8).

A closed loop: the LLM proposes parameter configurations likely to break the
policy, the GPU evaluator scores the whole proposal set at once, the best
results feed back into the next prompt, and the LLM refines. The LLM is the
search operator; the GPU provides massively-parallel fitness. Provider-flexible
(ANTHROPIC_API_KEY, then OPENAI_API_KEY, then OPENROUTER_API_KEY); degrades to
uniform random per round when no API key is set."""
from __future__ import annotations
import json
import os
import re

import numpy as np

from breakpoint_mjx.llm_seed import _DIM_DESC
from breakpoint_mjx.plausibility import gate
from breakpoint_mjx.search import _records, scores_from_trajectories


def _complete(prompt, max_tokens=4096):
    """Provider-flexible single completion -> raw text, or None."""
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic()
            model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
            resp = client.messages.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if resp.stop_reason == "refusal":
                return None
            return "".join(b.text for b in resp.content if b.type == "text")
        if os.environ.get("OPENAI_API_KEY"):
            import httpx
            key = os.environ["OPENAI_API_KEY"]
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            r = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        if os.environ.get("OPENROUTER_API_KEY"):
            import httpx
            key = os.environ["OPENROUTER_API_KEY"]
            model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
            r = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        return None
    except Exception as e:
        print("llm call skipped:", e)
        return None


def _space_desc(space):
    return "\n".join(
        f"- {name}: {_DIM_DESC.get(name, name)}. bounds [{d.bounds[0]}, {d.bounds[1]}], "
        f"nominal {d.nominal}, realistic band {d.plausible}"
        for name, d in space.dims.items()
    )


def _parse_configs(text, space):
    """Pure: extract config objects from `text`, clip each value to bounds, fill
    missing keys with nominal. Tries the full JSON array first, then salvages
    individual {...} objects (tolerates truncated output / markdown fences).
    Returns a list of dicts (possibly empty)."""
    if not text:
        return []
    objs = []
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                objs = [x for x in arr if isinstance(x, dict)]
        except Exception:
            objs = []
    if not objs:  # salvage complete flat {...} objects even if the array was truncated
        for block in re.findall(r"\{[^{}]*\}", text, re.DOTALL):
            try:
                d = json.loads(block)
                if isinstance(d, dict):
                    objs.append(d)
            except Exception:
                continue
    out = []
    for item in objs:
        cfg = {}
        for name, d in space.dims.items():
            v = item.get(name, d.nominal)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = d.nominal
            cfg[name] = float(min(max(v, d.bounds[0]), d.bounds[1]))
        out.append(cfg)
    return out


def _propose(space, leaderboard, n_proposals, robot_name=None):
    names = list(space.dims.keys())
    robot = robot_name or "quadruped"
    if leaderboard:
        lb = "\n".join(
            f"  failure_score={r['fs']:.2f} transfer={r['tr']:.2f} :: "
            + ", ".join(f"{k}={v:.3g}" for k, v in r["theta"].items())
            for r in leaderboard
        )
        history = ("Best failures found so far (maximize BOTH failure_score and transfer):\n"
                   f"{lb}\n\n")
    else:
        history = "No results yet; propose a diverse first batch.\n\n"
    prompt = (
        f"You are running an adversarial search for configurations that break a {robot} "
        "locomotion policy (trained with domain randomization; robust to mild single-axis "
        "perturbations, breaks under realistic COMBINATIONS). Maximize failure_score while "
        "keeping transfer high (values within each realistic band).\n\n"
        f"Parameter space:\n{_space_desc(space)}\n\n"
        f"{history}"
        f"Propose {n_proposals} NEW, DIVERSE configurations to try next. Refine around what "
        "worked but keep exploring untried regions. "
        f"Output ONLY a JSON array of {n_proposals} objects, each mapping every key in "
        f"{names} to a number. No prose."
    )
    text = _complete(prompt)
    cfgs = _parse_configs(text, space)
    if text and not cfgs:  # got a completion but couldn't parse it -- surface why
        print("  (LLM proposal parse failed; first 160 chars:", repr(text[:160]), ")")
    elif not text:
        print("  (no LLM completion -- check ANTHROPIC_API_KEY / OPENAI_API_KEY / "
              "OPENROUTER_API_KEY)")
    return cfgs


def gated_proposals(space, proposals, horizon=None, gate_opts=None, label="gate"):
    """Run raw LLM proposals through the plausibility gate and PRINT what it did.

    Clipping to bounds (all ``_parse_configs`` does) keeps a proposal inside the box but
    says nothing about whether the corner of the box it landed in is a machine anyone
    could build. This is the one place the two searches enforce that, so the policy is
    identical in both and there is exactly one line of log to read.

    A SILENT GATE IS A BAD GATE: the summary always prints, so an operator watching a run
    sees immediately when the model has started emitting junk instead of failure worlds.

    Returns the list of configs worth simulating (kept + repaired). It may be EMPTY --
    that is fine and must stay fine: ``_build_batch`` pads a short proposal set with
    uniform-random rows, so the evaluation budget for the round is unchanged whatever the
    gate decides. Shrinking the batch instead would silently change what a run measures.
    """
    res = gate(proposals, space, horizon=horizon, **(gate_opts or {}))
    print(f"  {label}: {res.summary()[6:]}", flush=True)
    return res.accepted


def _to_vec(space, cfg):
    return np.array([cfg[n] for n in space.names()], dtype=float)


def _build_batch(space, proposals, n, rng, jitter_frac=0.12):
    """Expand proposals into exactly n theta via per-proposal Gaussian jitter,
    padding with uniform-random rows. Pure (numpy)."""
    lo = np.array(space.lower()); hi = np.array(space.upper()); span = hi - lo
    rows = []
    if proposals:
        per = max(1, n // len(proposals))
        for cfg in proposals:
            base = _to_vec(space, cfg)
            rows.append(base)
            for _ in range(per - 1):
                rows.append(np.clip(base + rng.normal(size=len(base)) * jitter_frac * span, lo, hi))
    while len(rows) < n:
        rows.append(lo + rng.random(len(lo)) * span)
    return np.array(rows[:n], dtype=float)


def llm_guided_search(evaluate, space, cfg, key, rounds=5, n_proposals=12, robot_name=None,
                      objective=None, gate_opts=None):
    """Closed-loop LLM search. Returns list[FailureRecord] across all rounds.

    T7: ``objective`` (the ACTIVE target's Objective, legged by default) is threaded into
    scoring so an LLM-guided run works for a manipulation target too.

    ``gate_opts``: keyword overrides for ``plausibility.gate`` (e.g.
    ``{"min_transfer": 0.4}``, or ``{"repair_enabled": False}`` to reject rather than
    nudge). Every proposal passes the gate before ``_build_batch`` so simulation budget is
    not spent on worlds a customer would dismiss as broken physics."""
    import jax
    import jax.numpy as jnp
    N = cfg.popsize
    rng = np.random.default_rng(int(cfg.seed))
    leaderboard, all_records = [], []
    for r in range(rounds):
        proposals = _propose(space, leaderboard, n_proposals, robot_name)
        n_raw = len(proposals)
        proposals = gated_proposals(space, proposals, horizon=cfg.horizon,
                                    gate_opts=gate_opts)
        batch = _build_batch(space, proposals, N, rng)
        key, ek = jax.random.split(key)
        traj = evaluate(jnp.asarray(batch), ek)
        scores, sigs, comps = scores_from_trajectories(traj, cfg.thresholds, cfg.horizon, objective)
        recs = _records(space, batch, scores, sigs, comps, cfg.seed, f"llm-r{r}", len(all_records), ek)
        all_records += recs
        for rec in recs:
            leaderboard.append({"fs": rec.failure_score, "tr": rec.transfer_score or 0.0,
                                "theta": {k: round(v, 3) for k, v in rec.theta_named.items()}})
        leaderboard.sort(key=lambda x: x["fs"] * x["tr"], reverse=True)
        leaderboard = leaderboard[:12]
        best_w = leaderboard[0]["fs"] * leaderboard[0]["tr"] if leaderboard else 0.0
        print(f"llm round {r}: proposals={n_raw}->{len(proposals)} "
              f"max_score={scores.max():.3f} best_weighted={best_w:.3f}", flush=True)
    return all_records
