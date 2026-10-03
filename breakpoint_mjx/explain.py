"""Optional plain-language commentary on a failure.

Prefers the Anthropic API (Claude) via the official `anthropic` SDK when
ANTHROPIC_API_KEY is set; falls back to OpenRouter's free tier when only
OPENROUTER_API_KEY is set; skips cleanly (returns None) when neither is set.
Commentary only, never the source of truth. OPT-IN: runs only with BP_LLM=1
(llm_seed.llm_enabled); an API key alone never triggers a network call.
"""
from __future__ import annotations
import json
import os


def _prompt(theta_named, signals, transfer, robot_name=None):
    robot = robot_name or "robot"
    return (
        f"A {robot} control policy failed under these perturbations:\n"
        f"{json.dumps(theta_named, indent=2)}\n"
        f"Failure signals: {json.dumps(signals)}\n"
        f"Heuristic transfer-likelihood (0-1): {transfer:.2f}\n"
        "In 2 sentences, explain the likely physical failure mechanism and whether this "
        "regime is plausible on real hardware."
    )


def _explain_anthropic(prompt):
    """Claude via the official SDK. Model defaults to claude-opus-4-8 (override
    with ANTHROPIC_MODEL, e.g. claude-haiku-4-5 for a cheaper run). This is a
    short generation, so no extended thinking is needed."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
    resp = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text or None


def _explain_openrouter(prompt):
    import httpx

    key = os.environ["OPENROUTER_API_KEY"]
    model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def explain_failure(theta_named, signals, transfer, robot_name=None):
    from breakpoint_mjx.llm_seed import llm_enabled

    if not llm_enabled():  # opt-in (BP_LLM=1): an API key alone is not consent
        return None
    prompt = _prompt(theta_named, signals, transfer, robot_name)
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            return _explain_anthropic(prompt)
        if os.environ.get("OPENROUTER_API_KEY"):
            return _explain_openrouter(prompt)
        return None
    except Exception as e:
        print("explain skipped:", e)
        return None
