"""LLM-guided search seed: ask Claude (or OpenRouter) for the parameter
configuration most likely to break the policy, used to initialize CMA-ES.
Provider-flexible (mirrors explain.py); returns None (-> center start) when no key.
Commentary/heuristic only -- the GPU search still does the real work.

OPT-IN. A present API key is not consent to a network call: LLM seeding (and the LLM
failure explanations in explain.py, which share ``llm_enabled``) run only with BP_LLM=1.
Even then, a smoke-budget demo run (BP_DEMO=1 and BP_BUDGET=smoke) stays offline."""
from __future__ import annotations
import json
import os
import re

_DIM_DESC = {
    "friction_floor": "ground/floor friction coefficient; lower = more slippery",
    "torso_mass_add_kg": "extra payload mass bolted to the trunk, in kg",
    "com_offset_x_m": "center-of-mass shift forward(+)/backward(-), meters",
    "com_offset_y_m": "center-of-mass shift left(+)/right(-), meters",
    "kp_scale": "actuator position-gain (stiffness) multiplier; 1.0 = nominal",
    "kd_scale": "actuator damping-gain multiplier; 1.0 = nominal",
    "com_offset_z_m": "center-of-mass shift up(+)/down(-), meters",
    "joint_frictionloss_scale": "multiplier on all leg-joint friction loss; 1.0 = nominal",
    "joint_armature_scale": "multiplier on all leg-joint armature/reflected inertia; 1.0 = nominal",
    "obs_noise_scale": "stddev of Gaussian noise added to observations (sensor noise)",
    "latency_steps": "actuator/control delay in control steps (0 = none)",
}


_TRUTHY = ("1", "true", "yes", "on")
_NOTICE_SHOWN = False


def llm_enabled(environ=None):
    """True only when the user opted in with BP_LLM=1 and this is not a smoke demo."""
    env = os.environ if environ is None else environ
    if str(env.get("BP_LLM", "")).strip().lower() not in _TRUTHY:
        return False
    demo = str(env.get("BP_DEMO", "")).strip().lower() in _TRUTHY
    smoke = str(env.get("BP_BUDGET", "")).strip().lower() == "smoke"
    return not (demo and smoke)


def _off_notice():
    global _NOTICE_SHOWN
    if not _NOTICE_SHOWN:
        _NOTICE_SHOWN = True
        print("LLM seeding is off by default (set BP_LLM=1 to enable); no network calls.",
              flush=True)


def _prompt(space, robot_name=None):
    lines = []
    for name, d in space.dims.items():
        desc = _DIM_DESC.get(name, name)
        lines.append(
            f"- {name}: {desc}. bounds [{d.bounds[0]}, {d.bounds[1]}], "
            f"nominal {d.nominal}, realistic band {d.plausible}"
        )
    names = list(space.dims.keys())
    robot = robot_name or "quadruped"
    return (
        f"You are helping find where a {robot} locomotion policy fails. "
        "The policy tracks velocity commands and was trained with domain randomization, "
        "so it tolerates mild single-axis perturbations but breaks under adversarial "
        "COMBINATIONS, especially ones still within realistic hardware bounds (high "
        "sim-to-real transfer).\n\n"
        "Adversarial parameter space:\n" + "\n".join(lines) + "\n\n"
        "Propose ONE configuration you predict is most likely to make the policy fall or "
        "lose velocity tracking while staying as physically plausible as possible (prefer "
        "values near the edges of each realistic band over absurd extremes). "
        f"Output ONLY a JSON object mapping each of these keys to a number: {names}. No prose."
    )


def _extract_json(text):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    return json.loads(m.group(0)) if m else None


def _call_anthropic(prompt):
    import anthropic
    client = anthropic.Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
    resp = client.messages.create(
        model=model, max_tokens=512, messages=[{"role": "user", "content": prompt}]
    )
    if resp.stop_reason == "refusal":
        return None
    return _extract_json("".join(b.text for b in resp.content if b.type == "text"))


def _call_openai(prompt):
    import httpx
    key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    r = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    r.raise_for_status()
    return _extract_json(r.json()["choices"][0]["message"]["content"])


def _call_openrouter(prompt):
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
    return _extract_json(r.json()["choices"][0]["message"]["content"])


def propose_seed(space, robot_name=None):
    """Return {dim_name: value clipped to bounds} or None if unavailable or not opted in
    (``llm_enabled``); in the latter case no provider is contacted."""
    if not llm_enabled():
        _off_notice()
        return None
    prompt = _prompt(space, robot_name)
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            raw = _call_anthropic(prompt)
        elif os.environ.get("OPENAI_API_KEY"):
            raw = _call_openai(prompt)
        elif os.environ.get("OPENROUTER_API_KEY"):
            raw = _call_openrouter(prompt)
        else:
            return None
        if not isinstance(raw, dict):
            return None
        seed = {}
        for name, d in space.dims.items():
            v = raw.get(name, d.nominal)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = d.nominal
            seed[name] = float(min(max(v, d.bounds[0]), d.bounds[1]))  # clip to bounds
        return seed
    except Exception as e:
        print("llm seed skipped:", e)
        return None
