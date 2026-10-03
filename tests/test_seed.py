"""Pure tests for theta_to_cmaes_mean (no JAX, no LLM).

Verifies the inverse-squash helper against the real configs/go1/theta_space.yaml.
The squash in cmaes_search is: theta = lo + sigmoid(x) * (hi - lo)
The inverse is:                 x    = logit((theta - lo) / (hi - lo))
"""
from pathlib import Path
import numpy as np
import pytest

from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.search import theta_to_cmaes_mean
from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


@pytest.fixture(scope="module")
def lo_hi(space):
    lo = np.array(space.lower())
    hi = np.array(space.upper())
    return lo, hi


def recover(x0, lo, hi):
    """Apply the CMA-ES squash forward: theta = lo + sigmoid(x0) * (hi - lo)."""
    return lo + 1.0 / (1.0 + np.exp(-x0)) * (hi - lo)


# ---------------------------------------------------------------------------
# Test 1: all-nominal seed -> recovered theta is approximately the nominal vector
# ---------------------------------------------------------------------------
def test_nominal_seed_roundtrips(space, lo_hi):
    lo, hi = lo_hi
    nominal = {name: d.nominal for name, d in space.dims.items()}
    x0 = theta_to_cmaes_mean(space, nominal)
    recovered = recover(x0, lo, hi)
    expected = np.array([d.nominal for d in space.dims.values()])
    np.testing.assert_allclose(recovered, expected, atol=1e-2,
                               err_msg="Nominal seed did not recover nominal vector")


# ---------------------------------------------------------------------------
# Test 2: extreme values map correctly -- friction at lower bound, latency near upper
# ---------------------------------------------------------------------------
def test_extreme_seed_maps_to_bounds(space, lo_hi):
    lo, hi = lo_hi
    names = space.names()
    friction_idx = names.index("friction_floor")
    latency_idx = names.index("latency_steps")

    friction_lo = space.dims["friction_floor"].bounds[0]   # 0.15 (lower bound)
    latency_near_hi = space.dims["latency_steps"].bounds[1] - 0.1  # 3.9 (near upper)

    seed = {name: d.nominal for name, d in space.dims.items()}
    seed["friction_floor"] = friction_lo
    seed["latency_steps"] = latency_near_hi

    x0 = theta_to_cmaes_mean(space, seed)
    recovered = recover(x0, lo, hi)

    # friction at lower bound: recovered should be very close to the lower bound
    assert recovered[friction_idx] < (friction_lo + 0.05), (
        f"friction_floor expected near {friction_lo}, got {recovered[friction_idx]:.4f}"
    )
    # latency near upper bound: recovered should be above the midpoint of its range
    latency_mid = (space.dims["latency_steps"].bounds[0] + space.dims["latency_steps"].bounds[1]) / 2
    assert recovered[latency_idx] > latency_mid, (
        f"latency_steps expected > midpoint {latency_mid:.2f}, got {recovered[latency_idx]:.4f}"
    )
    # monotonicity: both seeds are more extreme than nominal -> x0 should be more extreme
    nominal_x0 = theta_to_cmaes_mean(space, {name: d.nominal for name, d in space.dims.items()})
    assert x0[friction_idx] < nominal_x0[friction_idx], (
        "friction at lower bound should yield a more negative x0 than nominal"
    )
    assert x0[latency_idx] > nominal_x0[latency_idx], (
        "latency near upper bound should yield a more positive x0 than nominal"
    )


# ---------------------------------------------------------------------------
# Test 3: missing key falls back to nominal
# ---------------------------------------------------------------------------
def test_missing_key_falls_back_to_nominal(space, lo_hi):
    lo, hi = lo_hi
    names = space.names()

    # Provide seed with only ONE key present; all others should fall back to nominal
    partial_seed = {"friction_floor": 0.40}
    x0 = theta_to_cmaes_mean(space, partial_seed)
    recovered = recover(x0, lo, hi)
    expected_nominal = np.array([d.nominal for d in space.dims.values()])

    for i, name in enumerate(names):
        if name == "friction_floor":
            continue  # explicitly set; skip
        np.testing.assert_allclose(
            recovered[i], expected_nominal[i], atol=1e-2,
            err_msg=f"Missing key '{name}' did not fall back to nominal"
        )


# ---------------------------------------------------------------------------
# LLM seeding is OPT-IN: an API key alone must never cause a network call
# ---------------------------------------------------------------------------
def _no_network(monkeypatch):
    """Make every provider path and raw HTTP client explode if touched."""
    import sys
    import types

    from breakpoint_mjx import explain, llm_seed

    calls = []

    def boom(*a, **k):
        calls.append((a, k))
        raise AssertionError("network call attempted")

    for mod, names in ((llm_seed, ("_call_anthropic", "_call_openai", "_call_openrouter")),
                       (explain, ("_explain_anthropic", "_explain_openrouter"))):
        for n in names:
            monkeypatch.setattr(mod, n, boom)
    fake_httpx = types.SimpleNamespace(post=boom, get=boom, Client=boom)
    fake_anthropic = types.SimpleNamespace(Anthropic=boom)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    return calls


@pytest.mark.parametrize("extra", [{}, {"BP_BUDGET": "smoke"},
                                   {"BP_LLM": "1", "BP_DEMO": "1", "BP_BUDGET": "smoke"}])
def test_llm_seed_and_explain_offline_without_opt_in(space, monkeypatch, capsys, extra):
    from breakpoint_mjx import llm_seed
    from breakpoint_mjx.explain import explain_failure

    calls = _no_network(monkeypatch)
    for k in ("BP_LLM", "BP_DEMO", "BP_BUDGET"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    for k, v in extra.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(llm_seed, "_NOTICE_SHOWN", False)
    assert llm_seed.propose_seed(space, robot_name="go1") is None
    assert explain_failure({"friction_floor": 0.3}, {}, 0.5, robot_name="go1") is None
    assert calls == []
    assert "LLM seeding is off by default" in capsys.readouterr().out


def test_llm_seed_runs_when_opted_in(space, monkeypatch):
    from breakpoint_mjx import llm_seed

    monkeypatch.setenv("BP_LLM", "1")
    monkeypatch.delenv("BP_DEMO", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(llm_seed, "_call_anthropic", lambda prompt: {"friction_floor": -5.0})
    seed = llm_seed.propose_seed(space)
    assert seed["friction_floor"] == space.dims["friction_floor"].bounds[0]   # clipped


def test_llm_enabled_matrix():
    from breakpoint_mjx.llm_seed import llm_enabled

    assert llm_enabled({}) is False
    assert llm_enabled({"ANTHROPIC_API_KEY": "x"}) is False
    assert llm_enabled({"BP_LLM": "1"}) is True
    assert llm_enabled({"BP_LLM": "1", "BP_BUDGET": "smoke"}) is True
    assert llm_enabled({"BP_LLM": "1", "BP_BUDGET": "smoke", "BP_DEMO": "1"}) is False
