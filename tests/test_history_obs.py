"""Stacked observation history: STOPPED, and refused loudly.

Spec item (2) asked for an obs-history ring buffer carried in the policy state inside the
evaluator's vmap, IF the adapter policy_fn interface allows state. It does not: the shared
rollout (breakpoint_mjx/rollout.py, which evaluator.py vmaps) calls a stateless
``policy_fn(obs, key) -> (action, extras)`` and its ``lax.scan`` carry is
``(env_state, latency_buffer, key)``. The build of (2) was stopped there (evaluator.py and
rollout.py were not edited); these tests pin that a history-stacked contract is refused
with that reason instead of being silently fed a single frame.
"""
import inspect

import pytest

from breakpoint_mjx.adapters.importers import from_isaaclab_cfg
from breakpoint_mjx.adapters.obs_spec import HISTORY_UNSUPPORTED, build_contract


def test_history_contract_is_refused_with_the_reason():
    spec = {"robot": "go1", "obs": ["base_ang_vel", "projected_gravity"],
            "history": {"length": 3, "order": "newest_first",
                        "terms": ["base_ang_vel", "projected_gravity"]}}
    with pytest.raises(ValueError) as e:
        build_contract(spec)
    assert "history" in str(e.value) and "STATELESS" in str(e.value)


def test_history_length_one_is_a_no_op():
    c = build_contract({"robot": "go1", "obs": ["base_ang_vel"], "history": {"length": 1}})
    assert c.obs_dim == 3


def test_isaaclab_history_length_is_fatal():
    res = from_isaaclab_cfg({"observations": {"policy": {
        "history_length": 3,
        "base_ang_vel": {"func": "isaaclab.envs.mdp.observations:base_ang_vel"}}}})
    assert [r["field"] for r in res.fatal] == ["observations.policy.history_length"]


def test_the_reason_still_holds():
    """If rollout_one ever grows a policy-state carry, this fails and (2) can be built."""
    jax = pytest.importorskip("jax")  # noqa: F841 - rollout imports jax
    from breakpoint_mjx import rollout

    src = inspect.getsource(rollout.rollout_one)
    assert "act, _ = policy_fn(obs, nk_policy)" in src
    assert "(st, buf, key), out" in src
    assert "rollout.py" in HISTORY_UNSUPPORTED
