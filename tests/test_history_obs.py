"""Stacked observation history: supported through the stateful-policy carry.

This used to be refused: the shared rollout called a stateless ``policy_fn(obs, key)`` and
its scan carry had nowhere to keep a per-env ring buffer. ``rollout_one`` now carries a
policy state for any policy exposing ``init_carry`` (adapters/stateful.py), so a contract
``history:`` block and an Isaac Lab ``history_length`` are honored. These tests pin the
contract and importer side; tests/test_stateful.py covers the math and the rollout.
"""
import inspect

import pytest

from breakpoint_mjx.adapters.importers import from_isaaclab_cfg
from breakpoint_mjx.adapters.obs_spec import build_contract

BAV = {"func": "isaaclab.envs.mdp.observations:base_ang_vel"}
PG = {"func": "isaaclab.envs.mdp.observations:projected_gravity"}


def test_history_contract_is_accepted():
    spec = {"robot": "go1", "obs": ["base_ang_vel", "projected_gravity"],
            "history": {"length": 3, "order": "newest_first", "layout": "frame"}}
    c = build_contract(spec)
    assert c.history == {"length": 3, "layout": "frame", "fill": "first",
                         "order": "newest_first"}
    assert c.obs_dim == 6                    # ONE frame; the policy input is 3 x 6


def test_history_length_one_is_a_no_op():
    c = build_contract({"robot": "go1", "obs": ["base_ang_vel"], "history": {"length": 1}})
    assert c.obs_dim == 3 and c.history is None


def test_isaaclab_history_length_becomes_a_term_layout_history():
    res = from_isaaclab_cfg({"observations": {"policy": {
        "history_length": 3, "base_ang_vel": BAV, "projected_gravity": PG}}})
    assert res.fatal == []
    assert res.contract["history"] == {"length": 3, "layout": "term", "fill": "first"}
    c = build_contract(res.contract)
    assert c.history["length"] == 3 and c.term_dims == (3, 3)


def test_isaaclab_per_term_history_lengths_must_agree():
    res = from_isaaclab_cfg({"observations": {"policy": {
        "base_ang_vel": {**BAV, "history_length": 3},
        "projected_gravity": {**PG, "history_length": 5}}}})
    assert [r["field"] for r in res.fatal] == ["observations.policy.history_length"]


def test_isaaclab_unflattened_history_is_refused():
    res = from_isaaclab_cfg({"observations": {"policy": {
        "history_length": 3, "flatten_history_dim": False, "base_ang_vel": BAV}}})
    assert [r["field"] for r in res.fatal] == ["observations.policy.flatten_history_dim"]


def test_rollout_carries_policy_state():
    """The reason history used to be refused is gone: rollout_one has a policy carry."""
    pytest.importorskip("jax")
    from breakpoint_mjx import rollout

    src = inspect.getsource(rollout.rollout_one)
    assert "init_carry" in src
    assert "act, pcarry = policy_fn(obs, nk_policy, pcarry)" in src
    assert "act, _ = policy_fn(obs, nk_policy)" in src      # stateless path unchanged
