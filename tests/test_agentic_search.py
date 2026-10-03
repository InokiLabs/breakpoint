"""CPU-pure tests for the multi-agent proposer's coordination logic.

The GPU evaluate and LLM completion are the injected seams; everything tested here
is deterministic numpy/stdlib.
"""
from pathlib import Path

import numpy as np
import pytest

from breakpoint_mjx.agentic_search import (
    ROLES, _role_prompt, reallocate, role_yield, slice_counts,
)
from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


# ---------------------------------------------------------------------------
# slice_counts
# ---------------------------------------------------------------------------

def test_slice_counts_sum_exactly():
    for n in (5, 17, 256, 2048):
        counts = slice_counts([0.2, 0.2, 0.2, 0.2, 0.2], n)
        assert sum(counts) == n
        assert all(c >= 0 for c in counts)


def test_slice_counts_follow_allocation():
    counts = slice_counts([0.7, 0.1, 0.1, 0.1], 100)
    assert counts[0] == 70 and sum(counts) == 100


def test_slice_counts_unnormalized_input_ok():
    assert sum(slice_counts([3.0, 1.0], 64)) == 64
    assert slice_counts([3.0, 1.0], 64)[0] == 48


# ---------------------------------------------------------------------------
# reallocate
# ---------------------------------------------------------------------------

def test_reallocate_moves_toward_yield():
    alloc = [0.25, 0.25, 0.25, 0.25]
    nxt = reallocate(alloc, [1.0, 0.0, 0.0, 0.0])
    assert nxt[0] > 0.25
    assert all(nxt[i] < 0.25 for i in (1, 2, 3))
    assert abs(sum(nxt) - 1.0) < 1e-9


def test_reallocate_respects_floor():
    # One dominant role for many rounds must never starve the others below the floor.
    alloc = [0.25] * 4
    for _ in range(50):
        alloc = reallocate(alloc, [1.0, 0.0, 0.0, 0.0], floor=0.10)
    assert min(alloc) >= 0.10 - 1e-9
    assert abs(sum(alloc) - 1.0) < 1e-9


def test_reallocate_no_signal_no_move():
    alloc = [0.4, 0.3, 0.2, 0.1]
    nxt = reallocate(alloc, [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(nxt, alloc)


# ---------------------------------------------------------------------------
# role_yield
# ---------------------------------------------------------------------------

def test_role_yield_transfer_weighted():
    # High score but zero transfer must yield nothing (absurd worlds don't get funded).
    assert role_yield([1.0, 1.0], [0.0, 0.0]) == 0.0
    assert role_yield([1.0, 0.0], [1.0, 1.0]) > role_yield([0.5, 0.0], [1.0, 1.0])


def test_role_yield_empty_slice():
    assert role_yield([], []) == 0.0


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def test_role_prompts_are_distinct_and_complete(space):
    prompts = [
        _role_prompt(space, stance, dims, [], [], 6, "go1")
        for stance, dims in ROLES.values()
    ]
    assert len(set(prompts)) == len(prompts)  # stances actually differ
    for p in prompts:
        for dim in space.dims:  # every dim named so the parser contract holds
            assert dim in p


def test_role_prompt_includes_own_memory(space):
    stance, dims = ROLES["timing"]
    own = [{"fs": 0.9, "tr": 0.8, "theta": {"latency_steps": 3.0}}]
    p = _role_prompt(space, stance, dims, [], own, 6, "go1")
    assert "fs=0.90" in p and "latency_steps=3" in p
