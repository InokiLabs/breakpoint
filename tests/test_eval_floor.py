"""CPU tests for the floor/portfolio aggregation (scripts/eval_floor.py) -- pure numpy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_floor import floor_stats  # noqa: E402


def test_floor_is_intersection_of_failures():
    F = np.array([
        [1, 1, 0, 0, 1],   # policy A fails worlds 0,1,4
        [1, 0, 1, 0, 1],   # policy B fails worlds 0,2,4
    ], dtype=bool)
    st = floor_stats(F)
    assert st["per_policy"] == [pytest.approx(0.6), pytest.approx(0.6)]
    assert st["floor_union"] == pytest.approx(0.4)      # worlds 0 and 4 beat everyone
    assert st["portfolio"] == st["floor_union"]         # oracle == union floor


def test_single_policy_floor_equals_its_rate():
    F = np.array([[1, 0, 0, 1]], dtype=bool)
    st = floor_stats(F)
    assert st["floor_union"] == pytest.approx(0.5) == st["per_policy"][0]


def test_disjoint_failures_have_zero_floor():
    F = np.array([[1, 0], [0, 1]], dtype=bool)
    assert floor_stats(F)["floor_union"] == 0.0


def test_bad_shapes_rejected():
    with pytest.raises(ValueError, match="n_policies"):
        floor_stats(np.zeros((0, 5), dtype=bool))
    with pytest.raises(ValueError, match="n_policies"):
        floor_stats(np.zeros(5, dtype=bool))
