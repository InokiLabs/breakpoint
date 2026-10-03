"""Unit tests for the PURE MAP-Elites archive logic in breakpoint_mjx.atlas.

Covers ttf binning bounds, transfer-weighted fitness, best-per-cell insertion, and
grid coverage. No JAX/GPU: the map_elites_search loop is validated on Colab."""
import numpy as np
import pytest

from breakpoint_mjx.atlas import _ttf_bin, _fitness, _insert
from breakpoint_mjx.schemas import FailureRecord


def _rec(failure_score, transfer_score, rid="r"):
    """Minimal FailureRecord with just the fields the atlas reads."""
    return FailureRecord(
        theta_vector=[0.0],
        theta_named={"x": 0.0},
        failure_score=float(failure_score),
        component_signals={"fall": 1.0},
        rollout_signature=[0.0],
        seed=0,
        transfer_score=transfer_score,
        id=rid,
    )


# ---------------------------------------------------------------------------
# _ttf_bin: bounds + clamping
# ---------------------------------------------------------------------------

def test_ttf_bin_endpoints():
    n = 8
    assert _ttf_bin(0.0, n) == 0
    assert _ttf_bin(1.0, n) == n - 1            # 1.0 lands in the last bin


def test_ttf_bin_midpoint():
    # int(0.5 * (8-1)) = int(3.5) = 3
    assert _ttf_bin(0.5, 8) == 3


def test_ttf_bin_clamps_out_of_range():
    assert _ttf_bin(-1.0, 8) == 0
    assert _ttf_bin(2.0, 8) == 7
    assert _ttf_bin(0.999999, 8) == 6           # just under 1.0 stays below the top bin


# ---------------------------------------------------------------------------
# _fitness: transfer-weighted failure score
# ---------------------------------------------------------------------------

def test_fitness_is_product():
    assert _fitness(_rec(0.8, 0.5)) == pytest.approx(0.4)


def test_fitness_none_transfer_is_zero():
    assert _fitness(_rec(0.9, None)) == 0.0


# ---------------------------------------------------------------------------
# _insert: best-per-cell, grid coverage
# ---------------------------------------------------------------------------

def test_insert_keeps_higher_fitness_per_cell():
    """Two records in the SAME cell (same mode + ttf bin) -> higher fitness wins,
    regardless of insertion order."""
    n = 8
    desc = [(0, 0.0), (0, 0.0)]   # both -> cell (0, 0)
    low = _rec(0.5, 0.5, "low")   # fitness 0.25
    high = _rec(0.9, 0.9, "high") # fitness 0.81

    a = _insert({}, desc, [low, high], n)
    assert set(a.keys()) == {(0, 0)}
    assert a[(0, 0)].id == "high"

    # reverse order -> same winner
    b = _insert({}, list(reversed(desc)), [high, low], n)
    assert b[(0, 0)].id == "high"


def test_insert_lower_fitness_does_not_replace():
    n = 8
    a = _insert({}, [(1, 0.0)], [_rec(0.9, 0.9, "first")], n)
    a = _insert(a, [(1, 0.0)], [_rec(0.1, 0.1, "second")], n)
    assert a[(1, 0)].id == "first"


def test_insert_distinct_cells_all_kept():
    """Distinct (mode, ttf) descriptors occupy distinct cells (grid coverage)."""
    n = 8
    desc = [(0, 0.0), (1, 0.0), (2, 1.0), (3, 0.5)]
    recs = [_rec(0.6, 0.6, f"r{i}") for i in range(4)]
    a = _insert({}, desc, recs, n)
    assert set(a.keys()) == {(0, 0), (1, 0), (2, n - 1), (3, _ttf_bin(0.5, n))}
    assert len(a) == 4


def test_insert_mode_and_bin_compose_key():
    """Same mode but different ttf bins -> separate cells, not overwrites."""
    n = 8
    desc = [(0, 0.0), (0, 1.0)]
    a = _insert({}, desc, [_rec(0.5, 0.5, "early"), _rec(0.5, 0.5, "late")], n)
    assert set(a.keys()) == {(0, 0), (0, n - 1)}
