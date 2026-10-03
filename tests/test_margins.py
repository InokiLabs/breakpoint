"""CPU unit tests for T2 safety margins + attribution (breakpoint_mjx/margins.py).

No GPU: the boundary line-search / crossing / formatting and the finite-difference
attribution are exercised with SYNTHETIC analytic score functions (e.g. "fails when
friction < 0.3"). The evaluator-backed score_fn is the thin GPU wrapper, Colab-validated.
Also covers FailureFamily round-trip with the new margin/boundaries/attribution fields +
legacy back-compat.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.margins import (
    analyze_family,
    axis_line_candidates,
    compute_attribution,
    compute_boundary,
    first_crossing_index,
)
from breakpoint_mjx.schemas import FailureFamily
from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


@pytest.fixture(scope="module")
def idx(space):
    names = space.names()
    return {n: names.index(n) for n in names}


def _cfg(failure_threshold=0.5):
    return types.SimpleNamespace(failure_threshold=failure_threshold)


# --------------------------------------------------------------------------- #
# Pure primitives
# --------------------------------------------------------------------------- #
def test_axis_line_candidates_sweeps_one_axis_only():
    nominal = np.array([0.8, 0.0, 1.0])
    target = np.array([0.15, 0.0, 1.0])
    cands, fracs = axis_line_candidates(nominal, target, axis=0, steps=11)
    assert cands.shape == (11, 3)
    assert fracs[0] == 0.0 and fracs[-1] == 1.0
    # axis 0 sweeps nominal -> target; the other axes stay at nominal.
    assert cands[0, 0] == pytest.approx(0.8)
    assert cands[-1, 0] == pytest.approx(0.15)
    np.testing.assert_allclose(cands[:, 1], 0.0)
    np.testing.assert_allclose(cands[:, 2], 1.0)


def test_first_crossing_index_uses_ge_rule():
    assert first_crossing_index([0.1, 0.2, 0.5, 0.9], 0.5) == 2   # >= counts
    assert first_crossing_index([0.1, 0.49], 0.5) is None


# --------------------------------------------------------------------------- #
# Boundary line-search + margin string
# --------------------------------------------------------------------------- #
def test_boundary_single_axis_low_friction(space, idx):
    """Synthetic: fails when friction_floor < 0.3. rep drives friction to its lower bound;
    the line search should find the boundary just below 0.3 with a '<' clause."""
    fi = idx["friction_floor"]
    nominal = np.array(space.nominal_vector())
    target = nominal.copy()
    target[fi] = 0.15                              # lower bound

    def score_fn(thetas):
        thetas = np.asarray(thetas)
        return (thetas[:, fi] < 0.3).astype(float)

    margin, boundaries = compute_boundary(
        nominal, target, space.lower(), space.upper(), space.names(),
        threshold=0.5, score_fn=score_fn, steps=11,
    )
    b = boundaries["friction_floor"]
    assert b["gates"] is True and b["op"] == "<" and b["direction"] == "low"
    # LAST-SAFE value: the last candidate still stable (>= the true 0.3 crossing), not the
    # first failing one -- so "stable until friction_floor < b" quotes a still-stable value.
    assert 0.3 <= b["boundary"] <= 0.4
    assert "friction_floor <" in margin and margin.startswith("stable until")
    # No other axis moved, so none gate.
    assert all(not v["gates"] for k, v in boundaries.items() if k != "friction_floor")


def test_boundary_joint_two_axis_AND(space, idx):
    """Spec example shape: 'stable until friction_floor < X AND latency_steps > Y'.
    Synthetic score fails when friction < 0.3 OR latency > 2.5; rep moves BOTH axes, so each
    independently gates and the margin string ANDs the two per-axis boundaries."""
    fi, li = idx["friction_floor"], idx["latency_steps"]
    nominal = np.array(space.nominal_vector())
    target = nominal.copy()
    target[fi] = 0.15
    target[li] = 4.0

    def score_fn(thetas):
        thetas = np.asarray(thetas)
        return ((thetas[:, fi] < 0.3) | (thetas[:, li] > 2.5)).astype(float)

    margin, boundaries = compute_boundary(
        nominal, target, space.lower(), space.upper(), space.names(),
        threshold=0.5, score_fn=score_fn, steps=11,
    )
    assert boundaries["friction_floor"]["gates"] and boundaries["friction_floor"]["op"] == "<"
    assert boundaries["latency_steps"]["gates"] and boundaries["latency_steps"]["op"] == ">"
    # LAST-SAFE latency value: the last candidate still stable, i.e. at/below the 2.5 crossing.
    assert 2.0 <= boundaries["latency_steps"]["boundary"] <= 2.5
    assert " AND " in margin
    assert "friction_floor <" in margin and "latency_steps >" in margin


def test_boundary_no_crossing_within_bounds(space, idx):
    fi = idx["friction_floor"]
    nominal = np.array(space.nominal_vector())
    target = nominal.copy()
    target[fi] = 0.15

    def score_fn(thetas):
        return np.zeros(len(np.asarray(thetas)))     # never fails

    margin, boundaries = compute_boundary(
        nominal, target, space.lower(), space.upper(), space.names(),
        threshold=0.5, score_fn=score_fn, steps=11,
    )
    assert margin == "no single-axis boundary within bounds"
    assert all(not v["gates"] for v in boundaries.values())


def test_boundary_fails_at_nominal(space, idx):
    nominal = np.array(space.nominal_vector())
    target = nominal.copy()
    target[idx["friction_floor"]] = 0.15

    def score_fn(thetas):
        return np.ones(len(np.asarray(thetas)))      # already failing everywhere

    margin, boundaries = compute_boundary(
        nominal, target, space.lower(), space.upper(), space.names(),
        threshold=0.5, score_fn=score_fn, steps=11,
    )
    assert margin == "fails at nominal (no perturbation needed)"
    # perturbation-based gating is suppressed when nominal already fails
    assert all(not v["gates"] for v in boundaries.values())


def test_boundary_candidates_are_in_bounds(space, idx):
    """Every candidate scored during the line search must respect the space bounds."""
    lower = np.array(space.lower())
    upper = np.array(space.upper())
    seen = []

    def score_fn(thetas):
        thetas = np.asarray(thetas)
        seen.append(thetas.copy())
        return np.zeros(len(thetas))

    nominal = np.array(space.nominal_vector())
    target = upper.copy()                            # push every axis to its upper bound
    compute_boundary(nominal, target, lower, upper, space.names(),
                     threshold=0.5, score_fn=score_fn, steps=9)
    allc = np.concatenate(seen, axis=0)
    assert (allc >= lower - 1e-9).all() and (allc <= upper + 1e-9).all()


# --------------------------------------------------------------------------- #
# Attribution (finite-difference sensitivity, normalized to sum 1)
# --------------------------------------------------------------------------- #
def test_attribution_ranks_driving_axis_and_sums_to_one(space, idx):
    """Analytic score = 3*(0.8 - friction) + 0.5*latency. Friction drives it ~6x more than
    latency; every other axis has zero sensitivity. Importances sum to 1."""
    fi, li = idx["friction_floor"], idx["latency_steps"]
    rep = np.array(space.nominal_vector())           # friction 0.8, latency 0.0

    def score_fn(thetas):
        thetas = np.asarray(thetas)
        return 3.0 * (0.8 - thetas[:, fi]) + 0.5 * thetas[:, li]

    attr = compute_attribution(rep, space.lower(), space.upper(), space.names(),
                               score_fn=score_fn, eps_frac=0.05)
    assert attr["friction_floor"] > attr["latency_steps"] > 0.0
    # all other axes are flat -> zero importance
    for name in space.names():
        if name not in ("friction_floor", "latency_steps"):
            assert attr[name] == pytest.approx(0.0)
    assert sum(attr.values()) == pytest.approx(1.0)
    # 3.0 vs 0.5 sensitivity -> ~0.857 / 0.143
    assert attr["friction_floor"] == pytest.approx(3.0 / 3.5, abs=1e-6)
    assert attr["latency_steps"] == pytest.approx(0.5 / 3.5, abs=1e-6)


def test_attribution_all_flat_is_zero(space):
    rep = np.array(space.nominal_vector())

    def score_fn(thetas):
        return np.full(len(np.asarray(thetas)), 0.42)   # constant -> no sensitivity

    attr = compute_attribution(rep, space.lower(), space.upper(), space.names(),
                               score_fn=score_fn)
    assert all(v == 0.0 for v in attr.values())


def test_attribution_candidates_respect_bounds(space, idx):
    """rep at the friction lower bound forces the -eps candidate to clip to the bound."""
    lower = np.array(space.lower())
    upper = np.array(space.upper())
    rep = np.array(space.nominal_vector())
    rep[idx["friction_floor"]] = lower[idx["friction_floor"]]   # at the bound
    seen = []

    def score_fn(thetas):
        thetas = np.asarray(thetas)
        seen.append(thetas.copy())
        return thetas[:, idx["friction_floor"]]

    compute_attribution(rep, lower, upper, space.names(), score_fn=score_fn, eps_frac=0.1)
    allc = np.concatenate(seen, axis=0)
    assert (allc >= lower - 1e-9).all() and (allc <= upper + 1e-9).all()


# --------------------------------------------------------------------------- #
# analyze_family ties boundary + attribution together
# --------------------------------------------------------------------------- #
def test_analyze_family_returns_margin_boundaries_attribution(space, idx):
    fi = idx["friction_floor"]
    rep_theta = list(space.nominal_vector())
    rep_theta[fi] = 0.15

    def score_fn(thetas):
        thetas = np.asarray(thetas)
        # smooth + monotone: rises as friction drops below nominal, crossing 0.5 mid-range,
        # so both the boundary line-search AND the finite-diff around rep are non-degenerate.
        return 1.5 * (0.8 - thetas[:, fi])

    margin, boundaries, attribution = analyze_family(
        rep_theta, space, _cfg(0.5), score_fn, steps=11, eps_frac=0.05
    )
    assert "friction_floor" in margin
    assert boundaries["friction_floor"]["gates"] is True
    assert attribution["friction_floor"] > 0.0
    assert sum(attribution.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# FailureFamily round-trip with the new T2 fields + legacy back-compat
# --------------------------------------------------------------------------- #
def _fam(**kw):
    base = dict(
        id="f1", name="low-friction slip", member_ids=["r1", "r2"],
        representative_id="r1", triggering_parameters={"friction_floor": 0.3},
        mean_failure_score=0.8, mean_transfer_score=0.6, priority_rank=1,
    )
    base.update(kw)
    return FailureFamily(**base)


def test_failure_family_roundtrips_margin_fields():
    f = _fam(
        margin="stable until friction_floor < 0.28 AND latency_steps > 2.8",
        boundaries={"friction_floor": {"gates": True, "boundary": 0.28, "direction": "low",
                                       "op": "<", "nominal": 0.8, "target": 0.15}},
        attribution={"friction_floor": 0.7, "latency_steps": 0.3},
    )
    back = FailureFamily(**json.loads(f.to_json()))
    assert back.margin == f.margin
    assert back.boundaries["friction_floor"]["op"] == "<"
    assert back.attribution == {"friction_floor": 0.7, "latency_steps": 0.3}


def test_failure_family_legacy_without_margin_fields():
    """A family serialized before T2 (no margin/boundaries/attribution) still loads, with the
    new fields defaulting to None (same discipline as T1 validity)."""
    legacy = json.dumps({
        "id": "f1", "name": "low-friction slip", "member_ids": ["r1"],
        "representative_id": "r1", "triggering_parameters": {"friction_floor": 0.3},
        "mean_failure_score": 0.8, "mean_transfer_score": 0.6, "priority_rank": 1,
        "validity": 0.5, "reproduction": [True, False],
    })
    back = FailureFamily(**json.loads(legacy))
    assert back.margin is None
    assert back.boundaries is None
    assert back.attribution is None
    assert back.validity == 0.5     # T1 field still round-trips
