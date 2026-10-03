"""Unit tests for breakpoint_mjx.clustering -- pure, no JAX/GPU required."""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pytest

from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.schemas import FailureRecord
from breakpoint_mjx.clustering import cluster_failures, name_family, build_families
from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]
SIGNATURE_LEN = 48  # 16 points * 3 channels (height, up_proj, tracking-error)


@pytest.fixture(scope="module")
def space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


def _make_record(
    space,
    theta_named: dict,
    signature: list,
    failure_score: float,
    transfer_score: float,
    severity: str,
    record_id: str,
) -> FailureRecord:
    """Build a FailureRecord from a named theta dict + explicit signature."""
    theta_vector = [theta_named[n] for n in space.names()]
    return FailureRecord(
        theta_vector=theta_vector,
        theta_named=theta_named,
        failure_score=failure_score,
        component_signals={"fall": 0.0, "time_to_failure": 500.0,
                           "tracking_error": 0.1, "instability": 0.0,
                           "min_height": 0.28},
        rollout_signature=signature,
        seed=0,
        severity=severity,
        transfer_score=transfer_score,
        id=record_id,
    )


# ---------------------------------------------------------------------------
# Family names are capped at 3 clauses (active/boundary search finds families that
# overlap on many axes, which otherwise deepen into unreadable 6-way names).
# ---------------------------------------------------------------------------

def test_family_names_capped_at_three_clauses(space):
    names = space.names()
    nominal = {n: space.dims[n].nominal for n in names}
    # Six families placed at distinct multi-dim corners so each deviates from the global
    # mean on MANY axes at once -- the case that made the disambiguation loop grow long
    # names. flip[j] chooses high/low per family across several dims.
    multi = ["torso_mass_add_kg", "kp_scale", "kd_scale", "friction_floor",
             "latency_steps", "obs_noise_scale"]
    records = []
    for c in range(6):
        theta = dict(nominal)
        for j, dim in enumerate(multi):
            lo, hi = space.dims[dim].bounds
            theta[dim] = hi if ((c >> j) & 1) else lo   # distinct sign pattern per family
        for i in range(4):
            records.append(_make_record(space, theta, [0.0] * SIGNATURE_LEN,
                                        0.8, 0.5, "critical", f"c{c}-{i}"))
    families, _ = build_families(records, space, k=6)
    assert len(families) >= 1
    for f in families:
        # Strip a trailing numeric disambiguator "(n)" before counting label clauses.
        base = re.sub(r"\s*\(\d+\)$", "", f.name)
        clauses = base.split(" + ")
        assert len(clauses) <= 3, f"family name has >3 clauses: {f.name!r}"
    # And the cap is actually exercised: at least one family used more than the primary label.
    assert any(" + " in re.sub(r"\s*\(\d+\)$", "", f.name) for f in families)


# ---------------------------------------------------------------------------
# cluster_failures: two clearly-separable groups end up in different clusters
# ---------------------------------------------------------------------------

def test_cluster_failures_separates_two_groups(space):
    names = space.names()
    nominal = {n: space.dims[n].nominal for n in names}
    upper = {n: space.dims[n].bounds[1] for n in names}

    # Group A: signature near zeros, theta near nominal
    sig_a = [0.0] * SIGNATURE_LEN
    group_a = [
        _make_record(space, nominal, sig_a, 0.2, 0.9, "low", f"a-{i}")
        for i in range(10)
    ]

    # Group B: signature near a large constant, theta near upper bounds
    sig_b = [5.0] * SIGNATURE_LEN
    group_b = [
        _make_record(space, upper, sig_b, 0.85, 0.3, "critical", f"b-{i}")
        for i in range(10)
    ]

    records = group_a + group_b
    labels = cluster_failures(records, space, k=2)

    assert len(labels) == len(records)
    # Two distinct cluster labels must be present
    assert len(set(int(l) for l in labels)) == 2, \
        f"Expected 2 clusters, got {set(labels)}"

    # All group A records should share the same cluster label (and it should differ from group B)
    labels_a = set(int(labels[i]) for i in range(10))
    labels_b = set(int(labels[i]) for i in range(10, 20))
    assert labels_a.isdisjoint(labels_b), \
        f"Groups A and B should be in different clusters; got A={labels_a}, B={labels_b}"


# ---------------------------------------------------------------------------
# name_family: friction_floor far below nominal -> name contains "friction"
# ---------------------------------------------------------------------------

def test_name_family_friction(space):
    names = space.names()
    nominal = {n: space.dims[n].nominal for n in names}

    # friction_floor far below nominal (0.80); all others at nominal
    low_friction = dict(nominal)
    low_friction["friction_floor"] = 0.18

    sig = [0.5] * SIGNATURE_LEN
    members = [
        _make_record(space, low_friction, sig, 0.75, 0.5, "high", f"fric-{i}")
        for i in range(5)
    ]

    result = name_family(members, space)
    assert "friction" in result.lower(), \
        f"Expected name containing 'friction', got: {result!r}"


# ---------------------------------------------------------------------------
# build_families: structural invariants hold over ~12 synthetic records
# ---------------------------------------------------------------------------

def test_build_families_invariants(space):
    names = space.names()
    nominal = {n: space.dims[n].nominal for n in names}
    upper = {n: space.dims[n].bounds[1] for n in names}
    lower = {n: space.dims[n].bounds[0] for n in names}

    rng = np.random.default_rng(42)

    def _rand_theta():
        return {n: float(rng.uniform(space.dims[n].bounds[0], space.dims[n].bounds[1]))
                for n in names}

    # Mix of severity/transfer/failure_score to exercise priority_key sorting
    specs = [
        (nominal,   [0.1] * SIGNATURE_LEN, 0.45, 0.9,  "medium"),
        (upper,     [5.0] * SIGNATURE_LEN, 0.92, 0.25, "critical"),
        (lower,     [3.0] * SIGNATURE_LEN, 0.78, 0.4,  "high"),
        (nominal,   [0.2] * SIGNATURE_LEN, 0.30, 0.8,  "low"),
        (upper,     [4.8] * SIGNATURE_LEN, 0.88, 0.3,  "critical"),
        (lower,     [2.9] * SIGNATURE_LEN, 0.71, 0.45, "high"),
        (_rand_theta(), [1.0] * SIGNATURE_LEN, 0.50, 0.6,  "medium"),
        (_rand_theta(), [1.2] * SIGNATURE_LEN, 0.60, 0.55, "medium"),
        (_rand_theta(), [4.5] * SIGNATURE_LEN, 0.82, 0.35, "high"),
        (nominal,   [0.05]*SIGNATURE_LEN, 0.25, 0.85, "low"),
        (upper,     [4.9] * SIGNATURE_LEN, 0.95, 0.2,  "critical"),
        (lower,     [3.1] * SIGNATURE_LEN, 0.70, 0.5,  "high"),
    ]

    records = [
        _make_record(space, theta, sig, fs, tr, sev, f"r-{i}")
        for i, (theta, sig, fs, tr, sev) in enumerate(specs)
    ]

    families, ranked = build_families(records, space, k=3)

    # At least 2 families returned
    assert len(families) >= 2, f"Expected >= 2 families, got {len(families)}"

    # Every record has a non-None priority_rank and family_id
    for r in ranked:
        assert r.priority_rank is not None, f"Record {r.id} missing priority_rank"
        assert r.family_id is not None, f"Record {r.id} missing family_id"

    # Every family has non-empty member_ids
    for fam in families:
        assert len(fam.member_ids) > 0, f"Family {fam.id} has no members"

    # representative_id is in member_ids
    for fam in families:
        assert fam.representative_id in fam.member_ids, (
            f"Family {fam.id}: representative_id {fam.representative_id!r} "
            f"not in member_ids {fam.member_ids}"
        )

    # Families are sorted by ascending priority_rank
    ranks = [fam.priority_rank for fam in families]
    assert ranks == sorted(ranks), f"Families not sorted by priority_rank: {ranks}"
