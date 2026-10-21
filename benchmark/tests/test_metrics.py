"""Unit tests for benchmark metrics, failure classification and candidate selection."""
import numpy as np
import pytest

from benchmark.analysis import metrics as M
from benchmark.evaluation.recorder import Recorder, failure_label


# ------------------------------------------------------------------ failure label
def test_failure_label_requires_valid_fall():
    fall = [1, 1, 0, 0]
    invalid = [0, 1, 0, 1]
    assert failure_label(fall, invalid).tolist() == [1.0, 0.0, 0.0, 0.0]


# -------------------------------------------------------------------- clustering
def test_leader_clusters_merges_within_eps_linf():
    U = np.array([[0.10, 0.10], [0.15, 0.18], [0.50, 0.50], [0.10, 0.25]])
    labels, leaders = M.leader_clusters(U, eps=0.1)
    # point 1 is within 0.08 of point 0 in L-inf; point 3 is 0.15 away in dim 1
    assert labels.tolist() == [0, 0, 1, 2]
    assert leaders == [0, 2, 3]


def test_leader_clusters_eps_boundary_is_inclusive():
    U = np.array([[0.0], [0.1]])
    labels, leaders = M.leader_clusters(U, eps=0.1)
    assert leaders == [0] and labels.tolist() == [0, 0]


def test_leader_clusters_empty():
    labels, leaders = M.leader_clusters(np.zeros((0, 3)), eps=0.1)
    assert leaders == [] and labels.size == 0


def test_near_duplicates_are_not_counted_twice():
    rng = np.random.default_rng(0)
    base = np.full((50, 4), 0.5)
    U = base + rng.uniform(-0.02, 0.02, base.shape)
    _, leaders = M.leader_clusters(U, eps=0.1)
    assert len(leaders) == 1


def test_discovery_curve_counts_only_new_clusters_and_ignores_non_failures():
    U = np.array([[0.0], [0.05], [0.9], [0.5], [0.52]])
    fail = np.array([True, True, False, True, True])
    curve = M.discovery_curve(fail, U, eps=0.1)
    assert curve.tolist() == [1, 1, 1, 2, 2]


def test_discovery_curve_matches_leader_count():
    rng = np.random.default_rng(1)
    U = rng.random((200, 3))
    fail = rng.random(200) < 0.4
    curve = M.discovery_curve(fail, U, eps=0.2)
    _, leaders = M.leader_clusters(U[fail], eps=0.2)
    assert curve[-1] == len(leaders)


def test_box_normalize_maps_bounds_to_unit_and_handles_zero_span():
    X = np.array([[1.0, 5.0, 2.0], [3.0, 5.0, 4.0]])
    U = M.box_normalize(X, lo=[1.0, 5.0, 2.0], hi=[3.0, 5.0, 6.0])
    assert np.allclose(U, [[0, 0, 0], [1, 0, 0.5]])


# --------------------------------------------------------------------- intervals
def test_clopper_pearson_known_values():
    lo, hi = M.clopper_pearson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(0.1684, abs=1e-3)
    lo, hi = M.clopper_pearson(20, 20)
    assert hi == 1.0 and lo == pytest.approx(0.8316, abs=1e-3)
    lo, hi = M.clopper_pearson(10, 20)
    assert lo == pytest.approx(0.2720, abs=1e-3) and hi == pytest.approx(0.7280, abs=1e-3)


def test_validation_rule_needs_15_of_20():
    assert M.min_k_to_validate(20) == 15
    assert not M.is_validated(14, 20)
    assert M.is_validated(15, 20)
    assert not M.is_validated(0, 0)


# ----------------------------------------------------------- candidate selection
def test_select_candidates_takes_all_leaders_when_few():
    first, sampled, exact = M.select_candidates([3, 7, 9, 11], [3, 9], 3, 8, seed=1)
    assert first == [3, 7, 9] and sampled == [3, 9] and exact


def test_select_candidates_random_and_deterministic():
    leaders = list(range(100))
    a = M.select_candidates([0, 1], leaders, 3, 8, seed=5)
    b = M.select_candidates([0, 1], leaders, 3, 8, seed=5)
    c = M.select_candidates([0, 1], leaders, 3, 8, seed=6)
    assert a == b and a[1] != c[1] and len(a[1]) == 8 and not a[2]


def test_time_to_first_validated():
    assert M.time_to_first_validated([4, 10, 20], {4: False, 10: True, 20: True}) == 11
    assert M.time_to_first_validated([4, 10], {4: False, 10: False}) is None


def test_estimated_validated_distinct():
    e = M.estimated_validated_distinct(100, 4, 8, exact=False)
    assert e["estimate"] == 50 and e["lo"] < 50 < e["hi"]
    e = M.estimated_validated_distinct(5, 3, 5, exact=True)
    assert e["estimate"] == 3 and e["exact"]


# ------------------------------------------------------------ across repetitions
def test_bootstrap_ratio_and_zero_baseline():
    r = M.bootstrap_ratio([4, 6, 5], [2, 3, 2.5])
    assert r["ratio"] == pytest.approx(2.0) and r["lo"] <= 2.0 <= r["hi"]
    assert M.bootstrap_ratio([1, 2], [0, 0])["ratio"] is None


def test_summarize_reps_ignores_none():
    s = M.summarize_reps([1.0, None, 3.0])
    assert s["mean"] == 2.0 and s["n"] == 2


def test_paired_fall_difference():
    d = M.paired_fall_difference([0, 0, 1, 0], [1, 0, 1, 1])
    assert d["diff"] == pytest.approx(0.5) and d["n"] == 4


# ------------------------------------------------------------------- recorder
class _FakeObjective:
    @staticmethod
    def component_signals(traj, thr):
        h = float(np.min(traj["h"]))
        return {"fall": float(h < thr["trunk_height_min"]), "sim_invalid": float(traj["bad"][0]),
                "sim_invalid_reason": "penetration" if traj["bad"][0] else "",
                "time_to_failure": 3.0, "min_height": h, "tracking_error": 0.1}


class _FakeTarget:
    objective = _FakeObjective()


def test_recorder_passes_through_and_labels_rows():
    def evaluate(thetas, key):
        th = np.asarray(thetas)
        h = np.repeat(th[:, :1], 5, axis=1)          # height = theta[0]
        bad = (th[:, 1:2] > 0.5).astype(float).repeat(5, axis=1)
        return {"h": h, "bad": bad}

    rec = Recorder(evaluate, _FakeTarget(), {"trunk_height_min": 0.18})
    X = np.array([[0.1, 0.0], [0.1, 0.9], [0.3, 0.0]])
    out = rec(X, None)
    assert set(out) == {"h", "bad"}                  # traj returned unchanged
    t = rec.table()
    assert t["fall"].tolist() == [1, 1, 0]
    assert t["invalid"].tolist() == [0, 1, 0]
    assert t["failure"].tolist() == [1, 0, 0]        # invalid fall is not a failure
    assert t["invalid_reason"][1] == "penetration"
    assert rec.timing()["batch_sizes"] == [3]
