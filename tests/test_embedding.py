"""CPU tests for the cross-policy failure embedding (M4)."""
import numpy as np

from breakpoint_mjx.embedding import (
    cosine_similarity, nearest_runs, run_embedding, transfer_hot_regions,
)


class _Space:
    def names(self):
        return ["friction_floor", "latency_steps", "torso_mass_add_kg"]
    def lower(self):
        return [0.1, 0.0, -1.0]
    def upper(self):
        return [1.0, 4.0, 6.0]
    def nominal_vector(self):
        return [0.55, 0.0, 0.0]


class _Fam:
    def __init__(self, theta, score):
        self.triggering_parameters = theta
        self.mean_failure_score = score


def _low_friction_run():
    return [
        _Fam({"friction_floor": 0.18, "latency_steps": 1.0, "torso_mass_add_kg": 0.5}, 1.0),
        _Fam({"friction_floor": 0.22, "latency_steps": 0.5, "torso_mass_add_kg": 0.0}, 0.9),
    ]


def _latency_run():
    return [
        _Fam({"friction_floor": 0.6, "latency_steps": 3.6, "torso_mass_add_kg": 0.0}, 1.0),
        _Fam({"friction_floor": 0.55, "latency_steps": 3.2, "torso_mass_add_kg": 0.2}, 0.8),
    ]


def test_same_mode_runs_embed_closer_than_different_mode():
    space = _Space()
    a = run_embedding(_low_friction_run(), space, failure_rate=0.08)
    # a second low-friction policy, slightly different numbers
    a2 = run_embedding(
        [_Fam({"friction_floor": 0.2, "latency_steps": 1.2, "torso_mass_add_kg": 0.3}, 1.0)],
        space, failure_rate=0.07,
    )
    b = run_embedding(_latency_run(), space, failure_rate=0.09)
    assert cosine_similarity(a, a2) > cosine_similarity(a, b)
    assert cosine_similarity(a, a2) > 0.8


def test_embedding_is_fixed_length_regardless_of_family_count():
    space = _Space()
    one = run_embedding(_low_friction_run()[:1], space)
    many = run_embedding(_low_friction_run() * 5, space)
    assert one.shape == many.shape == (2 * 3 + 1,)


def test_nearest_runs_retrieves_the_same_failure_mode():
    space = _Space()
    library = [
        run_embedding(_latency_run(), space),        # 0: latency
        run_embedding(_low_friction_run(), space),   # 1: friction
        run_embedding([_Fam({"torso_mass_add_kg": 5.0, "friction_floor": 0.55, "latency_steps": 0.0}, 1.0)], space),  # 2: payload
    ]
    query = run_embedding(
        [_Fam({"friction_floor": 0.19, "latency_steps": 0.8, "torso_mass_add_kg": 0.1}, 1.0)],
        space,
    )
    top = nearest_runs(query, library, k=1)
    assert top[0][0] == 1  # the friction run is nearest


def test_transfer_hot_regions_returns_top_severity_thetas():
    seeds = transfer_hot_regions(_low_friction_run() + _latency_run(), _Space(), top=2)
    assert len(seeds) == 2
    assert all(isinstance(s, dict) for s in seeds)
    # highest severity first (both 1.0 here); returns raw theta dicts for seeding
    assert "friction_floor" in seeds[0]
