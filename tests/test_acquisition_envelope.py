"""CPU tests for active-learning acquisition (M2) + safety-envelope certificate (M3)."""
import numpy as np

from breakpoint_mjx.surrogate import Ensemble
from breakpoint_mjx.acquisition import (
    boundary_uncertainty_score, sample_candidates, select_informative,
)
from breakpoint_mjx.envelope import safety_envelope


class _Space:
    def names(self):
        return ["friction_floor", "latency_steps"]
    def lower(self):
        return [0.1, 0.0]
    def upper(self):
        return [1.0, 4.0]
    def nominal_vector(self):
        return [0.8, 0.0]  # nominal = high friction, no latency = safe


def _true_score(friction, latency):
    # fails below friction 0.3 or above latency 3.0
    return np.clip(0.9 * (friction < 0.3) + 0.7 * (latency > 3.0), 0, 1.6).astype(float)


def _train_ensemble(seed=0, k=4, n=1500):
    space = _Space()
    rng = np.random.default_rng(seed)
    fr = rng.uniform(0.1, 1.0, n)
    la = rng.uniform(0.0, 4.0, n)
    X = np.stack([fr, la], axis=1)
    y = _true_score(fr, la) + rng.normal(0, 0.03, n)
    return space, Ensemble.train(X, y, space, k=k, epochs=180)


def test_boundary_uncertainty_peaks_at_threshold():
    thr = 0.5
    mean = np.array([0.0, 0.5, 1.0])
    std = np.array([0.2, 0.2, 0.2])
    v = boundary_uncertainty_score(mean, std, thr)
    assert v[1] > v[0] and v[1] > v[2]  # peaks at the decision boundary


def test_acquisition_targets_the_decision_boundary_more_than_random():
    space, ens = _train_ensemble()
    pool = sample_candidates(space, 3000, seed=7)
    picks = select_informative(ens, pool, n=40)
    thr = ens.threshold
    # informative picks should sit much closer to the DECISION BOUNDARY (predicted
    # score ~ threshold) than a uniform sample does. The boundary is the friction<0.3
    # / latency>3 curve; acquisition targets it directly.
    pick_mean, _ = ens.predict(pool[picks])
    pool_mean, _ = ens.predict(pool)
    picks_dist = float(np.abs(pick_mean - thr).mean())
    pool_dist = float(np.abs(pool_mean - thr).mean())
    assert picks_dist < pool_dist * 0.6
    assert len(set(picks)) == len(picks)  # no duplicate rows


def test_safety_envelope_excludes_the_failure_region_and_verifies():
    space, ens = _train_ensemble()
    env = safety_envelope(ens, space, tol=0.5, verify_samples=3000)
    fr_lo, fr_hi = env["envelope"]["friction_floor"]
    la_lo, la_hi = env["envelope"]["latency_steps"]
    # the safe box must stay above the friction failure edge and below the latency edge
    assert fr_lo >= 0.3 - 0.12
    assert la_hi <= 3.0 + 0.2
    assert env["verified_fraction"] >= 0.9
    assert 0.0 < env["safe_volume_fraction"] < 1.0
    assert "friction_floor" in env["summary"] or "latency_steps" in env["summary"]
