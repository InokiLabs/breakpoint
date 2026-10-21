"""A transparent wrapper around the engine's batched evaluator.

Every search method calls ``evaluate(thetas, key)`` exactly as it would in the engine.
The wrapper passes the call through unchanged, then labels each evaluated row with the
engine's own per-rollout failure signals (``target.objective.component_signals``) so all
methods are scored by one predefined criterion, computed from the SAME rollout the search
saw. The search algorithm never sees the labels; nothing it does is altered.

Failure label (BENCHMARK_PLAN.md section 8): a physically valid fall,
``fall == 1 and sim_invalid == 0``. Invalid rollouts are kept as artifacts.
"""
from __future__ import annotations

import time

import numpy as np


class Recorder:
    def __init__(self, evaluate, target, thresholds, env=None, base_model=None,
                 horizon=None):
        self._evaluate = evaluate
        self._target = target
        self._thr = dict(thresholds)
        self._env = env
        self._base = base_model
        self.horizon = horizon
        self.reset()

    def reset(self):
        self.thetas: list[np.ndarray] = []
        self.signals: list[dict] = []
        self.batch_times: list[float] = []     # seconds spent inside evaluate per call
        self.batch_ends: list[float] = []      # wall timestamp at the end of each call
        self.batch_sizes: list[int] = []
        self.t0 = time.time()

    def __call__(self, thetas, key):
        t = time.time()
        try:
            traj = self._evaluate(thetas, key)
            host = {k: np.asarray(v) for k, v in traj.items()}
        finally:
            # rollout_one rebinds env._mjx_model inside the trace; restore the concrete
            # model after every call (the discipline ratings.make_runner follows).
            if self._env is not None and self._base is not None:
                self._env._mjx_model = self._base
        X = np.asarray(thetas, dtype=float)
        n = X.shape[0]
        for i in range(n):
            sig = self._target.objective.component_signals({k: v[i] for k, v in host.items()},
                                                          self._thr)
            self.signals.append(sig)
        self.thetas.extend(X)
        dt = time.time() - t
        self.batch_times.append(dt)
        self.batch_ends.append(time.time())
        self.batch_sizes.append(n)
        return traj

    # ------------------------------------------------------------------ outputs
    def table(self) -> dict:
        """Column arrays for every recorded row, in evaluation order."""
        sig = self.signals
        fall = np.array([float(s.get("fall", 0.0)) for s in sig])
        invalid = np.array([float(s.get("sim_invalid", 0.0)) for s in sig])
        return {
            "theta": np.asarray(self.thetas, dtype=float).reshape(len(sig), -1)
            if sig else np.zeros((0, 0)),
            "fall": fall,
            "invalid": invalid,
            "failure": failure_label(fall, invalid),
            "invalid_reason": [str(s.get("sim_invalid_reason", "") or "") for s in sig],
            "ttf": np.array([float(s.get("time_to_failure", np.nan)) for s in sig]),
            "min_height": np.array([float(s.get("min_height", np.nan)) for s in sig]),
            "tracking_error": np.array([float(s.get("tracking_error", np.nan)) for s in sig]),
        }

    def timing(self) -> dict:
        ends = np.asarray(self.batch_ends) - self.t0
        return {"batch_sizes": list(map(int, self.batch_sizes)),
                "batch_eval_s": [round(x, 4) for x in self.batch_times],
                "batch_end_elapsed_s": [round(float(x), 4) for x in ends],
                "eval_s": round(float(np.sum(self.batch_times)), 3),
                "wall_s": round(float(ends[-1]) if len(ends) else 0.0, 3)}


def failure_label(fall, invalid):
    """The predefined failure criterion, vectorized. Kept separate so it is unit-tested."""
    fall = np.asarray(fall, dtype=float)
    invalid = np.asarray(invalid, dtype=float)
    return ((fall > 0.5) & (invalid < 0.5)).astype(float)
