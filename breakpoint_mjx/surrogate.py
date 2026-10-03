"""Learned failure surrogate (learned layer, M1).

A small MLP that predicts a policy's failure score from a world-perturbation vector
(theta), trained on the labeled (theta, failure_score) pairs every search already
produces. Pure numpy: no jax, no GPU -- it trains in seconds on a laptop and its
weights serialize to plain JSON, so the same forward pass runs in the browser (a live
"what-if" risk explorer) and in Python (surrogate-guided search + safety envelopes).

Why it matters:
  * Every run adds thousands of labeled points, so the surrogate sharpens with use.
  * It turns an expensive GPU query (evaluate a theta) into a free CPU/JS one, which
    is what makes M2 (active-learning acquisition) and M3 (safety envelopes) tractable.

Contract:
  * inputs are theta vectors in RAW units; the surrogate stores the ThetaSpace bounds
    and min-max normalizes internally to [0, 1], so callers pass raw theta.
  * target is the failure score (search's failure_score, roughly [0, ~1.4]); predict()
    also returns a failure-probability-LIKE number, clip(score / (2 * threshold), 0, 1),
    via a stored threshold, for the UI. It is a rescaled regression output, NOT a
    calibrated probability (0.5 means "predicted score == threshold"). For a probability
    with a guarantee, see riskcert.py, which uses the surrogate only as a control variate.
"""
from __future__ import annotations

import json
import math

import numpy as np

ACTIVATIONS = {
    "relu": (lambda x: np.maximum(x, 0.0), lambda x: (x > 0.0).astype(x.dtype)),
    "tanh": (np.tanh, lambda x: 1.0 - np.tanh(x) ** 2),
}


def _he_init(rng: np.random.Generator, n_in: int, n_out: int) -> np.ndarray:
    return rng.normal(0.0, math.sqrt(2.0 / max(n_in, 1)), size=(n_in, n_out)).astype(np.float64)


class Surrogate:
    """A min-max-normalized MLP theta -> failure score, trained with Adam. Pure numpy."""

    def __init__(self, dim_names, lo, hi, hidden=(64, 64), activation="relu",
                 threshold: float = 0.5):
        self.dim_names = list(dim_names)
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        self.span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self.hidden = tuple(hidden)
        self.activation = activation
        self.threshold = float(threshold)
        # y standardization (filled at fit time)
        self.y_mean = 0.0
        self.y_std = 1.0
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

    # ---- normalization ----
    def _nx(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        return np.clip((theta - self.lo) / self.span, 0.0, 1.0)

    def _init_params(self, rng: np.random.Generator) -> None:
        sizes = [len(self.dim_names), *self.hidden, 1]
        self.weights = [_he_init(rng, sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.biases = [np.zeros(sizes[i + 1], dtype=np.float64) for i in range(len(sizes) - 1)]

    def _forward(self, xn: np.ndarray):
        act, _ = ACTIVATIONS[self.activation]
        a = xn
        cache = [a]
        pre = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            pre.append(z)
            a = z if i == len(self.weights) - 1 else act(z)  # linear output head
            cache.append(a)
        return a[:, 0], cache, pre

    # ---- training ----
    def fit(self, thetas, scores, *, epochs: int = 400, lr: float = 3e-3,
            batch: int = 128, seed: int = 0, weight_decay: float = 1e-5) -> "Surrogate":
        rng = np.random.default_rng(seed)
        self._init_params(rng)
        xn = self._nx(thetas)
        y = np.asarray(scores, dtype=np.float64)
        self.y_mean, self.y_std = float(y.mean()), float(y.std() or 1.0)
        yt = (y - self.y_mean) / self.y_std
        n = xn.shape[0]
        _, dact = ACTIVATIONS[self.activation]

        mW = [np.zeros_like(w) for w in self.weights]
        vW = [np.zeros_like(w) for w in self.weights]
        mB = [np.zeros_like(b) for b in self.biases]
        vB = [np.zeros_like(b) for b in self.biases]
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        for _ in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s : s + batch]
                xb, yb = xn[idx], yt[idx]
                out, cache, pre = self._forward(xb)
                m = len(idx)
                grad = (2.0 / m) * (out - yb)[:, None]  # dL/dz_last
                t += 1
                for li in reversed(range(len(self.weights))):
                    a_prev = cache[li]
                    gW = a_prev.T @ grad + weight_decay * self.weights[li]
                    gB = grad.sum(axis=0)
                    if li > 0:
                        grad = (grad @ self.weights[li].T) * dact(pre[li - 1])
                    # Adam
                    for p, g, mm, vv in ((self.weights[li], gW, mW[li], vW[li]),):
                        mm[...] = b1 * mm + (1 - b1) * g
                        vv[...] = b2 * vv + (1 - b2) * (g * g)
                        p -= lr * (mm / (1 - b1**t)) / (np.sqrt(vv / (1 - b2**t)) + eps)
                    mB[li][...] = b1 * mB[li] + (1 - b1) * gB
                    vB[li][...] = b2 * vB[li] + (1 - b2) * (gB * gB)
                    self.biases[li] -= lr * (mB[li] / (1 - b1**t)) / (
                        np.sqrt(vB[li] / (1 - b2**t)) + eps
                    )
        return self

    # ---- inference ----
    def predict_score(self, theta) -> np.ndarray:
        out, _, _ = self._forward(self._nx(theta))
        return out * self.y_std + self.y_mean

    def predict_proba(self, theta) -> np.ndarray:
        """``clip(score / (2 * threshold), 0, 1)``: a UI-friendly rescaling, NOT calibrated.

        0.5 exactly at the failure threshold. Nothing guarantees that the fraction of
        worlds that fail among those predicted at 0.3 is 0.3; do not read it as a
        probability. riskcert.py is the path that produces one with a guarantee.
        """
        s = self.predict_score(theta)
        return np.clip(s / (2.0 * self.threshold), 0.0, 1.0)

    def r2(self, thetas, scores) -> float:
        y = np.asarray(scores, dtype=np.float64)
        pred = self.predict_score(thetas)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
        return 1.0 - ss_res / ss_tot

    # ---- portable serialization (browser reads this) ----
    def to_dict(self) -> dict:
        return {
            "format": "breakpoint-surrogate-1",
            "dim_names": self.dim_names,
            "lo": self.lo.tolist(),
            "hi": self.hi.tolist(),
            "hidden": list(self.hidden),
            "activation": self.activation,
            "threshold": self.threshold,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Surrogate":
        s = cls(d["dim_names"], d["lo"], d["hi"], tuple(d["hidden"]),
                d["activation"], d.get("threshold", 0.5))
        s.y_mean = float(d["y_mean"])
        s.y_std = float(d["y_std"])
        s.weights = [np.asarray(w, dtype=np.float64) for w in d["weights"]]
        s.biases = [np.asarray(b, dtype=np.float64) for b in d["biases"]]
        return s

    @classmethod
    def from_json(cls, text: str) -> "Surrogate":
        return cls.from_dict(json.loads(text))


R2_MIN_N = 200          # below this many labeled worlds, no R2 is shown
R2_MIN_MINORITY = 20    # below this many of the rarer class (fail vs pass), no R2 is shown


def cv_r2(thetas, scores, lo, hi, dim_names, *, k: int = 5, seed: int = 0,
          **fit_kw) -> float:
    """``k``-fold cross-validated R2: each fold is predicted by a surrogate that never saw
    it, and R2 is computed over the pooled out-of-fold predictions. In-sample R2 of a
    64x64 MLP on a few hundred points is ~1.0 by memorization and says nothing."""
    X = np.asarray(thetas, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    idx = np.random.default_rng(seed).permutation(len(y))
    pred = np.empty_like(y)
    for fold in np.array_split(idx, k):
        tr = np.setdiff1d(idx, fold, assume_unique=True)
        m = Surrogate(dim_names, lo, hi).fit(X[tr], y[tr], seed=seed, **fit_kw)
        pred[fold] = m.predict_score(X[fold])
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    return 1.0 - ss_res / ss_tot


def r2_report(records, space, threshold: float, *, k: int = 5, seed: int = 0,
              **fit_kw) -> dict:
    """The surrogate fit quality the product is allowed to print.

    ``{"r2": float | None, "method": "5-fold cross-validated", "n", "n_fail", "n_pass",
    "hidden_reason": str | None}``. The R2 is HIDDEN (``r2`` None, with the reason) when
    there are fewer than ``R2_MIN_N`` labeled worlds or fewer than ``R2_MIN_MINORITY`` of
    the rarer class: a search that labels 179 of 192 worlds as failures leaves too few
    passes for any fit number to mean "it knows where the boundary is"."""
    y = np.array([r.failure_score for r in records], dtype=np.float64)
    n = int(len(y))
    n_fail = int((y >= float(threshold)).sum())
    n_pass = n - n_fail
    out = {"r2": None, "method": f"{k}-fold cross-validated", "n": n, "n_fail": n_fail,
           "n_pass": n_pass, "hidden_reason": None}
    minority = min(n_fail, n_pass)
    if n < R2_MIN_N or minority < R2_MIN_MINORITY:
        out["hidden_reason"] = (
            f"not shown: {n} labeled worlds ({n_fail} fail / {n_pass} pass); a fit number "
            f"needs >= {R2_MIN_N} worlds and >= {R2_MIN_MINORITY} of the rarer class to "
            f"say anything about the failure boundary")
        return out
    X = np.array([r.theta_vector for r in records], dtype=np.float64)
    out["r2"] = round(float(cv_r2(X, y, space.lower(), space.upper(), space.names(), k=k,
                                  seed=seed, **fit_kw)), 4)
    return out


def train_from_records(records, space, **fit_kw) -> Surrogate:
    """Fit a surrogate on a search's FailureRecords (theta_vector -> failure_score)."""
    thetas = np.array([r.theta_vector for r in records], dtype=np.float64)
    scores = np.array([r.failure_score for r in records], dtype=np.float64)
    sur = Surrogate(space.names(), space.lower(), space.upper())
    return sur.fit(thetas, scores, **fit_kw)


class Ensemble:
    """A deep ensemble of surrogates. The MEAN is the prediction; the STD across members
    is epistemic uncertainty -- what powers active-learning acquisition (M2) and the
    confidence band on the safety envelope (M3). Different seeds => different inits =>
    disagreement where data is sparse."""

    def __init__(self, members: list[Surrogate]):
        self.members = members

    @classmethod
    def train(cls, thetas, scores, space, k: int = 5, **fit_kw) -> "Ensemble":
        members = []
        for i in range(k):
            s = Surrogate(space.names(), space.lower(), space.upper())
            s.fit(thetas, scores, seed=i, **fit_kw)
            members.append(s)
        return cls(members)

    def predict(self, theta):
        """Returns (mean_score, std_score) over the ensemble."""
        preds = np.stack([m.predict_score(theta) for m in self.members], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    @property
    def threshold(self) -> float:
        return self.members[0].threshold
