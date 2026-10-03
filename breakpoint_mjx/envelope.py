"""Safety envelope (learned layer, M3), and its simulator certificate.

A scattered list of failing worlds is not an operating spec. This turns the surrogate
(M1) into a candidate operating spec: the largest box around nominal in which the policy
is PREDICTED safe, checked by sampling the box and asking the SURROGATE whether each
sample stays below the failure threshold. The output is a per-axis operating envelope
("friction 0.42 - 1.00, latency 0.0 - 1.8 steps") plus the surrogate-predicted pass rate.

That surrogate check is not a certificate: it samples the model's predictions, not the
simulator. ``certify_envelope`` is the honest version: it runs fresh SIMULATOR rollouts
uniformly inside the box and reports a Clopper-Pearson upper bound on the failure rate
there (riskcert.certify_box). Only that result may be called certified.

Pure numpy; the envelope search + verification are unit-tested against a known boundary.
"""
from __future__ import annotations

import numpy as np


def _predict_proba(model, theta):
    """Works for a single Surrogate or an Ensemble (uses the mean)."""
    if hasattr(model, "members"):
        mean, _ = model.predict(theta)
        thr = model.threshold
        return np.clip(mean / (2.0 * thr), 0.0, 1.0)
    return model.predict_proba(theta)


def safety_envelope(model, space, *, tol: float = 0.5, steps: int = 24,
                    verify_samples: int = 4000, seed: int = 0) -> dict:
    """Grow the largest axis-aligned safe box around the space nominal.

    tol: max acceptable predicted failure probability inside the envelope.
    Method: expand each axis outward from nominal (both directions) until the surrogate
    predicts prob > tol along that axis, then VERIFY the resulting box by sampling it and
    measuring the fraction of samples that are actually predicted safe. Interactions can
    make the per-axis box optimistic, so we report the measured verified fraction rather
    than claiming certainty; the box is shrunk toward nominal until it verifies >= 0.98.
    """
    names = space.names()
    lo = np.asarray(space.lower(), dtype=np.float64)
    hi = np.asarray(space.upper(), dtype=np.float64)
    nominal = np.asarray(space.nominal_vector(), dtype=np.float64)
    d = len(names)

    def axis_reach(dim: int, bound: float) -> float:
        """Farthest value along `dim` toward `bound` that is still predicted safe.
        The bound (lo[dim] or hi[dim]) encodes the direction; probe interpolates
        nominal -> bound, so this works for either side without a sign multiplier."""
        last = nominal[dim]
        for f in np.linspace(0.0, 1.0, steps + 1)[1:]:
            probe = nominal.copy()
            probe[dim] = nominal[dim] + f * (bound - nominal[dim])
            if float(_predict_proba(model, probe[None, :])[0]) > tol:
                break
            last = probe[dim]
        return last

    box_lo = np.array([axis_reach(i, lo[i]) for i in range(d)])
    box_hi = np.array([axis_reach(i, hi[i]) for i in range(d)])

    rng = np.random.default_rng(seed)

    def verify(blo, bhi) -> float:
        pts = blo + rng.random((verify_samples, d)) * (bhi - blo)
        proba = _predict_proba(model, pts)
        return float((proba <= tol).mean())

    # NOTE: this samples the SURROGATE's predictions, not the simulator; it is a model
    # self-consistency check ("predicted_safe_fraction"), not a certificate. Use
    # certify_envelope() with a simulator evaluate for a real upper bound.
    verified = verify(box_lo, box_hi)
    shrink = 1.0
    while verified < 0.98 and shrink > 0.2:
        shrink -= 0.1
        blo = nominal + (box_lo - nominal) * shrink
        bhi = nominal + (box_hi - nominal) * shrink
        verified = verify(blo, bhi)
        box_lo, box_hi = blo, bhi

    full = np.prod(np.where(hi > lo, hi - lo, 1.0))
    box_vol = np.prod(np.maximum(box_hi - box_lo, 0.0))
    return {
        "envelope": {
            names[i]: [float(box_lo[i]), float(box_hi[i])] for i in range(d)
        },
        "nominal": {names[i]: float(nominal[i]) for i in range(d)},
        "tolerance": float(tol),
        # Kept under its old key for web/back-compat; it is SURROGATE-predicted.
        "verified_fraction": round(verified, 4),
        "verified_by": "surrogate",
        "safe_volume_fraction": round(float(box_vol / full) if full > 0 else 0.0, 6),
        "summary": _summarize(names, box_lo, box_hi, lo, hi),
    }


def _summarize(names, box_lo, box_hi, lo, hi) -> str:
    """The axes that actually constrain safety, as a human line.

    A zero-volume box (every axis collapsed to nominal) is not an operating envelope, so
    it is summarized as such rather than as a list of ``[nominal, nominal]`` intervals."""
    if len(box_lo) and all(abs(float(box_hi[i]) - float(box_lo[i])) <= 1e-12
                           for i in range(len(box_lo))):
        return "no failure-free box found around nominal"
    parts = []
    for i, n in enumerate(names):
        constrained_lo = box_lo[i] > lo[i] + 1e-9
        constrained_hi = box_hi[i] < hi[i] - 1e-9
        if constrained_lo or constrained_hi:
            parts.append(f"{n} in [{box_lo[i]:.3g}, {box_hi[i]:.3g}]")
    return "safe while " + " AND ".join(parts) if parts else "safe across the full searched range"


IMBALANCED_FAIL_RATE = 0.8
NO_BOX_IMBALANCED = "no failure-free box found at this budget and label balance"


def nominal_sentence(nominal_check) -> str:
    """'nominal parameters: 16/16 seeds pass' from ``{"n", "fails"}`` ('' if absent)."""
    if not nominal_check or not nominal_check.get("n"):
        return ""
    n, k = int(nominal_check["n"]), int(nominal_check.get("fails", 0))
    return f"nominal parameters: {n - k}/{n} seeds pass"


def annotate_label_balance(envelope: dict, n_labeled: int, n_fail: int,
                           nominal_check=None) -> dict:
    """Make the envelope line consistent with the rest of the report, in place.

    A search that labels most worlds as failures (fail rate > 0.8) trains a surrogate
    that predicts "unsafe" nearly everywhere, so a zero-volume box then says more about
    the label balance than about the policy, and "no failure-free box around nominal"
    reads as a contradiction next to a policy that passes every nominal seed. In that
    case the summary becomes ``NO_BOX_IMBALANCED``. The fail rate and the nominal pass
    count (``{"n", "fails"}``, measured by the caller) are attached and quoted either way.
    """
    if not envelope:
        return envelope
    rate = (float(n_fail) / n_labeled) if n_labeled else None
    envelope["label_fail_rate"] = None if rate is None else round(rate, 4)
    envelope["n_labeled"] = int(n_labeled)
    if nominal_check:
        envelope["nominal_check"] = dict(nominal_check)
    box = envelope.get("envelope") or {}
    zero = bool(box) and all(abs(float(h) - float(lo)) <= 1e-12 for lo, h in box.values())
    if zero and rate is not None and rate > IMBALANCED_FAIL_RATE:
        envelope["summary"] = NO_BOX_IMBALANCED
    parts = [envelope.get("summary", "")]
    if rate is not None:
        parts.append(f"(surrogate trained on {n_labeled} worlds, {rate:.0%} labeled as "
                     f"failures")
        nom = nominal_sentence(nominal_check)
        parts[-1] += f"; {nom})" if nom else ")"
    envelope["summary_long"] = " ".join(p for p in parts if p)
    return envelope


def certify_envelope(envelope: dict, space, evaluate_fail, *, n: int = 299,
                     eps: float = 0.01, delta: float = 0.05, seed: int = 0) -> dict:
    """Simulator-backed certificate for a ``safety_envelope`` box.

    ``evaluate_fail(thetas (n, d) ndarray) -> bool (n,)`` must run FRESH simulator
    rollouts. Returns riskcert.certify_box's dict: a (1 - delta) Clopper-Pearson upper
    bound on the failure rate for theta uniform in the box; ``certified`` iff it is
    <= eps. With 0 failures, n=299 is the smallest budget that certifies eps=0.01.
    A zero-width envelope (every axis collapsed to nominal) certifies only the nominal
    point, and the result says so.
    """
    from breakpoint_mjx.riskcert import certify_box

    names = space.names()
    box = envelope["envelope"]
    lo = np.array([box[nm][0] for nm in names], dtype=np.float64)
    hi = np.array([box[nm][1] for nm in names], dtype=np.float64)
    out = certify_box(evaluate_fail, lo, hi, n, eps=eps, delta=delta,
                      rng=np.random.default_rng(seed))
    out["box_is_point"] = bool(np.all(hi - lo <= 1e-12))
    return out
