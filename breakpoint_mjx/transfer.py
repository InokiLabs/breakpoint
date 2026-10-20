"""Single-sim transfer-likelihood heuristic + severity + priority. Pure."""
from __future__ import annotations
from breakpoint_mjx.config import ThetaSpace

SEVERITY_WEIGHT = {"low": 0.7, "medium": 0.85, "high": 1.0, "critical": 1.15}
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def _dim_plausibility(value, plausible, bounds):
    lo_p, hi_p = plausible; lo_b, hi_b = bounds
    if lo_p <= value <= hi_p:
        return 1.0
    if value < lo_p:
        span = max(lo_p - lo_b, 1e-9)
        return max(0.0, 1.0 - (lo_p - value) / span)
    span = max(hi_b - hi_p, 1e-9)
    return max(0.0, 1.0 - (value - hi_p) / span)


def transfer_likelihood(theta_named, space: ThetaSpace) -> float:
    vals = [
        _dim_plausibility(theta_named[name], dim.plausible, dim.bounds)
        for name, dim in space.dims.items() if name in theta_named
    ]
    return float(sum(vals) / len(vals)) if vals else 0.0


#: Credibility kept by a world with ONE axis at its hard bound and every other axis inside
#: its plausible band. Each further axis at its bound multiplies by this again, so
#: compounding extremes cost geometrically instead of 1/n_dims each.
EXTREME_AXIS_FACTOR = 0.5


def joint_plausibility(theta_named, space: ThetaSpace) -> float:
    """Joint sim-to-real credibility of a world: prod_i (1 - (1 - p_i) * (1 - f)).

    ``transfer_likelihood`` averages the per-dim plausibilities, so on a 16-dim space one
    axis pinned at its worst bound costs only 1/16: an ice-rink floor scores 0.94, and
    ice + 6 kg payload + 80 ms latency + 50% dropout + a 2 m/s shove still scores 0.69.
    An adversarial search weighted by that mean drifts into exactly those compound
    corners. Here every axis at its bound multiplies by ``EXTREME_AXIS_FACTOR`` (0.5):
    one extreme axis 0.5, three 0.125, five 0.03. An axis partway past its band costs
    proportionally less; an axis inside its band costs nothing, so nominal stays 1.0.

    This is the search objective's weight and each record's ``transfer_score``. The
    plausibility gate (plausibility.py) keeps the mean, which its thresholds are
    calibrated against.
    """
    f = EXTREME_AXIS_FACTOR
    out = 1.0
    for name, dim in space.dims.items():
        if name in theta_named:
            p = _dim_plausibility(theta_named[name], dim.plausible, dim.bounds)
            out *= 1.0 - (1.0 - p) * (1.0 - f)
    return float(out)


def severity_from_score(failure_score: float) -> str:
    if failure_score >= 0.9: return "critical"
    if failure_score >= 0.7: return "high"
    if failure_score >= 0.45: return "medium"
    return "low"


def priority_key(severity: str, transfer: float, failure_score: float) -> float:
    return SEVERITY_WEIGHT[severity] * (0.5 * transfer + 0.5 * failure_score) + 0.01 * SEVERITY_RANK[severity]
