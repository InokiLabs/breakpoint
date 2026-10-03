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


def severity_from_score(failure_score: float) -> str:
    if failure_score >= 0.9: return "critical"
    if failure_score >= 0.7: return "high"
    if failure_score >= 0.45: return "medium"
    return "low"


def priority_key(severity: str, transfer: float, failure_score: float) -> float:
    return SEVERITY_WEIGHT[severity] * (0.5 * transfer + 0.5 * failure_score) + 0.01 * SEVERITY_RANK[severity]
