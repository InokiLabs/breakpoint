"""Direction-aware family labels + dominant-deviating-dim ranking.

Pure numpy (NO sklearn / jax): factored out of ``clustering`` so lightweight consumers
-- notably ``regression`` on the CI ``--families`` path -- can reuse the discriminative
naming logic WITHOUT dragging in scikit-learn (which ``clustering`` imports at module top
for KMeans). ``clustering`` re-imports these so its public API is unchanged.
"""
from __future__ import annotations

import numpy as np

from breakpoint_mjx.config import ThetaSpace

# direction-aware family labels keyed by (dim, "high"|"low")
_LABELS = {
    ("friction_floor", "low"): "low-friction slip",
    ("friction_floor", "high"): "high-friction grip",
    ("torso_mass_add_kg", "high"): "payload overload",
    ("torso_mass_add_kg", "low"): "underweight trunk",
    ("com_offset_x_m", "high"): "forward CoM shift",
    ("com_offset_x_m", "low"): "rear CoM shift",
    ("com_offset_y_m", "high"): "lateral CoM shift",
    ("com_offset_y_m", "low"): "lateral CoM shift",
    ("kp_scale", "high"): "over-stiff actuators",
    ("kp_scale", "low"): "under-stiff actuators",
    ("kd_scale", "high"): "over-damped actuators",
    ("kd_scale", "low"): "under-damped actuators",
    ("obs_noise_scale", "high"): "sensor-noise drift",
    ("obs_noise_scale", "low"): "clean-sensor edge case",
    ("latency_steps", "high"): "latency gait collapse",
    ("latency_steps", "low"): "low-latency edge case",
}


def _label(name, direction):
    return _LABELS.get((name, direction), f"{direction} {name.replace('_', ' ')}")


def _reference(all_records, space: ThetaSpace):
    """Reference means: global failure mean if records given, else per-dim nominal."""
    if all_records is not None:
        return {n: float(np.mean([r.theta_named[n] for r in all_records])) for n in space.names()}
    return {n: space.dims[n].nominal for n in space.names()}


def _ranked_dims(member_records, ref, space: ThetaSpace):
    """[(dev, name, direction), ...] sorted by descending |family_mean - ref| / bound-span."""
    fmean = {n: float(np.mean([r.theta_named[n] for r in member_records])) for n in space.names()}
    out = []
    for name, dim in space.dims.items():
        span = max(dim.bounds[1] - dim.bounds[0], 1e-9)
        dev = abs(fmean[name] - ref[name]) / span
        out.append((dev, name, "high" if fmean[name] > ref[name] else "low"))
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def family_mean_theta(member_records, space: ThetaSpace) -> dict:
    """The family's per-dim MEAN theta (the point the displayed name is derived from)."""
    return {
        n: float(np.mean([r.theta_named[n] for r in member_records])) for n in space.names()
    }


def label_from_cause(cause) -> str:
    """Family label from a CERTIFIED minimal cause (breakpoint_mjx/mincause.py).

    ``cause`` is ``[(dim_name, "high"|"low"), ...]`` where each direction is the sign of
    (minimal counterexample value - nominal), so the label can never contradict the values it
    is shown next to. Conjunctive causes join with " + " (the same joiner the depth labels
    of ``clustering.build_families`` use); an empty cause means the policy fails at nominal.
    """
    if not cause:
        return "fails at nominal"
    return " + ".join(_label(name, direction) for name, direction in cause)
