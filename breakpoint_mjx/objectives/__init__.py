"""Per-target FAILURE OBJECTIVE + SUMMARY contract (T7).

The engine used to hardcode ONE failure objective -- "stay upright" -- woven through
``rollout.rollout_one`` (the summary dict), ``search.scores_from_trajectories``,
``failure.component_signals`` / ``failure_score`` / ``behavior_descriptor``, the atlas,
margins, and crossval. That made every target implicitly a floating-base LEGGED
locomotion policy.

An ``Objective`` bundles the four pluggable hooks a target's *class* provides, so the
same search / eval / render / atlas / margins / crossval machinery can drive a
different vertical (manipulation, ...) without touching the core:

  * ``summarize(state, consts) -> dict``       -- per-step SUMMARY from the sim state
                                                  (run INSIDE ``rollout_one`` under
                                                  jax.vmap; pure array ops, no jax import).
  * ``component_signals(traj, thresholds) -> dict``   -- pure-numpy per-rollout signals.
  * ``failure_score(signals, horizon) -> float``      -- higher == more broken.
  * ``behavior_descriptor(traj, thresholds, horizon, sig=None) -> (mode_idx, ttf_norm)``
                                                  -- the 2D MAP-Elites atlas axes, over
                                                  this objective's ``failure_modes``.
  * ``rollout_signature(traj) -> list[float]``  -- fixed-length clustering signature.

The LEGGED implementations (``breakpoint_mjx.failure`` + the former inline rollout
summary) are WRAPPED verbatim as ``objectives.legged.LEGGED_OBJECTIVE`` -- the callables
stored on it ARE ``failure.component_signals`` etc., so go1/barkour/g1 route through it
with ZERO behavior change (byte-identical scores/signals/descriptors). A new vertical
adds an ``Objective`` module + a target that points at it.

Import-safe: this module + ``legged`` + ``manipulation`` import only numpy/stdlib at
top (jax is never needed to *describe* an objective; ``summarize`` uses whatever array
type the traced state carries), so importing them on a CPU-only box is safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Objective:
    """The per-class failure objective + summary contract (see module docstring).

    Fields
    ------
    name                : short label, e.g. "legged" / "manipulation".
    summarize           : (state, consts) -> dict. Builds the per-step summary from the
                          sim state INSIDE ``rollout_one`` (under jax.vmap). ``consts``
                          is the target's own resolve_constants() blob (opaque to the
                          core); each objective interprets it. Must use only plain array
                          ops (no jax import) so the module stays CPU-importable.
    component_signals   : (traj, thresholds) -> dict. Pure-numpy per-rollout signals.
    failure_score       : (signals, horizon) -> float. Higher == more broken.
    behavior_descriptor : (traj, thresholds, horizon, sig=None) -> (mode_idx, ttf_norm).
    failure_modes       : ordered tuple of mode names; ``mode_idx`` indexes into it. The
                          atlas grid width == len(failure_modes).
    rollout_signature   : (traj) -> list[float]. Fixed-length signature for clustering.
    contract_keys       : the keys the summary/traj dict carries (documentation + a light
                          structural check for tests); not consumed by the core loop.
    nominal_performance : (signals: list[dict], thresholds) -> dict with at least
                          ``{"nominal_reward": float in [0, 1]}`` (HIGHER better), the
                          per-class "performance under the NOMINAL config" aggregate used
                          as the performance axis of the hardening Pareto
                          (the historical nominal-reward proxy). Takes the LIST of per-rollout
                          ``component_signals`` dicts so each objective owns its own
                          aggregation (the legged impl reproduces the historical
                          tracking*(1-fall_rate) math byte-for-byte). OUT-OF-CONTRACT:
                          the signals must come from THIS objective's component_signals
                          (callers always route the active target's own hook).
    """

    name: str
    summarize: Callable
    component_signals: Callable
    failure_score: Callable
    behavior_descriptor: Callable
    failure_modes: tuple
    rollout_signature: Callable
    contract_keys: tuple = ()
    nominal_performance: Callable = None


# Imported at the bottom so ``Objective`` is defined before the objective modules
# reference it (mirrors the targets package's registry pattern; no circular hazard
# because the objective modules import only ``Objective`` + numpy, never the core loop).
from breakpoint_mjx.objectives.legged import LEGGED_OBJECTIVE  # noqa: E402,F401
