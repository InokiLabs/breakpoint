"""Backend seam (T1): pluggable re-simulation engines for cross-config validity.

Two backends re-simulate a recorded failure and return the SAME per-step summary keys
`failure.component_signals` consumes, so their reproduction is directly comparable:

  * `MjxBackend`     -- re-sim via the current MJX `rollout_one` on a config-varied
                        model (same-backend config variation; primary, cheap).
  * `MujocoCBackend` -- re-sim on the standard C MuJoCo engine, host CPU (a genuinely
                        independent implementation; stronger evidence). Reset / obs /
                        command / ctrl parity with the MJX env is ported and measured
                        (tests/test_mujoco_c_parity.py). OPTIONAL: needs the `mujoco`
                        python bindings, so it is import-guarded.

`MjxBackend` is import-safe here (it defers jax to its methods). `MujocoCBackend` is
imported lazily by `make_mujoco_c_backend` so importing this package never requires
`mujoco`.
"""
from __future__ import annotations

from breakpoint_mjx.backends.mjx_backend import MjxBackend

__all__ = ["MjxBackend", "make_mujoco_c_backend"]


def make_mujoco_c_backend(**kwargs):
    """Construct a `MujocoCBackend` iff it is READY (mujoco importable AND parity ported),
    else return None.

    Returns None -- not a half-working backend -- when `mujoco` is missing, when
    ``READY`` is off, or for a non-legged target, so validity runs on the MJX configs
    alone rather than being corrupted by a fabricated reproduction. Callers treat None as
    "skip the independent-backend evidence entry". Parity is ported and measured
    (tests/test_mujoco_c_parity.py), so this returns a real backend wherever the bindings
    are present; the READY gate stays as the switch that keeps an unverified port off the
    validity vector.
    """
    from breakpoint_mjx.backends.mujoco_c import MujocoCBackend

    # Legged-only (T7): the C backend hardcodes the legged consts tuple +
    # apply_legged_theta_numpy + stay-upright summary, so skip it for a non-legged target
    # rather than feed it a manipulation consts dict (a cryptic unpack error later).
    target = kwargs.get("target")
    obj = getattr(target, "objective", None)
    if obj is not None and getattr(obj, "name", "legged") != "legged":
        return None

    if not MujocoCBackend.ready():
        return None
    return MujocoCBackend(**kwargs)
