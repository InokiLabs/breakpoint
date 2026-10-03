"""GPU throughput settings that must be applied BEFORE jax is imported.

Two free levers, neither of which changes results:

  * ``--xla_gpu_triton_gemm_any=true`` -- XLA's Triton GEMM path; documented ~30% on
    NVIDIA GPUs for workloads like this one.
  * MJX ``jacobian="dense"`` -- 10-20% on GPU for models of this size (MJX docs); the
    default "auto" picks sparse above a DOF threshold that a quadruped sits near.

Both are no-ops on CPU, so importing this on a laptop costs nothing. Call
``tune_for_gpu()`` at the TOP of an entry script, before any ``import jax``:

    from breakpoint_mjx.accel import tune_for_gpu
    tune_for_gpu()
    import jax                       # noqa: E402

An already-set XLA_FLAGS is respected (appended to, never overwritten), so a caller
can still tune by hand or disable via BP_NO_XLA_TUNE=1.
"""
from __future__ import annotations

import os

_FLAGS = ("--xla_gpu_triton_gemm_any=true",)


def tune_for_gpu(flags=_FLAGS):
    """Append the GPU XLA flags to XLA_FLAGS unless disabled. Returns the value set."""
    if os.environ.get("BP_NO_XLA_TUNE") == "1":
        return os.environ.get("XLA_FLAGS", "")
    existing = os.environ.get("XLA_FLAGS", "")
    missing = [f for f in flags if f.split("=")[0] not in existing]
    if missing:
        os.environ["XLA_FLAGS"] = " ".join(([existing] if existing else []) + missing)
    return os.environ["XLA_FLAGS"]


def mjx_model_opts(model, jacobian="dense"):
    """Return ``model`` with the MJX jacobian option set, or unchanged if unavailable.

    Kept separate from tune_for_gpu because it applies to an already-built model and
    must not fire on a mujoco (non-MJX) model.
    """
    opt = getattr(model, "opt", None)
    if opt is None or not hasattr(opt, "jacobian"):
        return model
    try:
        import mujoco

        value = getattr(mujoco.mjtJacobian, f"mjJAC_{jacobian.upper()}")
        return model.replace(opt=opt.replace(jacobian=value))
    except Exception:
        # A model type that does not support .replace (plain mujoco.MjModel) or an
        # unknown jacobian name: leave it exactly as it was rather than half-apply.
        return model
