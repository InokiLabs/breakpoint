"""Reproducibility record: software versions, hardware, config and checkpoint hashes."""
from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from datetime import datetime, timezone

from benchmark.evaluation.setup import REPO, sha256_file


def _ver(mod):
    try:
        return getattr(importlib.import_module(mod), "__version__", "unknown")
    except Exception as e:  # noqa: BLE001
        return f"not importable ({type(e).__name__})"


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def collect(cfg, policy_hashes: dict, theta_hashes: dict) -> dict:
    import jax

    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        cpu = platform.processor()
    devices = [str(d) for d in jax.devices()]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": cpu,
        "jax_backend": jax.default_backend(),
        "jax_devices": devices,
        "gpu": next((d for d in devices if "cuda" in d.lower() or "gpu" in d.lower()), None),
        "versions": {m: _ver(m) for m in ("jax", "jaxlib", "mujoco", "mujoco.mjx", "numpy",
                                          "scipy", "evosax", "flax", "brax",
                                          "mujoco_playground")},
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config": {"path": cfg.source, "sha256": cfg.sha256()},
        "policy_checkpoints": policy_hashes,
        "theta_spaces": theta_hashes,
        "basis": "simulation only (MuJoCo MJX search; C MuJoCo replay); no physical robot",
    }


__all__ = ["collect", "sha256_file"]
