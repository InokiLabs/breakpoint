"""Crash-proof pipeline plumbing: stage checkpoints, atomic writes, memory preflight.

Why this exists: on a CPU-only laptop the demo can die natively (LLVM "Unable to
allocate section memory", then MemoryError / OpenBLAS on a retry) after the expensive
evidence stages but before report.html. Before this module a crash there left nothing:
no partial report, no resume. Now

* each evidence stage (``cluster`` -> ``cmc`` -> ``replay``) writes
  ``<OUT>/stage_<name>.json`` atomically (tmp file then ``os.replace``), keyed by a hash
  of failures.json plus the budget knobs that change what the stage computes; a rerun
  loads the latest valid stage and skips the work (``resuming from <stage>``);
* ``free_memory()`` reports free physical RAM and commit (pagefile) headroom, so
  ``breakpoint doctor`` and ``breakpoint run`` can warn before a run that will not fit,
  and the pipeline can drop to low-memory mode (no mp4 rendering) on its own.

Deliberately light: stdlib only at import time (the CLI imports it without JAX).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import tempfile

STAGES = ("cluster", "cmc", "replay")       # in pipeline order; later = more work saved
STAGE_VERSION = 1

# Measured demo peak on the development box (Windows 11 Pro, i7-1360P, 16 GB RAM, jax
# CPU backend), 2026-09-27, sampled every 0.5 s by scripts/measure_demo_memory.py, which
# sums RSS and private commit over the CLI process tree (CLI + pipeline child). The
# largest of three rendering runs of the smoke demo (4118 / 4442 / 4582 MB RSS; 5994 /
# 6030 / 6484 MB commit); every peak fell in the render phase. `doctor` and `run` warn
# below these.
DEMO_PEAK_RSS_MB = 4582
DEMO_PEAK_COMMIT_MB = 6484
# The same demo with --low-mem (no mp4 rendering) peaked at 1577-1776 MB RSS over three
# measured runs (one of them crashed in the replay, out of memory), so rendering adds
# about 4582 - 1776 = ~2800 MB of resident memory, rounded up to 3000. Unset BP_LOWMEM,
# the pipeline skips rendering on its own when free physical RAM at the start of the
# render phase is below that.
LOWMEM_AUTO_MB = 3000


# --------------------------------------------------------------------------
# atomic writes + stage checkpoints
# --------------------------------------------------------------------------

def _jsonable(o):
    """numpy scalars/arrays (and anything else odd) inside a stage payload."""
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def write_json_atomic(path, obj):
    """Write ``obj`` as JSON to ``path`` so a crash mid-write never leaves a torn file:
    write a tmp file in the same directory, flush + fsync, then ``os.replace``."""
    path = os.fspath(path)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, default=_jsonable)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def replace_atomic(tmp, path):
    """Move a finished tmp file over ``path`` (same directory) in one step."""
    os.replace(os.fspath(tmp), os.fspath(path))
    return path


def run_key(failures_path, budget):
    """Hash of the failures.json bytes plus the budget knobs (any JSON-able dict).

    A different search result or a different budget (CMC seeds, replay seeds, horizon,
    number of families ...) gives a different key, so a stale stage is never reused."""
    h = hashlib.sha256()
    with open(failures_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    h.update(b"\0budget\0")
    h.update(json.dumps(budget, sort_keys=True, default=str).encode("utf-8"))
    h.update(f"\0v{STAGE_VERSION}".encode())
    return h.hexdigest()[:24]


def stage_path(out_dir, name):
    return os.path.join(os.fspath(out_dir), f"stage_{name}.json")


def save_stage(out_dir, name, key, families, extra=None):
    """Checkpoint one stage. ``families`` are FailureFamily objects (or dicts)."""
    if name not in STAGES:
        raise ValueError(f"unknown stage {name!r}; expected one of {STAGES}")
    payload = {"stage": name, "key": key, "version": STAGE_VERSION,
               "families": families_to_dicts(families), "extra": extra or {}}
    return write_json_atomic(stage_path(out_dir, name), payload)


def load_stage(out_dir, name, key):
    """The stage payload if it exists, parses and matches ``key``; else None."""
    p = stage_path(out_dir, name)
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("stage") != name or d.get("key") != key \
            or d.get("version") != STAGE_VERSION or not isinstance(d.get("families"), list):
        return None
    return d


def load_latest(out_dir, key):
    """(name, payload) of the furthest valid stage for ``key``, else (None, None)."""
    for name in reversed(STAGES):
        d = load_stage(out_dir, name, key)
        if d is not None:
            return name, d
    return None, None


def clear_stages(out_dir):
    for name in STAGES:
        try:
            os.remove(stage_path(out_dir, name))
        except OSError:
            pass


def families_to_dicts(families):
    return [f if isinstance(f, dict) else dataclasses.asdict(f) for f in families]


def families_from_dicts(rows):
    from breakpoint_mjx.schemas import FailureFamily

    known = {f.name for f in dataclasses.fields(FailureFamily)}
    return [FailureFamily(**{k: v for k, v in r.items() if k in known}) for r in rows]


def crash_point(name, environ=None):
    """Test hook: ``BP_TEST_CRASH_AT=<name>`` kills this process natively at ``name``,
    the way an LLVM/OpenBLAS allocation failure does (no Python cleanup, no finally)."""
    env = os.environ if environ is None else environ
    if env.get("BP_TEST_CRASH_AT") != name:
        return
    print(f"BP_TEST_CRASH_AT={name}: simulating a native crash", flush=True)
    sys.stdout.flush()
    if sys.platform == "win32":
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k32.TerminateProcess(k32.GetCurrentProcess(), 0xC0000005)  # access violation
    os._exit(134)


# --------------------------------------------------------------------------
# memory preflight
# --------------------------------------------------------------------------

def _win_memstatus():
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return st


def free_memory():
    """{"free_ram_mb", "total_ram_mb", "free_commit_mb", "commit_limit_mb", "source"}.

    Free RAM is psutil's ``available`` when psutil is installed. Commit headroom (what a
    Windows allocation actually needs: RAM plus pagefile) comes from GlobalMemoryStatusEx
    on Windows; elsewhere it is available RAM plus free swap. Values are None when no
    source works."""
    mb = 1 << 20
    out = {"free_ram_mb": None, "total_ram_mb": None, "free_commit_mb": None,
           "commit_limit_mb": None, "source": None}
    try:
        import psutil

        vm = psutil.virtual_memory()
        out.update(free_ram_mb=vm.available // mb, total_ram_mb=vm.total // mb,
                   source="psutil")
        sw = psutil.swap_memory()
        out.update(free_commit_mb=(vm.available + sw.free) // mb,
                   commit_limit_mb=(vm.total + sw.total) // mb)
    except Exception:  # noqa: BLE001 -- psutil missing or unsupported here
        pass
    if sys.platform == "win32":
        try:
            st = _win_memstatus()
            if st is not None:
                if out["free_ram_mb"] is None:
                    out.update(free_ram_mb=st.ullAvailPhys // mb,
                               total_ram_mb=st.ullTotalPhys // mb,
                               source="GlobalMemoryStatusEx")
                out.update(free_commit_mb=st.ullAvailPageFile // mb,
                           commit_limit_mb=st.ullTotalPageFile // mb)
        except Exception:  # noqa: BLE001
            pass
    elif out["free_ram_mb"] is None:
        try:
            info = {}
            with open("/proc/meminfo") as fh:
                for line in fh:
                    k, v = line.split(":", 1)
                    info[k] = int(v.split()[0]) * 1024
            out.update(free_ram_mb=info["MemAvailable"] // mb,
                       total_ram_mb=info["MemTotal"] // mb,
                       free_commit_mb=(info["MemAvailable"] + info.get("SwapFree", 0)) // mb,
                       source="/proc/meminfo")
        except Exception:  # noqa: BLE001
            pass
    return out


def memory_warning(mem=None):
    """One warning line when free RAM or commit is below the measured demo peak, else
    None (also None when memory cannot be read)."""
    mem = mem or free_memory()
    ram, com = mem.get("free_ram_mb"), mem.get("free_commit_mb")
    low = []
    if ram is not None and ram < DEMO_PEAK_RSS_MB:
        low.append(f"free RAM {ram} MB < {DEMO_PEAK_RSS_MB} MB")
    if com is not None and com < DEMO_PEAK_COMMIT_MB:
        low.append(f"free commit {com} MB < {DEMO_PEAK_COMMIT_MB} MB")
    if not low:
        return None
    return ("warning: " + ", ".join(low) + " (the measured CPU smoke-demo peak). The run "
            "may crash natively; close other programs, or pass --low-mem to skip mp4 "
            "rendering. A crash still leaves a resumable run (stage checkpoints).")


def lowmem_enabled(environ=None, mem=None):
    """(enabled, reason). ``BP_LOWMEM=1`` forces it on, ``BP_LOWMEM=0`` off; unset, it
    turns on automatically when free physical RAM is below ``LOWMEM_AUTO_MB``."""
    env = os.environ if environ is None else environ
    v = str(env.get("BP_LOWMEM", "")).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True, "BP_LOWMEM=1"
    if v in ("0", "false", "no", "off"):
        return False, "BP_LOWMEM=0"
    mem = mem or free_memory()
    ram = mem.get("free_ram_mb")
    if ram is not None and ram < LOWMEM_AUTO_MB:
        return True, f"auto: free RAM {ram} MB < {LOWMEM_AUTO_MB} MB"
    return False, "auto: enough free RAM"
