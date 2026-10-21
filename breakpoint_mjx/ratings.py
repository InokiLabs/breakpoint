"""BreakPoint Crash Ratings: a fixed, versioned battery of named conditions, run identically
on any policy, so results are comparable across policies (a crash-test protocol).

This is the OPPOSITE of the search: nothing adapts to the policy. A battery
(``breakpoint_mjx/batteries/<id>.yaml``) lists named cells, each a partial theta over the
robot's theta space; every cell runs every fixed seed through the same batched evaluator
the search uses, and each rollout is scored with the same fall signal
(``objectives/legged.py`` floor-relative height + ``failure.component_signals``).

    breakpoint rate --robot go1 --policy ./go1_policy.pkl --battery legged-quadruped-v1 \\
        --out report.json
    breakpoint rate --robot go1 --policy ./go1_policy.pkl --budget smoke   # a preview

The report follows the shared contract ``breakpoint.report/v1`` (kind "rating"). Signing is
done elsewhere: ``attestation`` is always written as null here.

Determinism: the rollout for (cell i, seed s) uses batch key ``PRNGKey(s)`` and row ``i``,
i.e. ``fold_in(PRNGKey(s), i)`` (rollout.env_key), so it depends only on the battery (cell
order is part of the hashed battery) and the seed, never on batch size or machine.

Accounting choices (stated in docs/ratings.md):
  * a rollout the physical-validity gate flags ``sim_invalid`` (simulator artifact) is
    COUNTED AS A FALL, conservatively, and the per-cell ``sim_invalid`` count is reported
    beside it, so a rating can never look better because the simulator misbehaved;
  * the fall thresholds come from the battery's ``fall_rule`` sentence (parsed), not from
    the target, so every policy on one battery is judged by the same rule.

Pure numpy + pyyaml at import time; jax / the simulator load only inside ``rate``.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from breakpoint_mjx.obs_adversary import OBS_PERTURBATION_MODEL

SCHEMA = "breakpoint.report/v1"
BATTERY_DIR = Path(__file__).resolve().parent / "batteries"

# Robots with a declared rating theta space, and the battery class each one belongs to.
# A robot outside this map is refused: its fall threshold / mass scale is not covered by
# any battery's fall_rule and payload levels.
ROBOT_CLASS = {
    "go1": "legged-quadruped",
    "barkour": "legged-quadruped",
    "g1": "legged-humanoid",
}

# --budget smoke: a plumbing preview, NOT a rating. Fewer seeds, shorter horizon, and the
# report's run block + summary label say so.
SMOKE_SEEDS = 4
SMOKE_HORIZON = 150
BUDGETS = ("full", "smoke")

_REQUIRED_BATTERY_KEYS = ("battery", "version", "robot_class", "horizon", "seeds",
                          "fall_rule", "cells", "pass_rule")


class BatteryError(ValueError):
    """A battery file, or a battery/robot combination, that cannot be rated."""


@dataclass(frozen=True)
class Battery:
    id: str
    version: int
    robot_class: str
    horizon: int
    seeds: tuple
    fall_rule: str
    pass_rule: str
    cells: tuple            # ((cell_id, {dim: value}), ...) in file order
    sha256: str
    fall_height_m: float
    up_proj_min: float
    pass_num: int
    pass_den: int


# --------------------------------------------------------------------------
# loading + hashing
# --------------------------------------------------------------------------

def battery_path(name_or_path) -> Path:
    """A shipped battery id (``legged-quadruped-v1``) or a path to a battery YAML."""
    p = Path(str(name_or_path))
    if p.suffix.lower() in (".yaml", ".yml") or p.exists():
        if not p.exists():
            raise BatteryError(f"no battery file at {p}")
        return p
    cand = BATTERY_DIR / f"{name_or_path}.yaml"
    if not cand.exists():
        known = sorted(q.stem for q in BATTERY_DIR.glob("*.yaml"))
        raise BatteryError(f"unknown battery {name_or_path!r}; shipped batteries: {known}")
    return cand


def hash_battery_bytes(data: bytes) -> str:
    """sha256 of the battery YAML with line endings normalized to LF, so a Windows
    checkout (autocrlf) and a Linux checkout of the same file hash identically."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


_NUM = r"([0-9]*\.?[0-9]+)"


def parse_fall_rule(rule: str):
    """(fall_height_m, up_proj_min) from a fall_rule sentence. Both must be stated."""
    h = re.search(r"height[^<]*<\s*" + _NUM + r"\s*m\b", rule)
    u = re.search(r"up[^<]*<\s*" + _NUM, rule)
    if not h or not u:
        raise BatteryError(
            f"fall_rule must state both thresholds, e.g. 'trunk height above the floor "
            f"< 0.18 m, or trunk up-axis z-projection < 0.5'; got {rule!r}")
    return float(h.group(1)), float(u.group(1))


def parse_pass_rule(rule: str):
    """(num, den) from 'a cell passes if fall rate <= num/den across seeds'."""
    m = re.search(r"fall rate\s*<=\s*(\d+)\s*/\s*(\d+)", rule)
    if not m or int(m.group(2)) <= 0:
        raise BatteryError(f"pass_rule must read 'fall rate <= k/n'; got {rule!r}")
    return int(m.group(1)), int(m.group(2))


def load_battery(name_or_path) -> Battery:
    path = battery_path(name_or_path)
    data = path.read_bytes()
    raw = yaml.safe_load(data.decode("utf-8"))
    if not isinstance(raw, dict):
        raise BatteryError(f"{path}: not a YAML mapping")
    missing = [k for k in _REQUIRED_BATTERY_KEYS if k not in raw]
    if missing:
        raise BatteryError(f"{path}: missing keys {missing}")
    seeds = raw["seeds"]
    if (not isinstance(seeds, list) or not seeds
            or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise BatteryError(f"{path}: seeds must be a non-empty list of distinct integers")
    horizon = raw["horizon"]
    if not isinstance(horizon, int) or horizon <= 0:
        raise BatteryError(f"{path}: horizon must be a positive integer")
    cells, seen = [], set()
    for c in raw["cells"] or []:
        if not isinstance(c, dict) or "id" not in c:
            raise BatteryError(f"{path}: every cell needs an id")
        cid = str(c["id"])
        if cid in seen:
            raise BatteryError(f"{path}: duplicate cell id {cid!r}")
        seen.add(cid)
        theta = c.get("theta") or {}
        if not isinstance(theta, dict):
            raise BatteryError(f"{path}: cell {cid!r} theta must be a mapping")
        cells.append((cid, {str(k): float(v) for k, v in theta.items()}))
    if not cells:
        raise BatteryError(f"{path}: no cells")
    fh, up = parse_fall_rule(str(raw["fall_rule"]))
    num, den = parse_pass_rule(str(raw["pass_rule"]))
    return Battery(
        id=str(raw["battery"]), version=int(raw["version"]),
        robot_class=str(raw["robot_class"]), horizon=horizon, seeds=tuple(seeds),
        fall_rule=str(raw["fall_rule"]), pass_rule=str(raw["pass_rule"]),
        cells=tuple(cells), sha256=hash_battery_bytes(data),
        fall_height_m=fh, up_proj_min=up, pass_num=num, pass_den=den)


def check_robot(battery: Battery, robot: str) -> None:
    cls = ROBOT_CLASS.get(robot)
    if cls is None:
        raise BatteryError(f"no rating battery covers robot {robot!r}; rated robots: "
                           f"{sorted(ROBOT_CLASS)}")
    if cls != battery.robot_class:
        raise BatteryError(f"battery {battery.id} is for {battery.robot_class}, but "
                           f"{robot} is {cls}")


def default_battery(robot: str) -> str:
    cls = ROBOT_CLASS.get(robot)
    if cls is None:
        raise BatteryError(f"no rating battery covers robot {robot!r}; rated robots: "
                           f"{sorted(ROBOT_CLASS)}")
    return f"{cls}-v1"


# --------------------------------------------------------------------------
# cells -> theta vectors
# --------------------------------------------------------------------------

def cell_thetas(battery: Battery, space, require_plausible: bool = True) -> np.ndarray:
    """(n_cells, D) theta matrix in the space's vector order; unlisted dims at nominal.

    Raises BatteryError for an unknown dim, a value outside the hard bounds, a value
    outside the plausible band (``require_plausible``), or a non-integer latency."""
    names = space.names()
    nominal = np.asarray(space.nominal_vector(), dtype=float)
    rows, errs = [], []
    for cid, theta in battery.cells:
        row = nominal.copy()
        for dim, v in theta.items():
            if dim not in space.dims:
                errs.append(f"cell {cid!r}: unknown theta dim {dim!r} (this robot's "
                            f"dims: {names})")
                continue
            d = space.dims[dim]
            lo, hi = d.bounds
            if not np.isfinite(v) or v < lo - 1e-9 or v > hi + 1e-9:
                errs.append(f"cell {cid!r}: {dim}={v} outside bounds [{lo}, {hi}]")
                continue
            plo, phi = d.plausible
            if require_plausible and (v < plo - 1e-9 or v > phi + 1e-9):
                errs.append(f"cell {cid!r}: {dim}={v} outside plausible band "
                            f"[{plo}, {phi}] (ratings use deployment-level values only)")
                continue
            if "latency" in dim and abs(v - round(v)) > 1e-9:
                errs.append(f"cell {cid!r}: {dim}={v} must be a whole number of steps")
                continue
            row[names.index(dim)] = v
        rows.append(row)
    if errs:
        raise BatteryError("battery does not fit this theta space:\n  " + "\n  ".join(errs))
    return np.stack(rows)


# --------------------------------------------------------------------------
# statistics + per-cell summary
# --------------------------------------------------------------------------

def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact two-sided (1 - alpha) Clopper-Pearson interval (riskcert's implementation)."""
    from breakpoint_mjx.riskcert import clopper_pearson as _cp

    return _cp(int(k), int(n), alpha)


def cell_passed(falls: int, n: int, num: int, den: int) -> bool:
    """falls / n <= num / den, compared exactly in integers. n == 0 never passes."""
    return n > 0 and falls * den <= num * n


def summarize_cell(cell_id, signals, battery: Battery) -> dict:
    n = len(signals)
    invalid = sum(1 for s in signals if float(s.get("sim_invalid", 0.0)))
    falls = sum(1 for s in signals
                if float(s.get("fall", 0.0)) or float(s.get("sim_invalid", 0.0)))
    lo, hi = clopper_pearson(falls, n)
    return {"id": cell_id, "n": n, "falls": falls,
            "fall_rate": round(falls / n, 6) if n else 0.0,
            # widened outward to 4 decimals, so rounding never narrows the interval
            "ci95": [math.floor(lo * 1e4) / 1e4, min(1.0, math.ceil(hi * 1e4) / 1e4)],
            "passed": cell_passed(falls, n, battery.pass_num, battery.pass_den),
            "sim_invalid": invalid}


def run_battery(battery: Battery, thetas: np.ndarray, run_fn, seeds) -> list:
    """Run every cell x seed. ``run_fn(thetas, seed)`` evaluates all cells for one seed
    (one batch) and returns one component_signals dict per row, in row order."""
    per_cell = [[] for _ in battery.cells]
    for s in seeds:
        sigs = list(run_fn(thetas, int(s)))
        if len(sigs) != len(battery.cells):
            raise RuntimeError(f"evaluator returned {len(sigs)} rows for "
                               f"{len(battery.cells)} cells")
        for i, sig in enumerate(sigs):
            per_cell[i].append(sig)
    return [summarize_cell(cid, per_cell[i], battery)
            for i, (cid, _) in enumerate(battery.cells)]


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def policy_file(spec: str) -> Path:
    """The checkpoint file behind a policy spec (strips onnx:/params:/brax: prefixes)."""
    s = str(spec)
    kind, sep, rest = s.partition(":")
    if sep and kind.lower() in ("onnx", "params", "npz", "weights", "brax") and len(kind) > 1:
        s = rest
    return Path(s)


def engine_commit():
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent),
                            "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        sha = r.stdout.strip()
        return sha if r.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else None
    except Exception:  # noqa: BLE001 - no git / not a checkout: commit is unknown
        return None


def summary_label(cells, budget, seeds, horizon) -> str:
    passed = sum(1 for c in cells if c["passed"])
    core = f"{passed}/{len(cells)} cells passed"
    if budget == "smoke":
        return (f"SMOKE PREVIEW, not a rating: {core} ({len(seeds)} seeds, "
                f"{horizon} steps)")
    return core


def build_report(*, battery: Battery, robot, policy, cells, backend, seeds, horizon,
                 budget="full", runner="self", engine_version=None, commit=None,
                 date=None) -> dict:
    if budget not in BUDGETS:
        raise ValueError(f"budget must be one of {BUDGETS}, got {budget!r}")
    if engine_version is None:
        from breakpoint_mjx.cli import __version__ as engine_version
    passed = sum(1 for c in cells if c["passed"])
    return {
        "schema": SCHEMA,
        "kind": "rating",
        "battery": {"id": battery.id, "sha256": battery.sha256},
        # obs_model: which observation-adversary semantics produced the cells (ratings
        # under different models are not comparable even on the same battery sha).
        "engine": {"version": str(engine_version), "commit": commit,
                   "obs_model": OBS_PERTURBATION_MODEL},
        "robot": robot,
        "policy": dict(policy),
        "run": {"date": date or _dt.datetime.now(_dt.timezone.utc).isoformat(
                    timespec="seconds"),
                "backend": backend, "seeds": list(seeds), "runner": runner,
                "budget": budget, "horizon": int(horizon)},
        "cells": list(cells),
        "summary": {"cells_passed": passed, "cells_total": len(cells),
                    "label": summary_label(cells, budget, seeds, horizon)},
        "gate": None,
        "attestation": None,
    }


_TOP_KEYS = ("schema", "kind", "battery", "engine", "robot", "policy", "run", "cells",
             "summary", "gate", "attestation")
_CELL_KEYS = ("id", "n", "falls", "fall_rate", "ci95", "passed")


def validate_report(rep: dict) -> list:
    """Contract problems in a rating report (empty list = valid)."""
    p = []
    for k in _TOP_KEYS:
        if k not in rep:
            p.append(f"missing top-level key {k!r}")
    if p:
        return p
    if rep["schema"] != SCHEMA:
        p.append(f"schema must be {SCHEMA!r}")
    if rep["kind"] not in ("rating", "gate"):
        p.append("kind must be rating or gate")
    for blk, keys in (("battery", ("id", "sha256")), ("engine", ("version", "commit")),
                      ("policy", ("name", "source", "sha256")),
                      ("run", ("date", "backend", "seeds", "runner")),
                      ("summary", ("cells_passed", "cells_total", "label"))):
        for k in keys:
            if k not in (rep[blk] or {}):
                p.append(f"missing {blk}.{k}")
    if rep["run"].get("backend") not in ("mjx-gpu", "mjx-cpu"):
        p.append("run.backend must be mjx-gpu or mjx-cpu")
    if rep["run"].get("runner") not in ("self", "inoki"):
        p.append("run.runner must be self or inoki")
    for c in rep["cells"]:
        for k in _CELL_KEYS:
            if k not in c:
                p.append(f"cell {c.get('id')!r} missing {k!r}")
        if all(k in c for k in _CELL_KEYS):
            if not (0 <= c["falls"] <= c["n"]):
                p.append(f"cell {c['id']!r}: falls outside [0, n]")
            lo, hi = c["ci95"]
            if not (0.0 <= lo <= c["fall_rate"] <= hi <= 1.0):
                p.append(f"cell {c['id']!r}: ci95 does not bracket fall_rate")
    if rep["summary"].get("cells_total") != len(rep["cells"]):
        p.append("summary.cells_total != number of cells")
    if rep["summary"].get("cells_passed") != sum(1 for c in rep["cells"] if c.get("passed")):
        p.append("summary.cells_passed does not match the cells")
    if rep["kind"] == "rating" and rep["gate"] is not None:
        p.append("a rating has gate null")
    return p


def format_table(rep: dict) -> str:
    rows = [("cell", "n", "falls", "fall_rate", "ci95", "result", "sim_invalid")]
    for c in rep["cells"]:
        rows.append((c["id"], str(c["n"]), str(c["falls"]), f"{c['fall_rate']:.3f}",
                     f"[{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]",
                     "PASS" if c["passed"] else "FAIL", str(c.get("sim_invalid", 0))))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(v.ljust(w[i]) for i, v in enumerate(r)) for r in rows]
    lines.insert(1, "  ".join("-" * x for x in w))
    head = (f"{rep['battery']['id']} on {rep['robot']} / {rep['policy']['name']} "
            f"[{rep['run']['backend']}, budget {rep['run'].get('budget', 'full')}]")
    return head + "\n" + "\n".join(lines) + "\n" + rep["summary"]["label"]


# --------------------------------------------------------------------------
# the real run
# --------------------------------------------------------------------------

def make_runner(target, env, policy_fn, space, horizon, thresholds):
    """``run_fn(thetas, seed)`` over the real batched evaluator (one compile, reused for
    every seed since the batch shape is fixed)."""
    import jax
    import jax.numpy as jnp

    from breakpoint_mjx.evaluator import make_evaluator

    consts = target.resolve_constants(env)
    evaluate = make_evaluator(env, policy_fn, space, horizon, target, consts)
    base_model = env._mjx_model

    def run(thetas, seed):
        try:
            traj = evaluate(jnp.asarray(thetas, dtype=jnp.float32),
                            jax.random.PRNGKey(int(seed)))
            host = {k: np.asarray(v) for k, v in traj.items()}
        finally:
            # rollout_one binds env._mjx_model inside the trace; restore the concrete
            # model (same discipline as check_policy.run_check).
            env._mjx_model = base_model
        n = thetas.shape[0]
        return [target.objective.component_signals({k: v[i] for k, v in host.items()},
                                                   thresholds) for i in range(n)]

    return run


def backend_name() -> str:
    import jax

    b = jax.default_backend()
    return "mjx-gpu" if b in ("gpu", "cuda", "rocm") else "mjx-cpu"


def rate(robot, policy_spec, battery_name=None, budget="full", opts=None,
         policy_name=None, policy_source="local", runner="self", log=print,
         _runner_factory=None, _backend=None) -> dict:
    """Run a battery on a policy and return the report dict (not written to disk).

    ``_runner_factory(target, env, policy_fn, space, horizon, thresholds) -> run_fn`` and
    ``_backend`` exist for tests (a mocked evaluator); production uses ``make_runner``."""
    if budget not in BUDGETS:
        raise ValueError(f"budget must be one of {BUDGETS}, got {budget!r}")
    battery = load_battery(battery_name or default_battery(robot))
    check_robot(battery, robot)
    ckpt = policy_file(policy_spec)
    if not ckpt.is_file():
        raise FileNotFoundError(f"no policy checkpoint file at {ckpt}")
    seeds = list(battery.seeds[:SMOKE_SEEDS]) if budget == "smoke" else list(battery.seeds)
    horizon = min(SMOKE_HORIZON, battery.horizon) if budget == "smoke" else battery.horizon

    from breakpoint_mjx.config import (load_search_config, load_theta_space,
                                       merge_target_thresholds)
    from breakpoint_mjx.paths import config_path
    from breakpoint_mjx.targets import get_target

    target = get_target(robot)
    space = load_theta_space(config_path(target.theta_space_path))
    thetas = cell_thetas(battery, space)
    cfg = merge_target_thresholds(load_search_config(config_path("search.yaml"), env={}),
                                  target)
    thresholds = dict(cfg.thresholds, trunk_height_min=battery.fall_height_m,
                      up_proj_min=battery.up_proj_min)

    if _runner_factory is None:
        from breakpoint_mjx.accel import tune_for_gpu

        tune_for_gpu()
        from breakpoint_mjx.check_policy import _load_env, load_policy

        opts = opts or {}
        env = _load_env(target, opts.get("obs_spec"))
        policy_fn, _norm, _map, label = load_policy(target, env, policy_spec, opts)
        factory = make_runner
    else:
        env, policy_fn, label, factory = None, None, f"mock {policy_spec}", _runner_factory
    tag = "SMOKE PREVIEW " if budget == "smoke" else ""
    log(f"rate: {tag}{battery.id} ({len(battery.cells)} cells x {len(seeds)} seeds, "
        f"{horizon} steps) on {robot} with {label}; first batch compiles ...")
    run_fn = factory(target, env, policy_fn, space, horizon, thresholds)
    cells = run_battery(battery, thetas, run_fn, seeds)
    return build_report(
        battery=battery, robot=robot,
        policy={"name": policy_name or ckpt.name, "source": policy_source or "local",
                "sha256": sha256_file(ckpt)},
        cells=cells, backend=_backend or backend_name(), seeds=seeds, horizon=horizon,
        budget=budget, runner=runner, commit=engine_commit())
