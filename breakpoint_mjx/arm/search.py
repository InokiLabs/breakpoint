"""Stress search for an AR4 program: find the conditions where it stops doing its job.

The same idea as the engine's policy search, applied to a programmed arm: the stress
space (``configs/ar4/theta_space.yaml``) is a search space, every candidate is a full
simulated run of the program, and the output is a set of failure families, each with the
most *plausible* failing condition, its minimal cause (which axes it really needs, and
the threshold on each) and a replay.

* ``run_search``: Sobol exploration, then a cross-entropy refinement around failing
  and near-failing worlds, weighted by plausibility so the search spends its budget near
  conditions a real installation can see. Families are named by failure mode and joint.
* ``minimal_cause``: reset axes to nominal one at a time (batched), keep only the ones
  the failure needs, then bisect each kept axis for its threshold.
* ``envelope``: a 2D sweep of two axes (default payload x acceleration), the safe
  operating envelope as a pass/fail map with the margin in every cell.
* ``endurance``: one condition over many cycles, the drift at each taught point.

Worlds run in lockstep batches (``sim.simulate_batch``) across a process pool.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from breakpoint_mjx.config import ThetaDim, ThetaSpace, load_theta_space
from breakpoint_mjx.paths import config_path
from breakpoint_mjx.plausibility import implausibility
from breakpoint_mjx.arm.sim import MODE_LABEL, simulate, simulate_batch, taught_offsets

BUDGETS = {"quick": 48, "standard": 160, "deep": 480}
CHUNK = 8


def _recentre(dim: ThetaDim, value: float) -> ThetaDim:
    lo, hi = dim.plausible
    return replace(dim, nominal=float(value), plausible=(min(lo, value), max(hi, value)))


def stress_space(prog, path=None, world_model=None) -> ThetaSpace:
    """The AR4 stress space, centred on this program and robot: the payload nominal is the
    program's own, and with a fitted world model the drive-train axes are centred on the
    calibrated values and ``world_model_scale`` (0 = raw simulator, 1 = calibrated) joins
    the space, exactly as the engine's rollout axis does."""
    from breakpoint_mjx.arm.robots import get_robot
    sp = load_theta_space(path or config_path(get_robot(prog.model).stress_space))
    dims = dict(sp.dims)
    dims["payload_kg"] = _recentre(dims["payload_kg"], prog.payload_kg)
    if world_model is not None:
        for n, v in world_model.params.items():
            if n in dims:
                dims[n] = _recentre(dims[n], v)
        dims["world_model_scale"] = ThetaDim("world_model_scale", "rollout", (0.0, 1.5), 1.0,
                                             (0.5, 1.25))
    return ThetaSpace(dims)


# ------------------------------------------------------------------------- execution
_W = {}


def _worker_init():
    from breakpoint_mjx.arm.program import Kinematics
    _W["kin"] = {}
    _W["Kinematics"] = Kinematics


def _worker_run(args):
    prog, chunk, wm, taught, keep = args
    kin = _W["kin"].get(prog.model)
    if kin is None:
        kin = _W["kin"][prog.model] = _W["Kinematics"](prog.model)
    res = simulate_batch(prog, chunk, kin=kin, world_model=wm, taught=taught,
                         keep_trace=bool(keep))
    for r in res:
        tr = r.pop("trace", None)
        if keep and tr:
            r["trace"] = {k: tr[k] for k in keep}
    return res


class Runner:
    """Evaluates lists of stress dicts on a process pool (or inline with workers=0)."""

    def __init__(self, workers: int | None = None):
        n = os.cpu_count() or 2
        self.workers = max(0, n - 1 if workers is None else workers)
        self._pool = None
        self._taught = {}

    def taught(self, prog, world_model=None):
        """Taught-point offsets of the nominal run, computed once per program/world model."""
        key = (id(prog), id(world_model))
        if key not in self._taught:
            self._taught[key] = taught_offsets(prog, world_model=world_model)
        return self._taught[key]

    def __enter__(self):
        if self.workers > 0:
            self._pool = ProcessPoolExecutor(self.workers, initializer=_worker_init)
        else:
            _worker_init()
        return self

    def __exit__(self, *exc):
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)

    def run(self, prog, stresses, world_model=None, keep=None):
        """``keep``: trace keys to return (e.g. ("t", "q_enc")); traces are dropped otherwise."""
        if not stresses:
            return []
        taught = None if prog.fixed_reference is not None else self.taught(prog, world_model)
        per = max(1, min(CHUNK, math.ceil(len(stresses) / max(1, self.workers))))
        chunks = [stresses[i:i + per] for i in range(0, len(stresses), per)]
        if self._pool is None:
            return [r for c in chunks for r in _worker_run((prog, c, world_model, taught, keep))]
        out = []
        for res in self._pool.map(_worker_run,
                                  [(prog, c, world_model, taught, keep) for c in chunks]):
            out.extend(res)
        return out


# ------------------------------------------------------------------------- the search
def _to_unit(space, theta):
    lo, hi = np.array(space.lower()), np.array(space.upper())
    return (np.array([theta[n] for n in space.names()]) - lo) / (hi - lo)


def _from_unit(space, u):
    lo, hi = np.array(space.lower()), np.array(space.upper())
    return space.vector_to_named(lo + np.clip(u, 0, 1) * (hi - lo))


def _family_key(r):
    mode = r["dominant"]
    m = r["metrics"]
    joint = None
    if mode == "missed_steps":
        joint = int(np.argmax(m["lost_steps_by_joint"])) + 1
    elif mode == "torque_margin":
        joint = int(np.argmax(m["torque_util_by_joint"])) + 1
    elif mode in ("trajectory_abort", "goal_tolerance"):
        joint = int(np.argmax(m["max_path_err_deg"])) + 1
    return f"{mode}:J{joint}" if joint else mode


def _family_name(key):
    mode, _, joint = key.partition(":")
    return MODE_LABEL.get(mode, mode) + (f" on {joint}" if joint else "")


def run_search(prog, *, budget="quick", seed=0, world_model=None, space=None,
               progress=None, workers=None, runner=None) -> dict:
    """Search the stress space. ``progress(done, total, note)`` is called as batches finish."""
    from scipy.stats import qmc

    space = space or stress_space(prog, world_model=world_model)
    n_total = BUDGETS.get(budget, budget) if isinstance(budget, str) else int(budget)
    names = space.names()
    D = len(names)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    records = []

    def evaluate(thetas, note, rn):
        res = rn.run(prog, thetas, world_model)
        for th, r in zip(thetas, res):
            imp = implausibility(th, space)
            r["theta"] = th
            r["plausibility"] = round(imp["transfer"], 4)
            r["outside"] = imp["worst"][:4]
            r["n_outside"] = imp["n_outside"]
            r["id"] = len(records)
            records.append(r)
        if progress:
            progress(len(records), n_total, note)

    own = runner is None
    rn = runner or Runner(workers)
    if own:
        rn.__enter__()
    try:
        nominal = {n: space.dims[n].nominal for n in names}
        n_explore = max(8, int(0.4 * n_total))
        m = int(math.ceil(math.log2(n_explore)))
        sob = qmc.Sobol(D, scramble=True, seed=seed).random_base2(m)[:n_explore - 1]
        evaluate([nominal] + [_from_unit(space, u) for u in sob], "exploring", rn)

        # rounds big enough that every worker gets a lockstep batch of at least two worlds
        per_round = max(2 * max(1, rn.workers), (n_total - len(records)) // 4)
        sigma = 0.18
        while len(records) < n_total:
            n = min(per_round, n_total - len(records))
            valid = [r for r in records if r["valid"]]
            # fitness: how badly it fails, discounted by how far-fetched the world is
            fit = np.array([min(r["failure_score"], 3.0) * (0.25 + r["plausibility"])
                            for r in valid])
            order = np.argsort(-fit)
            # one elite per family first, so every failure mode keeps being explored
            elites, seen = [], set()
            for i in order:
                r = valid[i]
                key = _family_key(r) if r["failed"] else "near"
                if key not in seen:
                    seen.add(key)
                    elites.append(r)
                if len(elites) >= 6:
                    break
            for i in order[:12]:
                if len(elites) >= 10:
                    break
                if valid[i] not in elites:
                    elites.append(valid[i])
            nom_u = _to_unit(space, nominal)
            cands = []
            for k in range(n):
                e = elites[k % len(elites)]
                u = _to_unit(space, e["theta"])
                # half the samples pull toward nominal: the failure closest to normal use
                if k % 2:
                    u = u + rng.uniform(0.1, 0.5) * (nom_u - u)
                cands.append(_from_unit(space, u + rng.normal(0, sigma, D)))
            evaluate(cands, "refining", rn)
            sigma *= 0.75
    finally:
        if own:
            rn.__exit__(None, None, None)

    fams = families(records, space)
    return {
        "program": prog.to_dict(), "budget": budget, "n_worlds": len(records),
        "n_failed": sum(r["failed"] for r in records),
        "n_invalid": sum(not r["valid"] for r in records),
        "seconds": round(time.time() - t0, 1), "seed": seed,
        "space": {n: {"bounds": list(d.bounds), "nominal": d.nominal,
                      "plausible": list(d.plausible), "kind": d.kind}
                  for n, d in space.dims.items()},
        "families": fams, "records": [_slim(r) for r in records],
        "world_model": world_model.summary() if world_model is not None else None,
    }


def _slim(r):
    keep = ("id", "theta", "failure_score", "failed", "dominant", "plausibility", "outside",
            "n_outside", "valid", "events")
    out = {k: r.get(k) for k in keep}
    out["metrics"] = {k: r["metrics"].get(k) for k in (
        "lost_steps", "max_tcp_error_mm", "max_hidden_error_mm", "repeat_drift_mm",
        "peak_torque_util", "aborted_at_s")} if r.get("metrics") else {}
    out["family"] = _family_key(r) if r["failed"] else None
    return out


def families(records, space):
    groups = {}
    for r in records:
        if r["valid"] and r["failed"]:
            groups.setdefault(_family_key(r), []).append(r)
    fams = []
    for key, rs in groups.items():
        # representative: the member closest to real operation: fewest axes outside their
        # plausible band, then most plausible, ties to the larger failure
        rep = max(rs, key=lambda r: (-r.get("n_outside", 0), round(r["plausibility"], 2),
                                     r["failure_score"]))
        worst = max(rs, key=lambda r: r["failure_score"])
        fams.append({
            "key": key, "name": _family_name(key), "mode": key.split(":")[0],
            "count": len(rs), "share": round(len(rs) / max(1, len(records)), 4),
            "representative": rep["id"], "worst": worst["id"],
            "plausibility": rep["plausibility"], "score": round(rep["failure_score"], 3),
            "n_outside": rep.get("n_outside", 0),
            "events": rep["events"][:4], "theta": rep["theta"],
            "deviations": _deviations(rep["theta"], space),
            "in_plausible_band": rep["plausibility"] >= 0.999,
        })
    fams.sort(key=lambda f: (f["n_outside"], -f["plausibility"], -f["count"]))
    return fams


def _deviations(theta, space):
    """Axes away from nominal, largest relative move first."""
    out = []
    for n, d in space.dims.items():
        span = d.bounds[1] - d.bounds[0]
        rel = (theta[n] - d.nominal) / span if span else 0.0
        if abs(rel) > 0.04:
            out.append({"name": n, "value": round(theta[n], 4), "nominal": d.nominal,
                        "rel": round(rel, 3),
                        "outside_plausible": not d.plausible[0] <= theta[n] <= d.plausible[1]})
    out.sort(key=lambda x: -abs(x["rel"]))
    return out


# ------------------------------------------------------------------- minimal cause
def minimal_cause(prog, theta, space, *, world_model=None, runner=None, bisect_iters=4):
    """Which axes does this failure actually need, and from what value does it start?"""
    names = space.names()
    nominal = {n: space.dims[n].nominal for n in names}
    moved = [n for n in names if abs(theta[n] - nominal[n]) > 1e-9]
    own = runner is None
    rn = runner or Runner()
    if own:
        rn.__enter__()
    try:
        base = rn.run(prog, [theta], world_model)[0]
        if not base["failed"]:
            return {"reproduced": False}
        family = _family_key(base)

        def same(r):                     # the same failure, not merely some failure
            return r["valid"] and r["failed"] and _family_key(r) == family
        # greedy pruning, batched: every axis whose reset alone keeps the failure is a
        # candidate; reset them all at once if that still fails, else the single best one
        cur = dict(theta)
        n_sims = 1
        while True:
            live = [n for n in names if abs(cur[n] - nominal[n]) > 1e-9]
            if len(live) <= 1:
                break
            trials = [{**cur, n: nominal[n]} for n in live]
            res = rn.run(prog, trials, world_model)
            n_sims += len(trials)
            keep = [(n, r) for n, r in zip(live, res) if same(r)]
            if not keep:
                break
            joint = {**cur, **{n: nominal[n] for n, _ in keep}}
            if len(keep) > 1:
                rj = rn.run(prog, [joint], world_model)[0]
                n_sims += 1
                if same(rj):
                    cur = joint
                    continue
            # one at a time: the reset that leaves the strongest failure
            n_best = max(keep, key=lambda x: x[1]["failure_score"])[0]
            cur = {**cur, n_best: nominal[n_best]}
        cause = [n for n in names if abs(cur[n] - nominal[n]) > 1e-9]
        start = cur
        # bisect each cause axis between nominal (passes) and its value (fails)
        lo = {n: nominal[n] for n in cause}
        hi = {n: start[n] for n in cause}
        for _ in range(bisect_iters):
            probes = []
            for n in cause:
                mid = 0.5 * (lo[n] + hi[n])
                probes.append({**start, n: mid})
            out = rn.run(prog, probes, world_model)
            for n, p, r in zip(cause, probes, out):
                if same(r):
                    hi[n] = p[n]
                else:
                    lo[n] = p[n]
        thresholds = []
        for n in cause:
            d = space.dims[n]
            direction = "above" if start[n] > nominal[n] else "below"
            thresholds.append({
                "name": n, "threshold": round(hi[n], 4), "direction": direction,
                "nominal": d.nominal, "plausible": list(d.plausible),
                "within_plausible": d.plausible[0] <= hi[n] <= d.plausible[1]})
        return {"reproduced": True, "family": family, "axes": cause, "thresholds": thresholds,
                "irrelevant": [n for n in moved if n not in cause],
                "n_sims": n_sims + bisect_iters * len(cause)}
    finally:
        if own:
            rn.__exit__(None, None, None)


# ------------------------------------------------------------------------- envelope
def envelope(prog, x="payload_kg", y="accel_scale", *, nx=7, ny=7, base=None,
             world_model=None, space=None, runner=None, progress=None):
    space = space or stress_space(prog, world_model=world_model)
    dx, dy = space.dims[x], space.dims[y]
    xs = np.linspace(dx.bounds[0], dx.bounds[1], nx)
    ys = np.linspace(dy.bounds[0], dy.bounds[1], ny)
    base = {**{n: d.nominal for n, d in space.dims.items()}, **(base or {})}
    grid = [{**base, x: float(a), y: float(b)} for b in ys for a in xs]
    own = runner is None
    rn = runner or Runner()
    if own:
        rn.__enter__()
    try:
        res = rn.run(prog, grid, world_model)
    finally:
        if own:
            rn.__exit__(None, None, None)
    if progress:
        progress(len(grid), len(grid), "envelope")
    cells = [{"x": g[x], "y": g[y], "score": round(r["failure_score"], 3),
              "failed": r["failed"], "dominant": r["dominant"],
              "valid": r["valid"],
              "tcp_error_mm": r["metrics"].get("max_tcp_error_mm") if r["valid"] else None,
              "lost_steps": r["metrics"].get("lost_steps") if r["valid"] else None}
             for g, r in zip(grid, res)]
    return {"x": x, "y": y, "xs": xs.tolist(), "ys": ys.tolist(), "cells": cells,
            "base": base, "plausible": {x: list(dx.plausible), y: list(dy.plausible)}}


# ------------------------------------------------------------------------- endurance
def endurance(prog, cycles=10, stress=None, *, world_model=None):
    st = {**(stress or {}), "cycles": int(cycles)}
    r = simulate(prog, st, world_model=world_model, keep_trace=False)
    per = {}
    for p in r.get("taught_points", []):
        per.setdefault(p["label"], []).append(p)
    r["per_point"] = per
    return r
