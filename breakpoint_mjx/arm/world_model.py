"""Calibrate the simulator to a real AR4 from its own logs: a world model.

Simulation and the real arm disagree mostly in the drive train (friction, backlash,
compliance, motor torque), so the world model has two layers:

1. **Grey box.** The physical axes of the simulator (``friction_scale``,
   ``torque_scale``, ``backlash_scale``, ``stiffness_scale``, optionally ``payload_kg``)
   fitted so that replaying the logged command through the simulated drive train
   reproduces the logged feedback. Cross-entropy search, every candidate a full replay.
2. **Residual.** What physics does not explain (serial and driver latency, cogging,
   transmission error) is regressed, ridge, on features of the commanded motion. Its
   prediction is added to the commanded joint angle, scaled by the ``world_model_scale``
   stress axis: the same contract as the engine's ``world_model_scale`` rollout axis and
   ``env._bp_action_correction`` hook, here applied to a programmed arm.

**What a log can identify.** The AR4's encoders are on the motor shafts, so joint
feedback sees missed steps, lag and friction-driven following error, but is blind to
backlash and gearbox wind-up. Those need an external TCP measurement (dial indicator,
camera, laser tracker, a touch-off probe at the taught points): the optional ``tcp_*``
columns. ``fit`` reports, per parameter, whether this log actually constrained it.

Log format (CSV; ``log_from_csv``): ``t`` [s], ``cmd_j1..cmd_j6`` [deg] (the
joint_trajectory_controller reference), ``fb_j1..fb_j6`` [deg] (``/joint_states``), and
optionally ``tcp_x, tcp_y, tcp_z`` [mm] where an external measurement exists (blank
elsewhere). ``#payload_kg=0.5`` style comment lines set metadata.
"""
from __future__ import annotations

import csv
import io
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np

from breakpoint_mjx.arm import ar4
from breakpoint_mjx.arm.program import Program, reference_from_samples

PHYS = {  # name: (lower, upper, nominal)
    "friction_scale": (0.3, 4.0, 1.0),
    "torque_scale": (0.4, 1.2, 1.0),
    "backlash_scale": (0.2, 8.0, 1.0),
    "stiffness_scale": (0.2, 2.0, 1.0),
}
PAYLOAD = (0.0, 3.0)
ENC_SIGMA_DEG = 0.02       # joint-feedback residual that counts as "matched"
TCP_SIGMA_MM = 0.05


# ------------------------------------------------------------------------------ logs
@dataclass
class RobotLog:
    t: np.ndarray                 # (N,) s
    cmd: np.ndarray               # (N, 6) rad
    fb: np.ndarray                # (N, 6) rad
    tcp: np.ndarray | None = None  # (N, 3) m, NaN where not measured
    meta: dict = field(default_factory=dict)

    @property
    def has_tcp(self) -> bool:
        return self.tcp is not None and np.isfinite(self.tcp).any()

    def summary(self) -> dict:
        return {"samples": int(len(self.t)), "duration_s": round(float(self.t[-1] - self.t[0]), 3),
                "rate_hz": round(float((len(self.t) - 1) / max(self.t[-1] - self.t[0], 1e-9)), 1),
                "tcp_samples": int(np.isfinite(self.tcp).all(1).sum()) if self.has_tcp else 0,
                "meta": self.meta}


def log_from_csv(text: str) -> RobotLog:
    meta, rows = {}, []
    for line in io.StringIO(text):
        s = line.strip()
        if s.startswith("#"):
            if "=" in s:
                k, v = s[1:].split("=", 1)
                try:
                    meta[k.strip()] = float(v)
                except ValueError:
                    meta[k.strip()] = v.strip()
            continue
        if s:
            rows.append(line)
    rd = csv.DictReader(rows)
    need = ["t"] + [f"cmd_j{i}" for i in range(1, 7)] + [f"fb_j{i}" for i in range(1, 7)]
    missing = [c for c in need if c not in (rd.fieldnames or [])]
    if missing:
        raise ValueError(f"log is missing columns: {', '.join(missing)}")
    t, cmd, fb, tcp = [], [], [], []
    has_tcp = all(c in rd.fieldnames for c in ("tcp_x", "tcp_y", "tcp_z"))
    for r in rd:
        t.append(float(r["t"]))
        cmd.append([float(r[f"cmd_j{i}"]) for i in range(1, 7)])
        fb.append([float(r[f"fb_j{i}"]) for i in range(1, 7)])
        if has_tcp:
            tcp.append([float(r[c]) if r[c] not in ("", None) else math.nan
                        for c in ("tcp_x", "tcp_y", "tcp_z")])
    if len(t) < 10:
        raise ValueError("log needs at least 10 samples")
    t = np.asarray(t)
    order = np.argsort(t)
    return RobotLog(t=t[order] - t[order][0], cmd=np.deg2rad(np.asarray(cmd))[order],
                    fb=np.deg2rad(np.asarray(fb))[order],
                    tcp=(np.asarray(tcp)[order] * 1e-3) if has_tcp else None, meta=meta)


def log_to_csv(log: RobotLog) -> str:
    out = io.StringIO()
    for k, v in log.meta.items():
        out.write(f"#{k}={v}\n")
    cols = ["t"] + [f"cmd_j{i}" for i in range(1, 7)] + [f"fb_j{i}" for i in range(1, 7)]
    if log.tcp is not None:
        cols += ["tcp_x", "tcp_y", "tcp_z"]
    w = csv.writer(out)
    w.writerow(cols)
    for i in range(len(log.t)):
        row = [f"{log.t[i]:.4f}"] + [f"{x:.5f}" for x in np.rad2deg(log.cmd[i])] + \
              [f"{x:.5f}" for x in np.rad2deg(log.fb[i])]
        if log.tcp is not None:
            row += ["" if not np.isfinite(x) else f"{x * 1e3:.4f}" for x in log.tcp[i]]
        w.writerow(row)
    return out.getvalue()


# ------------------------------------------------------------------------ the model
# Dynamic features only: velocity (latency, viscous lag) and its sign (direction-dependent
# lag). They vanish at rest, so the correction never moves a taught point at a dwell: the
# encoders cannot say where the link is at rest (motor-shaft encoders), the TCP data and
# the physics layer do.
FEATURES = ("qd", "sign_qd")


def _features(q, qd):
    """Per-joint features (B, 6, F) of the commanded motion."""
    qd = np.atleast_2d(qd)
    return np.stack([qd, np.tanh(qd / 0.02)], -1)


@dataclass
class WorldModel:
    """A fitted AR4 world model: calibrated physics + a residual action correction."""
    params: dict                       # calibrated stress-axis values (the new nominal)
    W: np.ndarray                      # (6, F) residual weights, joint rad per feature
    report: dict = field(default_factory=dict)
    model: str = "mk5"

    def correction(self, q, qd):
        """Joint-angle correction [rad] for commanded (q, qd), shape (B, 6)."""
        return np.einsum("bjf,jf->bj", _features(q, qd), self.W)

    def nominal_stress(self) -> dict:
        return {**self.params, "world_model_scale": 1.0}

    def summary(self) -> dict:
        return {"params": self.params, "features": list(FEATURES),
                "residual_rms_deg": self.report.get("after", {}).get("joint_rms_deg"),
                "fitted_at": self.report.get("fitted_at")}

    def to_json(self) -> str:
        return json.dumps({"kind": "breakpoint.ar4.world_model/v1", "model": self.model,
                           "params": self.params, "W": self.W.tolist(),
                           "features": list(FEATURES), "report": self.report}, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "WorldModel":
        d = json.loads(text)
        if d.get("kind") != "breakpoint.ar4.world_model/v1":
            raise ValueError("not an AR4 world model file")
        return cls(params=d["params"], W=np.asarray(d["W"]), report=d.get("report", {}),
                   model=d.get("model", "mk5"))


# ------------------------------------------------------------------------- fitting
def _replay_program(log: RobotLog, model: str, payload_kg: float, cog_mm) -> Program:
    ref = reference_from_samples(log.t, log.cmd)
    return Program(name="log replay", moves=[{"type": "dwell", "s": 0.0}], model=model,
                   control=str(log.meta.get("control", "ros2")), cycles=1,
                   payload_kg=payload_kg, payload_cog_mm=list(cog_mm),
                   start_deg=np.rad2deg(log.cmd[0]).tolist(), fixed_reference=ref)


def _errors(log, res):
    """Joint (deg) and TCP (mm) residuals of one replay against the log."""
    tr = res.get("trace")
    if not res.get("valid") or not tr:
        return None
    ts = np.asarray(tr["t"])
    enc = np.asarray(tr["q_enc"])
    sim_fb = np.stack([np.interp(log.t, ts, enc[:, j]) for j in range(6)], 1)
    ej = np.rad2deg(sim_fb - log.fb)
    et = None
    if log.has_tcp:
        tcp = np.asarray(tr["tcp"])
        sim_tcp = np.stack([np.interp(log.t, ts, tcp[:, j]) for j in range(3)], 1)
        m = np.isfinite(log.tcp).all(1)
        et = (sim_tcp[m] - log.tcp[m]) * 1e3
    return ej, et, sim_fb


def _loss(errs):
    if errs is None:
        return float("inf")
    ej, et, _ = errs
    L = float(np.mean((ej / ENC_SIGMA_DEG) ** 2))
    if et is not None and len(et):
        L += float(np.mean((et / TCP_SIGMA_MM) ** 2))
    return L


def _stats(log, errs):
    ej, et, _ = errs
    out = {"joint_rms_deg": round(float(np.sqrt(np.mean(ej ** 2))), 5),
           "joint_rms_by_joint_deg": [round(float(x), 5) for x in np.sqrt(np.mean(ej ** 2, 0))],
           "joint_max_deg": round(float(np.max(np.abs(ej))), 4)}
    if et is not None and len(et):
        n = np.linalg.norm(et, axis=1)
        out.update({"tcp_rms_mm": round(float(np.sqrt(np.mean(n ** 2))), 4),
                    "tcp_max_mm": round(float(n.max()), 4)})
    return out


def fit(log: RobotLog, *, model: str = "mk5", payload_kg: float | None = None,
        cog_mm=(0, 0, 0), fit_payload: bool = False, iters: int = 6, pop: int = 14,
        seed: int = 0, runner=None, progress=None) -> WorldModel:
    """Fit a world model to ``log``. ``progress(done, total, note)`` as batches finish."""
    from breakpoint_mjx.arm.search import Runner

    t0 = time.time()
    pk = float(log.meta.get("payload_kg", 0.0) if payload_kg is None else payload_kg)
    prog = _replay_program(log, model, pk, cog_mm)
    names = list(PHYS) + (["payload_kg"] if fit_payload else [])
    lo = np.array([PHYS[n][0] for n in PHYS] + ([PAYLOAD[0]] if fit_payload else []))
    hi = np.array([PHYS[n][1] for n in PHYS] + ([PAYLOAD[1]] if fit_payload else []))
    # scales searched in log space, payload in linear space
    is_log = np.array([True] * len(PHYS) + ([False] if fit_payload else []))
    to_u = lambda x: np.where(is_log, np.log(x), x)
    from_u = lambda u: np.clip(np.where(is_log, np.exp(u), u), lo, hi)
    mean = to_u(np.array([PHYS[n][2] for n in PHYS] + ([pk] if fit_payload else [])))
    std = np.where(is_log, 0.5, 0.6)
    rng = np.random.default_rng(seed)
    total = 1 + iters * pop + len(names) * 2 + 1
    done = 0
    keep = ("t", "q_enc", "tcp")
    own = runner is None
    rn = runner or Runner()
    if own:
        rn.__enter__()
    try:
        def run(xs):
            nonlocal done
            sts = [dict(zip(names, map(float, x))) for x in xs]
            res = rn.run(prog, sts, keep=keep)
            done += len(sts)
            if progress:
                progress(done, total, "calibrating physics")
            return [(_errors(log, r)) for r in res]

        base_err = run([from_u(mean)])[0]
        if base_err is None:
            raise ValueError("the simulator could not replay this log (check units: degrees)")
        before = _stats(log, base_err)
        best_x, best_L, best_err = from_u(mean), _loss(base_err), base_err
        for _ in range(iters):
            U = mean + rng.normal(0, 1, (pop, len(names))) * std
            X = [from_u(u) for u in U]
            errs = run(X)
            L = np.array([_loss(e) for e in errs])
            k = np.argsort(L)[: max(3, pop // 3)]
            if L[k[0]] < best_L:
                best_L, best_x, best_err = float(L[k[0]]), X[k[0]], errs[k[0]]
            mean = 0.5 * mean + 0.5 * np.mean(U[k], 0)
            std = np.maximum(0.6 * std + 0.4 * np.std(U[k], 0), 0.03)
        params = dict(zip(names, (round(float(v), 4) for v in best_x)))
        grey = _stats(log, best_err)

        # identifiability: does nudging each parameter +-25% change the fit?
        probes = []
        for i, n in enumerate(names):
            for f in (0.75, 1.25):
                x = np.array(best_x, float)
                x[i] = np.clip(x[i] * f if is_log[i] else x[i] + (f - 1) * 2.0, lo[i], hi[i])
                probes.append(x)
        perr = run(probes)
        ident = {}
        for i, n in enumerate(names):
            dL = max(_loss(perr[2 * i]), _loss(perr[2 * i + 1])) - best_L
            rel = dL / max(best_L, 1e-9)
            ident[n] = {"loss_increase": round(float(rel), 4),
                        "status": "identified" if rel > 0.05 else
                                  "weak" if rel > 0.005 else "not identifiable from this log"}
        for n in ("backlash_scale", "stiffness_scale"):
            if n in ident and not log.has_tcp and ident[n]["status"] != "identified":
                ident[n]["why"] = ("the AR4 encoders are on the motor shafts: backlash and "
                                   "gearbox wind-up never reach them. Add TCP measurements.")

        # residual: what the calibrated physics still misses, regressed on the command
        _, _, sim_fb = best_err
        qd = np.gradient(log.cmd, log.t, axis=0)
        Phi = _features(log.cmd, qd)                                  # (N, 6, F)
        R = log.fb - sim_fb                                           # (N, 6) rad
        F = Phi.shape[-1]
        W = np.zeros((6, F))
        for j in range(6):
            A = Phi[:, j]
            sc = np.maximum(np.std(A, 0), 1e-6)
            An = A / sc
            lam = 1e-3 * len(A)
            w = np.linalg.solve(An.T @ An + lam * np.eye(F), An.T @ R[:, j])
            W[j] = w / sc
        wm = WorldModel(params=params, W=W, model=model)
        res = rn.run(prog, [{**params, "world_model_scale": 1.0}], world_model=wm, keep=keep)
        done += 1
        after_err = _errors(log, res[0])
        after = _stats(log, after_err) if after_err is not None else grey
        _, _, after_fb = after_err if after_err is not None else best_err
    finally:
        if own:
            rn.__exit__(None, None, None)

    step = max(1, len(log.t) // 600)
    wm.report = {
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - t0, 1), "log": log.summary(), "payload_kg": pk,
        "before": before, "physics": grey, "after": after, "params": params,
        "defaults": {n: PHYS[n][2] for n in PHYS}, "identifiability": ident,
        "series": {"t": log.t[::step].round(4).tolist(),
                   "real": np.rad2deg(log.fb[::step]).round(4).tolist(),
                   "cmd": np.rad2deg(log.cmd[::step]).round(4).tolist(),
                   "sim_before": np.rad2deg(base_err[2][::step]).round(4).tolist(),
                   "sim_after": np.rad2deg(after_fb[::step]).round(4).tolist()},
        "n_sims": done,
    }
    return wm


# ----------------------------------------------------------------- synthetic robot
SYNTHETIC_TRUTH = {"friction_scale": 1.8, "torque_scale": 0.85, "backlash_scale": 2.5,
                   "stiffness_scale": 0.7}


def synthetic_log(prog: Program, *, truth: dict | None = None, latency_s: float = 0.017,
                  rate_hz: float = 100.0, tcp_noise_mm: float = 0.03, seed: int = 0) -> RobotLog:
    """A stand-in for a real robot's log, for demos and tests: the program run on a
    simulator with hidden drive-train parameters (``truth``) plus a command latency the
    simulator does not model, logged at ``rate_hz`` with TCP measured at the dwells.
    Labelled SYNTHETIC in its metadata; replace it with your robot's log."""
    from breakpoint_mjx.arm.program import build_reference
    from breakpoint_mjx.arm.sim import simulate

    truth = dict(SYNTHETIC_TRUTH if truth is None else truth)
    rng = np.random.default_rng(seed)
    ref = build_reference(prog, {"cycles": 1}, rate_hz=1000.0)
    lag = int(round(latency_s * 1000))
    delayed = np.concatenate([np.repeat(ref.q[:1], lag, 0), ref.q[:-lag or None]], 0)
    from breakpoint_mjx.arm.program import Reference
    real_ref = Reference(t=ref.t, q=delayed, qd=np.gradient(delayed, 1e-3, axis=0),
                         segment=ref.segment, cycle=ref.cycle, measure=ref.measure,
                         seg_end=ref.seg_end, issues=[], min_manip=ref.min_manip,
                         peak_speed_ratio=ref.peak_speed_ratio)
    real = Program(**{**prog.__dict__, "fixed_reference": real_ref, "cycles": 1})
    r = simulate(real, truth, keep_trace=True)
    tr = r["trace"]
    ts = np.asarray(tr["t"])
    t = np.arange(0.0, float(ref.t[-1]), 1.0 / rate_hz)
    cmd = np.stack([np.interp(t, ref.t, ref.q[:, j]) for j in range(6)], 1)
    enc = np.asarray(tr["q_enc"])
    fb = np.stack([np.interp(t, ts, enc[:, j]) for j in range(6)], 1)
    P = np.asarray(tr["tcp"])
    tcp = np.full((len(t), 3), np.nan)
    for (i, _, _) in ref.measure:                 # a probe reading near the end of each dwell
        ti = ref.t[i]
        sel = (t > ti - 0.25) & (t <= ti)
        for k in np.flatnonzero(sel):
            tcp[k] = [np.interp(t[k], ts, P[:, j]) for j in range(3)]
            tcp[k] += rng.normal(0, tcp_noise_mm * 1e-3, 3)
    meta = {"source": "SYNTHETIC (simulated stand-in for a real AR4 log)",
            "payload_kg": prog.payload_kg, "control": prog.control,
            "program": prog.name, "model": prog.model}
    return RobotLog(t=t, cmd=cmd, fb=fb, tcp=tcp, meta=meta)
