"""Robot programs: a list of moves turned into a timed joint reference q_ref(t).

A program is plain YAML (``breakpoint_mjx/arm/programs/*.yaml``)::

    name: Pick and place
    model: mk5
    control: ros2            # ros2 | ros2_pid | ros2_position | open_loop (see sim.py)
    cycles: 3
    tolerance_mm: 1.0        # TCP error that counts as a failure
    start_deg: [0, 0, 0, 0, 90, 0]
    moves:
      - {type: joint, deg: [0, 30, -20, 0, 70, 0], speed: 80, accel: 100}
      - {type: linear, xyz_mm: [0, -380, 150], speed_mm_s: 60, accel_mm_s2: 200}
      - {type: dwell, s: 0.4, measure: pick}

``model`` names the robot (``robots.py``: ar4_mk5 / mk5, ar4_mk3, ur5e, xarm6).
``speed``/``accel``/``decel`` on joint moves are percent of that robot's program reference
(``RobotDef.v100_deg_s`` / ``a100_deg_s2``; on the AR4 the firmware's 60 deg/s and
30 deg/s^2, like its own speed setting). Linear moves interpolate the TCP on a
straight line (orientation slerped, or held when ``rpy_deg`` is omitted) and are solved by
damped least squares IK at every sample, so singularities and reach limits show up as
joint-speed spikes and IK misses rather than being planned around.

``build_reference`` applies the stress axes that belong to the program (speed, accel and
decel scale, a job shift, a reach scale) and returns arrays at ``rate_hz``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from breakpoint_mjx.arm.robots import build_model, get_robot

PROGRAM_DIR = Path(__file__).parent / "programs"


@dataclass
class Program:
    name: str
    moves: list
    model: str = "mk5"
    control: str = "ros2"
    cycles: int = 1
    tolerance_mm: float = 1.0
    start_deg: list = field(default_factory=lambda: [0, 0, 0, 0, 90, 0])
    payload_kg: float = 0.0
    payload_cog_mm: list = field(default_factory=lambda: [0, 0, 0])
    description: str = ""
    source: str = ""
    # a logged command to replay instead of the moves (world_model.py): stress axes that
    # retime or move the program do not apply to it
    fixed_reference: "Reference | None" = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "name", "description", "model", "control", "cycles", "tolerance_mm",
            "start_deg", "payload_kg", "payload_cog_mm", "moves")}


def reference_from_samples(t, q, rate_hz: float = 1000.0) -> "Reference":
    """A replayable 1 kHz reference from logged joint commands (t [s], q [rad], any rate)."""
    t = np.asarray(t, float)
    q = np.asarray(q, float)
    tt = np.arange(t[0], t[-1], 1.0 / rate_hz)
    Q = np.stack([np.interp(tt, t, q[:, j]) for j in range(6)], 1)
    QD = np.gradient(Q, 1.0 / rate_hz, axis=0)
    return Reference(t=tt - tt[0], q=Q, qd=QD, segment=np.zeros(len(tt), int),
                     cycle=np.zeros(len(tt), int), measure=[], seg_end=[], issues=[],
                     min_manip=float("nan"), peak_speed_ratio=float(
                         np.max(np.abs(QD)) / math.radians(60.0)))


def parse_program(raw: dict, source: str = "") -> Program:
    if not isinstance(raw, dict) or not raw.get("moves"):
        raise ValueError("a program needs a non-empty 'moves' list")
    for i, mv in enumerate(raw["moves"]):
        t = mv.get("type")
        if t not in ("joint", "linear", "dwell"):
            raise ValueError(f"move {i}: type must be joint, linear or dwell (got {t!r})")
        if t == "joint" and len(mv.get("deg", [])) != 6:
            raise ValueError(f"move {i}: a joint move needs 6 angles in 'deg'")
        if t == "linear" and len(mv.get("xyz_mm", [])) != 3:
            raise ValueError(f"move {i}: a linear move needs 'xyz_mm' [x, y, z]")
    robot = get_robot(raw.get("model", "mk5"))
    control = raw.get("control", robot.default_control)
    if control not in robot.control_modes:
        raise ValueError(f"control for {robot.name} must be one of {', '.join(robot.control_modes)}")
    raw = {**raw, "control": control}
    tool = raw.get("tool", {})
    return Program(
        name=str(raw.get("name", "program")), moves=list(raw["moves"]),
        model=robot.id, control=str(control),
        cycles=int(raw.get("cycles", 1)), tolerance_mm=float(raw.get("tolerance_mm", 1.0)),
        start_deg=list(raw.get("start_deg", get_robot(raw.get("model", "mk5")).home_deg)),
        payload_kg=float(tool.get("payload_kg", raw.get("payload_kg", 0.0))),
        payload_cog_mm=list(tool.get("cog_mm", raw.get("payload_cog_mm", [0, 0, 0]))),
        description=str(raw.get("description", "")), source=source)


def load_program(path_or_name) -> Program:
    p = Path(path_or_name)
    if not p.exists():
        p = PROGRAM_DIR / f"{path_or_name}.yaml"
    return parse_program(yaml.safe_load(p.read_text()), source=str(p))


def builtin_programs() -> list[str]:
    return sorted(p.stem for p in PROGRAM_DIR.glob("*.yaml"))


# --------------------------------------------------------------------------- kinematics
class Kinematics:
    """FK / Jacobian / DLS IK on a payload-free copy of the model."""

    def __init__(self, model_name: str = "mk5"):
        import mujoco

        self._mj = mujoco
        self.robot = get_robot(model_name)
        self.m = build_model(self.robot)
        self.d = mujoco.MjData(self.m)
        self.site = self.m.site("tcp").id
        self.lim = self.robot.limits_rad
        self.path_cache: dict = {}

    def line_path(self, q0, p1, R1, n):
        """IK along a straight TCP line from FK(q0) to (p1, R1) at ``n`` evenly spaced path
        points, warm-started point to point. Cached: retiming a move (speed and accel
        scales) reuses the geometry. Returns (joint path (n+1, 6), worst miss [m], manip)."""
        key = (tuple(np.round(q0, 9)), tuple(np.round(p1, 9)), tuple(np.round(R1, 9).ravel()), n)
        hit = self.path_cache.get(key)
        if hit is not None:
            return hit
        p0, R0 = self.fk(q0)
        path, worst, manip = [np.array(q0, float)], 0.0, float("inf")
        qk = np.array(q0, float)
        for sk in np.linspace(0.0, 1.0, n + 1)[1:]:
            qk, err, _ = self.ik(qk, p0 + sk * (p1 - p0), _slerp_mat(R0, R1, sk), iters=40)
            worst = max(worst, err)
            path.append(qk.copy())
            manip = min(manip, self.manipulability(qk))
        out = (np.asarray(path), worst, manip)
        if len(self.path_cache) > 512:
            self.path_cache.clear()
        self.path_cache[key] = out
        return out

    def fk(self, q):
        self.d.qpos[:] = q
        self._mj.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def jac(self, q):
        self.fk(q)
        self._mj.mj_comPos(self.m, self.d)
        jp = np.zeros((3, 6))
        jr = np.zeros((3, 6))
        self._mj.mj_jacSite(self.m, self.d, jp, jr, self.site)
        return jp, jr

    def manipulability(self, q) -> float:
        jp, jr = self.jac(q)
        J = np.vstack([jp, jr * 0.1])        # rotation weighted to metres (10 cm lever)
        return float(math.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def ik(self, q0, pos, rot, iters=60, tol=1e-5):
        """Damped least squares. Returns (q, position error [m], converged)."""
        q = np.array(q0, float)
        lam = 1e-2
        for _ in range(iters):
            p, R = self.fk(q)
            ep = pos - p
            M = rot @ R.T                     # orientation error as an axis-angle vee
            er = 0.5 * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 1e-4:
                break
            jp, jr = self.jac(q)
            J = np.vstack([jp, jr])
            e = np.concatenate([ep, er])
            dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e)
            q = np.clip(q + dq, self.lim[:, 0], self.lim[:, 1])
        p, _ = self.fk(q)
        err = float(np.linalg.norm(pos - p))
        return q, err, err < 5e-4


def rpy_to_mat(rpy_deg):
    r, p, y = np.deg2rad(rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def mat_to_rpy(R):
    p = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    r = math.atan2(R[2, 1], R[2, 2])
    y = math.atan2(R[1, 0], R[0, 0])
    return np.rad2deg([r, p, y])


def _slerp_mat(R0, R1, s):
    """Rotation interpolation via the relative axis-angle."""
    Rr = R0.T @ R1
    c = max(-1.0, min(1.0, (np.trace(Rr) - 1) / 2))
    ang = math.acos(c)
    if ang < 1e-9:
        return R0
    axis = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]])
    axis /= (2 * math.sin(ang)) if math.sin(ang) > 1e-9 else 1.0
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = ang * s
    return R0 @ (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K)


# ----------------------------------------------------------------------- time scaling
def trapezoid(dist: float, vmax: float, acc: float, dec: float, dt: float) -> np.ndarray:
    """Normalised path parameter s(t) in [0, 1] for a trapezoid over ``dist`` with
    separate accel/decel. Triangular when the distance is too short to reach vmax."""
    if dist <= 1e-12:
        return np.array([1.0])
    ta, td = vmax / acc, vmax / dec
    da, dd = 0.5 * vmax * ta, 0.5 * vmax * td
    if da + dd > dist:                       # triangular profile
        vpk = math.sqrt(2 * dist * acc * dec / (acc + dec))
        ta, td, tc = vpk / acc, vpk / dec, 0.0
    else:
        vpk, tc = vmax, (dist - da - dd) / vmax
    T = ta + tc + td
    t = np.arange(0.0, T + dt, dt)
    t[-1] = min(t[-1], T)
    x = np.where(t < ta, 0.5 * acc * t ** 2,
                 np.where(t < ta + tc, 0.5 * acc * ta ** 2 + vpk * (t - ta),
                          dist - 0.5 * dec * np.maximum(T - t, 0.0) ** 2))
    return np.clip(x / dist, 0.0, 1.0)


@dataclass
class Reference:
    t: np.ndarray            # (T,)
    q: np.ndarray            # (T, 6) joint reference [rad]
    qd: np.ndarray           # (T, 6)
    segment: np.ndarray      # (T,) move index (within the cycle)
    cycle: np.ndarray        # (T,)
    measure: list            # [(sample index, label, cycle)] where TCP accuracy is checked
    seg_end: list            # sample indices where a move ends (goal tolerance checks)
    issues: list             # program-level findings: unreachable, singular, limit
    min_manip: float         # smallest manipulability met on linear moves
    peak_speed_ratio: float  # largest joint speed / joint speed limit


def build_reference(prog: Program, stress: dict | None = None, rate_hz: float = 500.0,
                    kin: Kinematics | None = None) -> Reference:
    if prog.fixed_reference is not None:
        return prog.fixed_reference
    st = stress or {}
    vs = float(st.get("speed_scale", 1.0))
    acs = float(st.get("accel_scale", 1.0))
    dcs = float(st.get("decel_scale", acs))
    shift = np.array([st.get("shift_x_mm", 0.0), st.get("shift_y_mm", 0.0),
                      st.get("shift_z_mm", 0.0)]) * 1e-3
    reach = float(st.get("reach_scale", 1.0))
    cycles = int(round(st.get("cycles", prog.cycles)))
    kin = kin or Kinematics(prog.model)
    lim = kin.lim
    dt = 1.0 / rate_hz
    rb = kin.robot
    v100 = math.radians(rb.v100_deg_s)
    alim = math.radians(rb.a100_deg_s2)
    vlim = rb.vmax                                   # per-joint hard speed limit

    q = np.deg2rad(np.asarray(prog.start_deg, float))
    qs, segs, cyc, measure, seg_end, issues = [q[None]], [0], [0], [], [], []
    min_manip = float("inf")
    pts = 1

    def add(block, seg, c):
        nonlocal pts
        qs.append(block)
        segs.extend([seg] * len(block))
        cyc.extend([c] * len(block))
        pts += len(block)

    for c in range(cycles):
        for i, mv in enumerate(prog.moves):
            typ = mv["type"]
            if typ == "dwell":
                n = max(1, int(round(float(mv.get("s", 0.5)) * rate_hz)))
                add(np.repeat(q[None], n, 0), i, c)
                if mv.get("measure"):
                    measure.append((pts - 1, str(mv["measure"]), c))
                continue
            sp = float(mv.get("speed", 100)) / 100.0 * vs
            ac = float(mv.get("accel", 100)) / 100.0 * acs
            de = float(mv.get("decel", mv.get("accel", 100))) / 100.0 * dcs
            if typ == "joint":
                tgt = np.deg2rad(np.asarray(mv["deg"], float))
                for j in range(6):
                    if not lim[j, 0] - 1e-9 <= tgt[j] <= lim[j, 1] + 1e-9:
                        issues.append({"kind": "joint_limit", "move": i, "joint": j + 1,
                                       "detail": f"J{j + 1} target {math.degrees(tgt[j]):.1f} deg "
                                                 "is outside the joint limits"})
                tgt = np.clip(tgt, lim[:, 0], lim[:, 1])
                delta = tgt - q
                big = np.max(np.abs(delta))
                if big < 1e-9:
                    continue
                # common path parameter: the most constrained joint sets every limit
                s = trapezoid(1.0, sp * v100 / big, ac * alim / big, de * alim / big, dt)
                add(q[None] + s[1:, None] * delta[None], i, c)
                q = tgt
            else:
                p0, R0 = kin.fk(q)
                p1 = np.asarray(mv["xyz_mm"], float) * 1e-3
                p1 = np.array([p1[0] * reach, p1[1] * reach, p1[2]]) + shift
                R1 = rpy_to_mat(mv["rpy_deg"]) if "rpy_deg" in mv else R0
                v = float(mv.get("speed_mm_s", 50.0)) * 1e-3 * vs
                a = float(mv.get("accel_mm_s2", 200.0)) * 1e-3 * acs
                d = float(mv.get("decel_mm_s2", mv.get("accel_mm_s2", 200.0))) * 1e-3 * dcs
                L = float(np.linalg.norm(p1 - p0))
                if L < 1e-6:
                    L, v, a, d = 1e-3, 1e-3, 1e-2, 1e-2   # pure reorientation: 1 s-ish move
                s = trapezoid(L, v, a, d, dt)
                n_geo = int(min(400, max(24, L / 1e-3)))          # ~1 mm path resolution
                path, worst_err, manip = kin.line_path(q, p1, R1, n_geo)
                min_manip = min(min_manip, manip)
                u = np.linspace(0.0, 1.0, n_geo + 1)
                block = np.stack([np.interp(s[1:], u, path[:, j]) for j in range(6)], 1)
                if worst_err > 1e-3:
                    issues.append({"kind": "unreachable", "move": i,
                                   "detail": f"linear move {i} misses its path by up to "
                                             f"{worst_err * 1e3:.1f} mm (outside the envelope "
                                             "or locked by a joint limit)"})
                if len(block):
                    add(block, i, c)
                    q = block[-1]
            seg_end.append(pts - 1)

    Q = np.concatenate(qs, 0)
    t = np.arange(len(Q)) * dt
    QD = np.gradient(Q, dt, axis=0)
    peak = float(np.max(np.abs(QD) / vlim)) if len(Q) > 1 else 0.0
    # a straight-line move that needs a joint far beyond its speed limit is a singularity
    for i, mv in enumerate(prog.moves):
        if mv["type"] != "linear":
            continue
        mask = np.asarray(segs) == i
        if mask.any():
            r = np.max(np.abs(QD[mask]), axis=0) / vlim
            j = int(np.argmax(r))
            if r[j] > 1.5 * max(vs, 1.0):
                issues.append({"kind": "singularity", "move": i, "joint": j + 1,
                               "detail": f"linear move {i} needs J{j + 1} at "
                                         f"{r[j] * math.degrees(vlim[j]):.0f} deg/s "
                                         f"({r[j]:.1f}x the joint limit): passing near a "
                                         "singularity"})
    uniq, seen = [], set()
    for it in issues:
        k = (it["kind"], it["move"], it.get("joint"))
        if k not in seen:
            seen.add(k)
            uniq.append(it)
    return Reference(t=t, q=Q, qd=QD, segment=np.asarray(segs), cycle=np.asarray(cyc),
                     measure=measure, seg_end=seg_end, issues=uniq,
                     min_manip=min_manip if math.isfinite(min_manip) else float("nan"),
                     peak_speed_ratio=peak)
