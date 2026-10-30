"""One arm run: a program, a set of stress conditions, and what the robot actually did.

Two drive trains (``robots.RobotDef.drive``), each simulated per joint at the motor, not as
an ideal position source. ``servo`` (UR5e, xArm 6): ``ServoDrive`` below, a brushless servo
on a motor-side encoder with a cascaded position loop at the robot's servo rate, model-based
gravity feed-forward, a peak torque limit, a compliant harmonic drive, and a protective stop
on following error or sustained torque saturation. ``stepper`` (the AR4), ``StepperDrive``:

  step generator --(commanded motor angle)--> stepper rotor --(gearbox: backlash +
  stiffness)--> link (MuJoCo: gravity, inertia, payload, joint friction)

* **Stepper.** Torque on the rotor is ``T(w) sin(N_r (theta_cmd - theta_rotor))`` with
  ``N_r`` = 50 rotor teeth and a pull-out torque ``T(w)`` that falls with speed. When the
  load needs more than ``T(w)`` the load angle passes 90 electrical degrees, the rotor
  slips to the next stable tooth and the axis has lost 4 full steps. That is a missed
  step, from the physics rather than a threshold.
* **Gearbox.** Lost motion (backlash) and torsional compliance between rotor and link.
* **Encoders sit on the motor shafts** (AR4: AMT102 on each stepper, 4000 counts/rev), so
  they see missed steps but are blind to backlash and gearbox wind-up. The run reports
  both the TCP error the encoders can see and the error they cannot.
* **Controller** (``program.control``), the AR4 ROS 2 driver's three configurations:

  - ``ros2`` (as shipped: ``velocity_control_enabled: true``, command interfaces
    position + velocity). With both interfaces claimed, ros2_controllers'
    joint_trajectory_controller does not use its PID adapter: every 60 Hz it streams the
    desired velocity (``MV``) and the firmware runs each stepper at that speed, clipped to
    60 deg/s. Nothing corrects a lost step; the path (0.2 rad) and goal (0.02 rad)
    tolerances on the encoder reading are the only checks.
  - ``ros2_pid``: velocity as the only command interface, so the PID in
    controllers.yaml (p=10, i=1, d=1 on the velocity error, i_clamp=1) closes the loop on
    the encoders.
  - ``ros2_position`` (``velocity_control_enabled: false``): the driver streams positions
    (``MT``) and the firmware re-targets each stepper by the encoder error with
    AccelStepper ramps (60 deg/s, 30 deg/s^2), which makes up lost steps.
  - ``open_loop``: the commanded motor angle follows the reference exactly (an ideal
    step generator; a baseline with no controller in the way).

``world_model`` (optional) is a fitted correction from ``world_model.py``: its residual is
added to the commanded joint angle, scaled by the ``world_model_scale`` stress axis, the
same contract as the engine's ``world_model_scale`` rollout axis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from breakpoint_mjx.arm import ar4
from breakpoint_mjx.arm.program import Kinematics, Program, build_reference
from breakpoint_mjx.arm.robots import build_model, get_robot

ROS2_MODES = ("ros2", "ros2_pid", "ros2_position")
CONTROL_MODES = ROS2_MODES + ("open_loop", "servo")
SAMPLE_HZ = 50.0          # what is kept for replay and plots
SUBSTEPS = 2              # drive-train integration substeps per 1 ms MuJoCo step

FAILURE_MODES = ("missed_steps", "trajectory_abort", "protective_stop", "goal_tolerance",
                 "tcp_error", "hidden_error", "repeatability", "torque_margin", "joint_limit",
                 "singularity", "unreachable", "vibration", "table_collision")

# a missed step causes the abort, which causes the TCP error: the earliest link names it
CAUSAL_ORDER = ("unreachable", "joint_limit", "singularity", "table_collision",
                "missed_steps", "protective_stop", "trajectory_abort", "goal_tolerance",
                "torque_margin", "repeatability", "vibration", "hidden_error", "tcp_error")

MODE_LABEL = {
    "missed_steps": "Missed steps",
    "trajectory_abort": "Controller abort (path tolerance)",
    "protective_stop": "Protective stop",
    "goal_tolerance": "Controller goal tolerance missed",
    "tcp_error": "Off its taught point",
    "hidden_error": "Off its taught point, invisible to the motor encoders",
    "repeatability": "Accumulated drift over cycles",
    "torque_margin": "Running out of motor torque",
    "joint_limit": "Joint limit",
    "singularity": "Singularity (joint speed spike)",
    "unreachable": "Outside the working envelope",
    "vibration": "Residual vibration at stop",
    "table_collision": "Tool below the table",
}


def _deadzone(x, half):
    return np.where(x > half, x - half, np.where(x < -half, x + half, 0.0))


@dataclass
class Conditions:
    """Stress axes resolved against the program's own payload/tooling."""
    payload_kg: float
    cog: tuple
    torque_scale: float
    corner_scale: float
    friction_scale: float
    backlash_scale: float
    stiffness_scale: float
    wm_scale: float
    ff_scale: float = 1.0
    gain_scale: float = 1.0

    @classmethod
    def from_stress(cls, prog: Program, st: dict) -> "Conditions":
        cog = np.asarray(prog.payload_cog_mm, float) * 1e-3
        cog = cog + np.array([st.get("cog_x_mm", 0.0), st.get("cog_y_mm", 0.0),
                              st.get("cog_z_mm", 0.0)]) * 1e-3
        return cls(
            payload_kg=float(st.get("payload_kg", prog.payload_kg)),
            cog=tuple(float(c) for c in cog),
            torque_scale=float(st.get("torque_scale", 1.0)),
            corner_scale=float(st.get("supply_scale", 1.0)),
            friction_scale=float(st.get("friction_scale", 1.0)),
            backlash_scale=float(st.get("backlash_scale", 1.0)),
            stiffness_scale=float(st.get("stiffness_scale", 1.0)),
            wm_scale=float(st.get("world_model_scale", 0.0)),
            ff_scale=float(st.get("ff_scale", 1.0)),
            gain_scale=float(st.get("gain_scale", 1.0)),
        )


def simulate(prog: Program, stress: dict | None = None, **kw) -> dict:
    """Run ``prog`` under one set of stress conditions (keeps the 50 Hz replay trace)."""
    kw.setdefault("keep_trace", True)
    return simulate_batch(prog, [stress or {}], **kw)[0]


def _cols(items, f, B):
    return np.array([[getattr(x, f) for x in items]] * B, float)


def _per_world(conds, f):
    return np.array([[getattr(c, f)] for c in conds], float)


class StepperDrive:
    """The AR4: stepper rotor, belt/planetary with backlash and compliance, motor-shaft
    encoders, and the ROS 2 driver's controller configurations (see module docstring)."""

    aux_label = "load angle (electrical deg)"
    aux_lines = (90.0, -90.0)
    stop_mode = "trajectory_abort"

    def __init__(self, robot, conds, mode, jtc_gains=None):
        spec = ar4.AR4Spec(model=robot.id.split("_")[-1])
        B = len(conds)
        self.B, self.mode = B, mode
        mot = spec.motors
        self.G = spec.ratios[None]
        self.Nr = float(spec.rotor_teeth)
        self.micro = self.G / spec.steps_per_rad          # motor rad per (micro)step pulse
        self.enc_q = 1.0 / spec.enc_counts_per_joint_rad[None]
        self.T_hold = _cols(mot, "holding_nm", B) * spec.efficiency * _per_world(conds, "torque_scale")
        self.Jr = _cols(mot, "rotor_inertia", B)
        self.corner = _cols(mot, "corner_rps", B) * 2 * math.pi * _per_world(conds, "corner_scale")
        self.Kg = _cols(mot, "stiffness_nm_rad", B) * _per_world(conds, "stiffness_scale")
        self.half_bl = np.deg2rad(_cols(mot, "backlash_deg", B)) * _per_world(conds, "backlash_scale") / 2
        Ke = self.T_hold * self.Nr
        self.Be = 2 * 0.3 * np.sqrt(Ke * self.Jr)            # the drive's electrical damping
        self.Cg = 2 * 0.05 * np.sqrt(self.Kg * (_cols(mot, "armature", B) + 0.01))
        self.max_rate = spec.max_step_rate * self.micro      # motor rad/s
        self.v_fw = math.radians(ar4.FIRMWARE_MAX_SPEED_DEG_S)
        self.a_fw = math.radians(ar4.FIRMWARE_MAX_ACCEL_DEG_S2)
        self.gains = {**ar4.JTC_GAINS, **(jtc_gains or {})}
        self.ctrl_every = max(1, int(round(1000.0 / ar4.JTC_RATE_HZ)))
        self.path_tol = ar4.JTC_PATH_TOL_RAD
        self.goal_tol = ar4.JTC_GOAL_TOL_RAD if mode in ROS2_MODES else None
        self.slips = np.zeros((B, 6), int)
        self.peak_util = np.zeros((B, 6))
        self.max_path_err = np.zeros((B, 6))
        self.aborted = np.zeros(B, bool)
        self.aborted_at = [None] * B
        self.abort_joint = [None] * B
        self.abort_why = [None] * B

    def init(self, q0, tau_g, Mdiag):
        """Start as after homing: steps commanded to the start pose, the rotor lagging by
        the load angle that holds gravity, the link sagging by wind-up and backlash."""
        G = self.G
        self.th_c = G * q0
        self.th_r = self.th_c - np.arcsin(np.clip(tau_g / G / self.T_hold, -0.99, 0.99)) / self.Nr
        self.w_r = np.zeros_like(q0)
        self.w_c = np.zeros_like(q0)
        self.v_cmd = np.zeros_like(q0)
        self.integ = np.zeros_like(q0)
        self.q_enc_prev = None
        self.mt_target = self.th_c.copy()
        self.prev_cyc = np.zeros(q0.shape, int)
        self.th_cq = self.th_c.copy()
        self.T_av = self.T_hold.copy()
        self.tau = np.zeros_like(q0)
        return self.th_r / G - (tau_g / self.Kg + np.sign(tau_g) * self.half_bl)

    def encoder(self):
        return np.round(self.th_r / self.G / self.enc_q) * self.enc_q

    def control(self, k, q_ref, qd_ref, q_enc, active, n_ref, bias):
        G = self.G
        if self.mode in ROS2_MODES:
            if k % self.ctrl_every == 0:
                err = q_ref - q_enc
                if self.mode == "ros2":
                    # position+velocity command interfaces: the JTC streams the desired
                    # velocity (MV), no PID; the firmware clips at 60 deg/s
                    v_new = np.clip(qd_ref, -self.v_fw, self.v_fw)
                elif self.mode == "ros2_pid":
                    # velocity-only interface: PID, d on the velocity error, with the
                    # velocity the firmware reports (encoder delta over the period)
                    g = self.gains
                    dtc = self.ctrl_every * 1e-3
                    self.integ = np.clip(self.integ + err * dtc, -g["i_clamp"], g["i_clamp"])
                    v_meas = (np.zeros_like(err) if self.q_enc_prev is None
                              else (q_enc - self.q_enc_prev) / dtc)
                    v_new = np.clip(qd_ref + g["p"] * err + g["i"] * self.integ
                                    + g["d"] * (qd_ref - v_meas), -self.v_fw, self.v_fw)
                else:
                    # ros2_position (MT): the firmware re-targets each stepper by the
                    # encoder error, so lost steps are made up
                    v_new = None
                    mt_new = self.th_c + G * err
                bad = (np.abs(err) > self.path_tol).any(1) & (k < n_ref) & ~self.aborted
                for b in np.flatnonzero(bad):
                    self.aborted_at[b] = k * 1e-3
                    self.abort_joint[b] = int(np.argmax(np.abs(err[b]))) + 1
                    self.abort_why[b] = (f"J{self.abort_joint[b]} path error > "
                                         f"{math.degrees(self.path_tol):.1f} deg")
                self.aborted |= bad
                if v_new is not None:
                    self.v_cmd = np.where(self.aborted[:, None], 0.0, v_new)
                else:
                    self.mt_target = np.where(self.aborted[:, None], self.th_c, mt_new)
                self.q_enc_prev = q_enc
                self.max_path_err = np.maximum(self.max_path_err, np.abs(err) * active)
            if self.mode == "ros2_position":
                # AccelStepper toward the target: 60 deg/s cap, 30 deg/s^2 ramps
                dist = self.mt_target - self.th_c
                a = self.a_fw * G * 1e-3
                v_des = np.sign(dist) * np.minimum(self.v_fw * G,
                                                   np.sqrt(2 * self.a_fw * G * np.abs(dist)))
                self.w_c = self.w_c + np.clip(v_des - self.w_c, -a, a)
            else:
                self.w_c = self.v_cmd * G
            self.w_c = np.clip(self.w_c, -self.max_rate, self.max_rate)
        else:
            self.w_c = np.clip((G * q_ref - self.th_c) / 1e-3, -self.max_rate, self.max_rate)

    def step(self, dt, q, qd):
        G, Nr = self.G, self.Nr
        self.th_c = self.th_c + self.w_c * dt
        self.th_cq = np.round(self.th_c / self.micro) * self.micro   # step pulses are discrete
        sub = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            self.T_av = self.T_hold / np.sqrt(1.0 + (self.w_r / self.corner) ** 2)
            tau_em = self.T_av * np.sin(Nr * (self.th_cq - self.th_r)) + self.Be * (self.w_c - self.w_r)
            dz = _deadzone(self.th_r / G - q, self.half_bl)
            self.tau = self.Kg * dz + self.Cg * (self.w_r / G - qd) * (dz != 0.0)
            self.w_r = self.w_r + sub * (tau_em - self.tau / G) / self.Jr
            self.th_r = self.th_r + sub * self.w_r
        self.peak_util = np.maximum(self.peak_util, np.abs(self.tau / G) / self.T_av)
        return self.tau

    def after_ms(self, k, n_ref):
        # slips: the electrical cycle the rotor sits in moved (4 full steps per cycle)
        cyc = np.round(self.Nr * (self.th_cq - self.th_r) / (2 * math.pi)).astype(int)
        self.slips += np.abs(cyc - self.prev_cyc)
        self.prev_cyc = cyc

    def aux(self, b):
        return np.degrees(self.Nr * (self.th_cq[b] - self.th_r[b]))

    def util(self, b):
        return np.abs(self.tau[b] / self.G[0]) / self.T_av[b]


class ServoDrive:
    """A brushless servo through a harmonic drive (UR5e, xArm 6), joint-side quantities:
    motor state (angle, speed) on a motor-side encoder, a cascaded position loop at the
    robot's servo rate (PID tuned to ``bandwidth_hz`` on the joint's effective inertia),
    gravity/Coriolis feed-forward from the dynamics model (``ff_scale`` < 1 is a payload
    configured too light), a peak-torque limit, and a protective stop when following error
    exceeds ``following_stop_rad`` or torque stays saturated ``saturation_stop_ms``."""

    aux_label = "harmonic-drive wind-up (arcmin)"
    aux_lines = ()
    stop_mode = "protective_stop"

    def __init__(self, robot, conds, mode, jtc_gains=None):
        B = len(conds)
        self.B = B
        sj = robot.servo
        c = robot.controller
        self.tmax = _cols(sj, "torque_max", B) * _per_world(conds, "torque_scale")
        self.Jm = _cols(sj, "motor_inertia", B)
        self.Kg = _cols(sj, "stiffness", B) * _per_world(conds, "stiffness_scale")
        self.half_bl = np.deg2rad(_cols(sj, "backlash_deg", B)) * _per_world(conds, "backlash_scale") / 2
        self.Cg = 2 * 0.05 * np.sqrt(self.Kg * self.Jm)
        self.ff = _per_world(conds, "ff_scale")
        self.wb = 2 * math.pi * c["bandwidth_hz"] * _per_world(conds, "gain_scale")
        self.zeta = c["zeta"]
        # the drive closes its loop at 1 kHz; the robot controller's setpoints arrive at
        # rate_hz (UR 500 Hz, xArm 250 Hz) and are held in between
        self.ctrl_every = 1
        self.cmd_every = max(1, int(round(1000.0 / c["rate_hz"])))
        self.stop_rad = c["following_stop_rad"]
        self.sat_stop = int(c["saturation_stop_ms"])
        self.enc_res = 2 * math.pi / (2 ** 18 * _cols(sj, "gear", 1))   # joint rad per count
        self.vmax = robot.vmax[None]
        self.goal_tol = None
        self.slips = np.zeros((B, 6), int)
        self.peak_util = np.zeros((B, 6))
        self.max_path_err = np.zeros((B, 6))
        self.aborted = np.zeros(B, bool)
        self.aborted_at = [None] * B
        self.abort_joint = [None] * B
        self.abort_why = [None] * B

    def init(self, q0, tau_g, Mdiag):
        J = self.Jm + Mdiag
        self.Kp = J * self.wb ** 2
        self.Kd = 2 * self.zeta * self.wb * J
        self.Ki = self.Kp * self.wb / 8.0
        self.th = q0.copy()                      # the loop holds the motor at the command
        self.w = np.zeros_like(q0)
        self.integ = tau_g * (1 - self.ff) / np.maximum(self.Ki, 1e-9)
        self.tau_cmd = tau_g.copy()
        self.tau = tau_g.copy()
        self.u_raw = tau_g.copy()
        self.sat_ms = np.zeros(q0.shape, int)
        self.hold = None
        self.held = None
        return q0 - (tau_g / self.Kg + np.sign(tau_g) * self.half_bl)

    def encoder(self):
        return np.round(self.th / self.enc_res) * self.enc_res

    def control(self, k, q_ref, qd_ref, q_enc, active, n_ref, bias):
        if self.hold is not None:            # protective stop: hold where it stopped
            q_ref = np.where(self.aborted[:, None], self.hold, q_ref)
            qd_ref = np.where(self.aborted[:, None], 0.0, qd_ref)
        if k % self.cmd_every == 0 or self.held is None:
            self.held = (q_ref, qd_ref)
        q_ref, qd_ref = self.held
        err = q_ref - q_enc
        self.max_path_err = np.maximum(self.max_path_err, np.abs(err) * active)
        dtc = 1e-3
        if True:   # noqa: the drive loop runs every millisecond
            self.integ = self.integ + err * dtc
            u = (self.Kp * err + self.Kd * (np.clip(qd_ref, -self.vmax, self.vmax) - self.w)
                 + self.Ki * self.integ + self.ff * bias)
            self.u_raw = u
            self.tau_cmd = np.clip(u, -self.tmax, self.tmax)
            sat = np.abs(u) > self.tmax
            self.integ = np.where(sat, self.integ - err * dtc, self.integ)   # anti-windup
            self.sat_ms = np.where(sat, self.sat_ms + 1, 0)
        self.peak_util = np.maximum(self.peak_util, np.abs(self.u_raw) / self.tmax)
        follow = (np.abs(err) > self.stop_rad).any(1)
        torque = (self.sat_ms >= self.sat_stop).any(1)
        bad = (follow | torque) & (k < n_ref) & ~self.aborted
        for b in np.flatnonzero(bad):
            self.aborted_at[b] = k * 1e-3
            if torque[b]:
                j = int(np.argmax(self.sat_ms[b])) + 1
                self.abort_why[b] = f"J{j} at its torque limit for {self.sat_stop} ms"
            else:
                j = int(np.argmax(np.abs(err[b]))) + 1
                self.abort_why[b] = (f"J{j} following error > "
                                     f"{math.degrees(self.stop_rad):.1f} deg")
            self.abort_joint[b] = j
        if bad.any():
            self.hold = np.where(bad[:, None], q_enc, self.hold if self.hold is not None else q_enc)
            self.aborted |= bad
            self.integ = np.where(bad[:, None], 0.0, self.integ)

    def step(self, dt, q, qd):
        sub = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            dz = _deadzone(self.th - q, self.half_bl)
            self.tau = self.Kg * dz + self.Cg * (self.w - qd) * (dz != 0.0)
            self.w = self.w + sub * (self.tau_cmd - self.tau) / self.Jm
            self.th = self.th + sub * self.w
        return self.tau

    def after_ms(self, k, n_ref):
        pass

    def aux(self, b):
        return np.degrees(self.tau[b] / self.Kg[b]) * 60.0

    def util(self, b):
        return np.abs(self.u_raw[b]) / self.tmax[b]


def simulate_batch(prog: Program, stresses: list, *, spec=None, world_model=None,
                   kin: Kinematics | None = None, keep_trace: bool = False,
                   jtc_gains: dict | None = None, taught: dict | None = None) -> list[dict]:
    """Run ``prog`` once per entry of ``stresses``, all worlds stepped in lockstep (numpy
    over worlds x joints for the drive train, one MuJoCo model per world).

    ``taught`` (from ``taught_offsets``) holds, per taught point, where the nominal run
    really put the TCP relative to the ideal kinematics. Points on a real robot are taught by
    jogging, so the static sag of the nominal robot is already in them; with ``taught``
    the positioning error is measured against that, not against ideal kinematics."""
    import mujoco

    sts = [dict(s or {}) for s in stresses]
    B = len(sts)
    robot = get_robot(prog.model)
    kin = kin or Kinematics(prog.model)
    refs = [build_reference(prog, st, rate_hz=1000.0, kin=kin) for st in sts]
    conds = [Conditions.from_stress(prog, st) for st in sts]

    models, datas = [], []
    for c in conds:
        m = build_model(robot, payload_kg=c.payload_kg, payload_cog=c.cog)
        m.dof_frictionloss[:] *= c.friction_scale
        m.dof_damping[:] *= c.friction_scale
        models.append(m)
        datas.append(mujoco.MjData(m))
    tcp_site = models[0].site("tcp").id
    dt = float(models[0].opt.timestep)
    lim = robot.limits_rad
    mode = prog.control
    drive = (ServoDrive if robot.drive == "servo" else StepperDrive)(robot, conds, mode, jtc_gains)
    wm_scale = np.array([[c.wm_scale] for c in conds])
    wm = world_model if (world_model is not None and np.any(wm_scale != 0.0)) else None

    n_ref = np.array([len(r.t) for r in refs])
    T_max = int(n_ref.max())
    Qr = np.empty((B, T_max, 6))
    QDr = np.zeros((B, T_max, 6))
    for b, r in enumerate(refs):
        Qr[b, :n_ref[b]] = r.q
        Qr[b, n_ref[b]:] = r.q[-1]
        QDr[b, :n_ref[b]] = r.qd
    seg_end = [set(r.seg_end) for r in refs]
    meas_at = []
    for r in refs:
        mm = {}
        for (i, lab, c) in r.measure:
            mm.setdefault(i, []).append((lab, c))
        meas_at.append(mm)

    tau_g = np.zeros((B, 6))
    Mdiag = np.zeros((B, 6))
    for b in range(B):
        datas[b].qpos[:] = Qr[b, 0]
        mujoco.mj_forward(models[b], datas[b])
        tau_g[b] = datas[b].qfrc_bias
        M = np.zeros((6, 6))
        mujoco.mj_fullM(models[b], datas[b], M)
        Mdiag[b] = np.diag(M)
    q = drive.init(Qr[:, 0].copy(), tau_g, Mdiag)
    qd = np.zeros((B, 6))
    bias = tau_g.copy()
    for b in range(B):
        datas[b].qpos[:] = q[b]
        mujoco.mj_forward(models[b], datas[b])
    goal_viol = [[] for _ in range(B)]
    diverged = np.zeros(B, bool)
    tcp_at = [{} for _ in range(B)]
    tcp_log = [[] for _ in range(B)]          # (t, xyz) at SAMPLE_HZ, for vibration/table
    tr = [{k: [] for k in ("t", "q", "q_ref", "q_enc", "aux", "util", "slips", "tcp",
                           "tcp_ref", "tcp_enc", "xpos", "xquat")} for _ in range(B)] \
        if keep_trace else None

    steps_per_ref = int(round(1e-3 / dt))
    total = T_max + 1000                                            # 1 s to settle
    keep_every = int(round(1000.0 / SAMPLE_HZ))
    rows = np.arange(B)
    corr_prev = None
    ctrl_every = getattr(drive, "ctrl_every", 1)

    for k in range(total):
        kr = np.minimum(k, n_ref - 1)
        active = (k < n_ref)[:, None]
        q_ref = Qr[rows, kr]
        qd_ref = QDr[rows, kr] * active
        if wm is not None:
            # the correction moves the commanded trajectory: position and, differenced
            # over the controller period, velocity (the AR4 streams velocities at 60 Hz)
            corr = wm_scale * wm.correction(q_ref, qd_ref)
            q_ref = q_ref + corr
            if k % ctrl_every == 0:
                if corr_prev is not None:
                    qd_ref = qd_ref + (corr - corr_prev) / (ctrl_every * 1e-3)
                corr_prev = corr
        q_enc = drive.encoder()
        drive.control(k, q_ref, qd_ref, q_enc, active, n_ref, bias)

        for _ in range(steps_per_ref):
            tau = drive.step(dt, q, qd)
            for b in range(B):
                d = datas[b]
                d.qfrc_applied[:] = tau[b]
                mujoco.mj_step(models[b], d)
                q[b] = d.qpos
                qd[b] = d.qvel
                bias[b] = d.qfrc_bias
        drive.after_ms(k, n_ref)
        bad = ~np.isfinite(q).all(1)
        if bad.any():
            diverged |= bad
            q[bad] = 0.0
            qd[bad] = 0.0

        if drive.goal_tol is not None:
            for b in range(B):
                if k in seg_end[b]:
                    ge = np.abs(q_ref[b] - q_enc[b])
                    if ge.max() > drive.goal_tol:
                        goal_viol[b].append({"t": k * 1e-3, "joint": int(np.argmax(ge)) + 1,
                                             "err_deg": math.degrees(float(ge.max()))})

        sample = k % keep_every == 0
        for b in range(B):
            meas = meas_at[b].get(k) if k < n_ref[b] else None
            if not (sample or meas):
                continue
            p_true = datas[b].site_xpos[tcp_site].copy()
            if sample:
                tcp_log[b].append((k * 1e-3, p_true))
            if meas or keep_trace:
                p_ref, _ = kin.fk(Qr[b, kr[b]])
                p_enc, _ = kin.fk(q_enc[b])
                for (label, c) in meas or ():
                    tcp_at[b][(label, c)] = (p_true, p_ref, p_enc)
            if keep_trace and sample:
                t_b = tr[b]
                t_b["t"].append(k * 1e-3)
                t_b["q"].append(q[b].copy())
                t_b["q_ref"].append(Qr[b, kr[b]].copy())
                t_b["q_enc"].append(q_enc[b].copy())
                t_b["aux"].append(drive.aux(b))
                t_b["util"].append(drive.util(b))
                t_b["slips"].append(drive.slips[b].copy())
                t_b["tcp"].append(p_true)
                t_b["tcp_ref"].append(p_ref)
                t_b["tcp_enc"].append(p_enc)
                t_b["xpos"].append(datas[b].xpos[1:].copy())
                t_b["xquat"].append(datas[b].xquat[1:].copy())

    out = []
    for b in range(B):
        if diverged[b]:
            out.append(_invalid(prog, sts[b], conds[b], "simulation diverged"))
            continue
        r = _score(prog, sts[b], conds[b], refs[b], q[b], kin, drive.slips[b],
                   drive.peak_util[b], drive.max_path_err[b], goal_viol[b],
                   drive.aborted_at[b], drive.abort_why[b], tcp_at[b], tcp_log[b],
                   n_ref[b] * 1e-3, tr[b] if keep_trace else None, lim, taught,
                   drive=drive, robot=robot)
        r["robot"] = robot.id
        if keep_trace and "trace" in r:
            r["trace"]["aux_label"] = drive.aux_label
            r["trace"]["aux_lines"] = list(drive.aux_lines)
        out.append(r)
    return out


def taught_offsets(prog: Program, *, world_model=None, kin=None, spec=None) -> dict:
    """Run the program once at nominal conditions and record, per taught point, the TCP
    offset (true - ideal) and the encoder-blind offset (true - encoder FK) of cycle 1."""
    st = {"cycles": 1, **(world_model.nominal_stress() if world_model is not None else {})}
    r = simulate_batch(prog, [st], world_model=world_model, kin=kin)[0]
    return {p["label"]: {"err": (np.array(p["xyz_mm"]) - np.array(p["ref_mm"])).tolist(),
                         "hidden": p["hidden_vec_mm"]}
            for p in r.get("taught_points", []) if p["cycle"] == 1}


def _invalid(prog, st, cond, why):
    return {"valid": False, "invalid_reason": why, "stress": st, "failure_score": 0.0,
            "failed": False, "modes": {}, "dominant": None, "metrics": {}, "events": []}


def _score(prog, st, cond, ref, q_end, kin, slips, peak_util, max_path_err, goal_viol,
           aborted_at, abort_why, tcp_at, tcp_log, t_end, tr, lim, taught=None, drive=None,
           robot=None):
    tol = float(st.get("tolerance_mm", prog.tolerance_mm))
    events = []
    sev = {}

    lost = int(slips.sum()) * 4
    sev["missed_steps"] = 0.0 if lost == 0 else 1.0 + min(lost / 40.0, 1.0)
    if lost:
        j = int(np.argmax(slips)) + 1
        events.append(f"J{j} lost {int(slips[j - 1]) * 4} full steps "
                      f"({', '.join(f'J{i + 1}:{int(s) * 4}' for i, s in enumerate(slips) if s)})")

    stop_mode = drive.stop_mode if drive is not None else "trajectory_abort"
    sev["trajectory_abort"] = 0.0
    sev["protective_stop"] = 0.0
    if aborted_at is not None:
        sev[stop_mode] = 1.5
        what = "protective stop" if stop_mode == "protective_stop" else "controller aborted"
        events.append(f"{what} at t={aborted_at:.2f}s: {abort_why}")

    errs, hidden, seen_by_enc, points = [], [], [], []
    by_label = {}
    abs_errs = []
    for (label, c), (pt, pr, pe) in sorted(tcp_at.items(), key=lambda x: x[0][1]):
        tv = (taught or {}).get(label)
        off = np.array(tv["err"]) * 1e-3 if tv else 0.0
        hoff = np.array(tv["hidden"]) * 1e-3 if tv else 0.0
        abs_errs.append(float(np.linalg.norm(pt - pr)) * 1e3)
        e = float(np.linalg.norm(pt - pr - off)) * 1e3
        h = float(np.linalg.norm(pt - pe - hoff)) * 1e3
        errs.append(e)
        hidden.append(h)
        seen_by_enc.append(float(np.linalg.norm(pe - pr)) * 1e3)
        by_label.setdefault(label, []).append(pt)
        points.append({"label": label, "cycle": c + 1, "err_mm": round(e, 4),
                       "hidden_mm": round(h, 4), "xyz_mm": (pt * 1e3).round(3).tolist(),
                       "ref_mm": (pr * 1e3).round(3).tolist(),
                       "hidden_vec_mm": ((pt - pe) * 1e3).round(4).tolist()})
    max_err = max(errs) if errs else 0.0
    max_hidden = max(hidden) if hidden else 0.0
    sev["tcp_error"] = max_err / tol
    sev["hidden_error"] = max_hidden / tol if max_hidden > 0.5 * max_err else 0.0
    drift = 0.0
    for label, pts in by_label.items():
        if len(pts) > 1:
            P = np.asarray(pts)
            drift = max(drift, float(np.max(np.linalg.norm(P - P[0], axis=1))) * 1e3)
    sev["repeatability"] = drift / tol
    if drift > tol:
        events.append(f"the same taught point drifted {drift:.2f} mm across cycles")

    util = float(peak_util.max())
    if robot is not None and robot.drive == "servo":
        # a servo may touch its peak torque briefly; the protective stop covers sustained
        sev["torque_margin"] = max(0.0, (util - 0.7) / 0.35)
    else:
        sev["torque_margin"] = max(0.0, (util - 0.6) / 0.35)   # > 95% of pull-out fails
    over = np.maximum(lim[:, 0] - q_end, q_end - lim[:, 1])
    sev["joint_limit"] = 0.0
    for it in ref.issues:
        if it["kind"] == "joint_limit":
            sev["joint_limit"] = 1.2
            events.append(it["detail"])
    if (over > math.radians(1.0)).any():
        sev["joint_limit"] = 1.2
        events.append(f"J{int(np.argmax(over)) + 1} ran past its limit")
    sev["singularity"] = 0.0
    sev["unreachable"] = 0.0
    for it in ref.issues:
        if it["kind"] in ("singularity", "unreachable"):
            sev[it["kind"]] = 1.1
            events.append(it["detail"])

    # residual vibration: TCP motion 0.4-1.0 s after this world's program ended
    vib = 0.0
    tcp_min_z = None
    if tcp_log:
        ts = np.array([x[0] for x in tcp_log])
        P = np.array([x[1] for x in tcp_log])
        tail = P[(ts >= t_end + 0.4) & (ts <= t_end + 1.0)]
        if len(tail) > 1:
            vib = float(np.max(np.linalg.norm(tail - tail.mean(0), axis=1))) * 1e3
        tcp_min_z = float(P[ts <= t_end + 1.0, 2].min())
    sev["vibration"] = vib / tol
    sev["table_collision"] = 0.0
    if tcp_min_z is not None and tcp_min_z < 0.0:
        sev["table_collision"] = 1.3
        events.append(f"tool flange went {abs(tcp_min_z) * 1e3:.0f} mm below the table")
    g_err = max((g["err_deg"] for g in goal_viol), default=0.0)
    sev["goal_tolerance"] = math.radians(g_err) / ar4.JTC_GOAL_TOL_RAD if goal_viol else 0.0
    if robot is not None and robot.drive == "servo":
        sev["missed_steps"] = 0.0
    if goal_viol:
        g = max(goal_viol, key=lambda x: x["err_deg"])
        events.append(f"goal tolerance missed at t={g['t']:.2f}s (J{g['joint']} off by "
                      f"{g['err_deg']:.2f} deg)")
    if max_err > tol:
        events.append(f"TCP off by {max_err:.2f} mm at a taught point (tolerance {tol:g} mm); "
                      f"{max_hidden:.2f} mm of it invisible to the encoders")

    sev = {k: min(v, 5.0) for k, v in sev.items()}
    score = float(max(sev.values()))
    # name the failure by its cause: the first failing mode in causal order
    dominant = next((m for m in CAUSAL_ORDER if sev.get(m, 0.0) >= 1.0), max(sev, key=sev.get))
    metrics = {
        "lost_steps": lost, "lost_steps_by_joint": [int(s) * 4 for s in slips],
        "max_tcp_error_mm": round(max_err, 4), "max_hidden_error_mm": round(max_hidden, 4),
        "max_absolute_error_mm": round(max(abs_errs) if abs_errs else 0.0, 4),
        "error_reference": "taught (nominal run)" if taught else "ideal kinematics",
        "max_encoder_visible_error_mm": round(max(seen_by_enc) if seen_by_enc else 0.0, 4),
        "repeat_drift_mm": round(drift, 4), "peak_torque_util": round(util, 4),
        "torque_util_by_joint": [round(float(u), 3) for u in peak_util],
        "max_path_err_deg": [round(math.degrees(float(x)), 4) for x in max_path_err],
        "residual_vibration_mm": round(vib, 4), "aborted_at_s": aborted_at,
        "goal_violations": len(goal_viol), "min_manipulability": ref.min_manip,
        "peak_joint_speed_ratio": round(ref.peak_speed_ratio, 3),
        "duration_s": round(float(ref.t[-1]), 3), "tolerance_mm": tol,
    }
    out = {"valid": True, "stress": st, "conditions": cond.__dict__, "failure_score": score,
           "failed": score >= 1.0, "modes": {k: round(v, 4) for k, v in sev.items()},
           "dominant": dominant if score >= 1.0 else None, "metrics": metrics,
           "events": events, "issues": ref.issues, "taught_points": points}
    if tr is not None:
        out["trace"] = {k: np.asarray(v).round(6).tolist() for k, v in tr.items()}
        out["trace"]["measure"] = [
            {"label": lab, "cycle": c, "t": i * 1e-3} for (i, lab, c) in ref.measure]
    return out
