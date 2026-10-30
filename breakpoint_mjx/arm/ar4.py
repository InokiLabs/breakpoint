"""The Annin Robotics AR4 as a MuJoCo model plus its drive-train constants.

Everything kinematic and inertial here is transcribed from the AR4 ROS 2 driver
(github.com/Annin-Robotics/ar4_ros_driver, MIT licence): joint origins, axes and link
inertials from ``annin_ar4_description/urdf/ar_macro.xacro``, joint limits from
``annin_ar4_description/config/mk*.yaml``, and the step/encoder constants from the Teensy
firmware ``annin_ar4_firmware/AR4_teensy/AR4_teensy.ino``. Controller gains, tolerances and
the 60 Hz update rate come from ``annin_ar4_driver/config/controllers.yaml``.

Derived, not copied: the gear ratio of each joint. The firmware gives motor steps per
joint degree and the encoder multiplier (encoder counts = motor steps x ENC_MULT, 4000
counts per motor revolution), so motor steps per revolution = 4000 / ENC_MULT and
ratio = steps_per_deg * 360 / steps_per_rev. On every model that gives
40, 50, 50, 44.8, 9.84, 20.

ASSUMED (no source in the repository; edit ``MotorSpec`` to match your build): stepper
holding torques, rotor inertias, the torque-speed corner, gearbox stiffness and
backlash. They are typical datasheet values for the motor frame sizes the AR4 uses and
are exactly the quantities the world model in ``world_model.py`` exists to calibrate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

JOINTS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
N_JOINTS = 6

# --- firmware constants (AR4_teensy.ino) ------------------------------------------------
STEPS_PER_DEG = {
    "mk3": (44.44444444, 55.55555556, 55.55555556, 49.77777777, 21.86024888, 22.22222222),
    "mk5": (88.888, 111.111, 111.111, 99.555, 43.720, 44.444),
}
ENC_MULT = {"mk3": (10, 10, 10, 10, 5, 10), "mk5": (5, 5, 5, 5, 2.5, 5)}
ENC_COUNTS_PER_MOTOR_REV = 4000
LIMITS_DEG = {
    "mk3": ((-170, 170), (-42, 90), (-89, 52), (-180, 180), (-105, 105), (-180, 180)),
    "mk5": ((-160, 160), (-42, 90), (-89, 52), (-180, 180), (-105, 105), (-180, 180)),
}
FIRMWARE_MAX_SPEED_DEG_S = 60.0    # JOINT_MAX_SPEED
FIRMWARE_MAX_ACCEL_DEG_S2 = 30.0   # JOINT_MAX_ACCEL (also MoveIt joint_limits.yaml)

# --- ros2_control joint_trajectory_controller (controllers.yaml) -------------------------
JTC_RATE_HZ = 60.0
JTC_GAINS = {"p": 10.0, "d": 1.0, "i": 1.0, "i_clamp": 1.0}
JTC_PATH_TOL_RAD = 0.2
JTC_GOAL_TOL_RAD = 0.02


def gear_ratios(model: str = "mk5") -> np.ndarray:
    spd = np.asarray(STEPS_PER_DEG[model])
    steps_per_rev = ENC_COUNTS_PER_MOTOR_REV / np.asarray(ENC_MULT[model], float)
    return spd * 360.0 / steps_per_rev


@dataclass(frozen=True)
class MotorSpec:
    """One joint's stepper + gearbox. Torques at the motor shaft, stiffness at the joint."""
    holding_nm: float          # stepper holding torque
    rotor_inertia: float       # kg m^2 at the motor shaft
    corner_rps: float          # torque-speed corner: pull-out torque halves near here
    stiffness_nm_rad: float    # gearbox/belt torsional stiffness at the joint
    backlash_deg: float        # total lost motion at the joint
    friction_nm: float         # Coulomb friction at the joint
    damping_nms: float         # viscous friction at the joint
    armature: float            # output-stage reflected inertia at the joint (kg m^2)


# Datasheet-typical values for the frame sizes on the AR4 bill of materials: NEMA 23
# (1.26 Nm) on J1/J2, NEMA 17 on J3-J5, NEMA 11 on J6. Stiffness and backlash are typical of
# hobby planetary/belt reductions. All of it is calibrated by world_model.py.
DEFAULT_MOTORS = (
    MotorSpec(1.26, 3.0e-5, 5.0, 3500.0, 0.05, 0.60, 0.20, 2e-3),
    MotorSpec(1.26, 3.0e-5, 5.0, 3000.0, 0.05, 0.80, 0.25, 2e-3),
    MotorSpec(0.40, 5.4e-6, 4.0, 1500.0, 0.05, 0.40, 0.10, 1e-3),
    MotorSpec(0.40, 5.4e-6, 4.0, 400.0, 0.08, 0.10, 0.03, 4e-4),
    MotorSpec(0.26, 3.5e-6, 4.0, 300.0, 0.10, 0.08, 0.02, 3e-4),
    MotorSpec(0.095, 1.8e-6, 6.0, 150.0, 0.10, 0.04, 0.01, 2e-4),
)


@dataclass(frozen=True)
class AR4Spec:
    model: str = "mk5"
    motors: tuple[MotorSpec, ...] = DEFAULT_MOTORS
    rotor_teeth: int = 50                   # 1.8 deg hybrid stepper
    efficiency: float = 0.85                # gearbox efficiency, derates motor torque
    max_step_rate: float = 40000.0          # step pulses per second per axis (Teensy 4.1)
    extra: dict = field(default_factory=dict)

    @property
    def ratios(self) -> np.ndarray:
        return gear_ratios(self.model)

    @property
    def steps_per_rad(self) -> np.ndarray:
        return np.asarray(STEPS_PER_DEG[self.model]) * 180.0 / math.pi

    @property
    def enc_counts_per_joint_rad(self) -> np.ndarray:
        return self.steps_per_rad * np.asarray(ENC_MULT[self.model], float)

    @property
    def limits_rad(self) -> np.ndarray:
        return np.deg2rad(np.asarray(LIMITS_DEG[self.model], float))


# --- kinematics + inertials (ar_macro.xacro), meshes stripped -----------------------------
# (name, parent, origin xyz, origin rpy, axis, mass, com, inertia ixx ixy ixz iyy iyz izz)
_LINKS = [
    ("link_1", "base_link", (0, 0, 0.092), (math.pi, 0, 0), (0, 0, 1), 0.88065,
     (-0.022706, 0.04294, -0.12205),
     (0.0034, 0.00042296, -0.00089231, 0.0041778, 0.0010848, 0.0027077)),
    ("link_2", "link_1", (0, 0.06415, -0.07778), (1.5708, 0, -1.5708), (0, 0, -1), 0.57738,
     (0.064818, -0.11189, -0.038671),
     (0.0047312, 0.0022624, 0.00032144, 0.0020836, -0.00056569, 0.0056129)),
    ("link_3", "link_2", (0, -0.305, 0), (0, 0, 3.1416), (0, 0, -1), 0.1787,
     (-0.00029765, -0.023661, -0.0019125),
     (0.0001685, -2.7713e-05, 5.6885e-06, 0.00012865, 2.9256e-05, 0.00020744)),
    ("link_4", "link_3", (0, 0, 0), (1.5708, 0, -1.5708), (0, 0, -1), 0.34936,
     (-0.0016798, -0.00057319, -0.074404),
     (0.0030532, -1.8615e-05, -7.0047e-05, 0.0031033, -2.3301e-05, 0.00022264)),
    ("link_5", "link_4", (0, 0, -0.22294), (math.pi, 0, -1.5708), (1, 0, 0), 0.11562,
     (0.0015066, -1.3102e-05, -0.012585),
     (5.5035e-05, -1.019e-08, -2.6243e-06, 8.2921e-05, 1.4437e-08, 5.2518e-05)),
    ("link_6", "link_5", (0, 0, 0.041), (0, 0, 0), (0, 0, 1), 0.013863,
     (2.9287e-10, -1.6472e-09, 0.0091432),
     (1.3596e-06, 3.0585e-13, 5.7102e-14, 1.7157e-06, 6.3369e-09, 2.4332e-06)),
]


def urdf_text(spec: AR4Spec = AR4Spec()) -> str:
    """A mesh-free URDF with the driver's exact frames, inertials and limits."""
    lim = spec.limits_rad

    def inertial(mass, com, I):
        ixx, ixy, ixz, iyy, iyz, izz = I
        return (f'<inertial><origin xyz="{com[0]} {com[1]} {com[2]}" rpy="0 0 0"/>'
                f'<mass value="{mass}"/><inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}" '
                f'iyy="{iyy}" iyz="{iyz}" izz="{izz}"/></inertial>')

    parts = ['<robot name="ar4">', '<link name="world"/>',
             '<link name="base_link">' + inertial(
                 0.7102, (-4.6941e-06, 0.054174, 0.038824),
                 (0.0039943, 3.697e-07, -5.7364e-08, 0.0014946, -0.00036051, 0.0042554))
             + '</link>',
             '<joint name="base_joint" type="fixed"><parent link="world"/>'
             '<child link="base_link"/><origin xyz="0 0 0" rpy="0 0 0"/></joint>']
    for i, (name, parent, xyz, rpy, axis, mass, com, I) in enumerate(_LINKS):
        parts.append(f'<link name="{name}">{inertial(mass, com, I)}</link>')
        parts.append(
            f'<joint name="joint_{i + 1}" type="revolute"><parent link="{parent}"/>'
            f'<child link="{name}"/><origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" '
            f'rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/><axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>'
            f'<limit lower="{lim[i, 0]}" upper="{lim[i, 1]}" effort="100" velocity="10"/>'
            '</joint>')
    parts.append("</robot>")
    return "\n".join(parts)


def build_model(spec: AR4Spec = AR4Spec(), *, payload_kg: float = 0.0,
                payload_cog=(0.0, 0.0, 0.0), timestep: float = 1e-3):
    """Compile the AR4 with a payload body on the flange. Joints are torque-driven through
    ``qfrc_applied`` by ``drive.py``; MuJoCo handles gravity, inertia and joint friction
    (frictionloss + damping, solved implicitly so stiction is exact)."""
    import mujoco

    s = mujoco.MjSpec.from_string(urdf_text(spec))
    s.option.timestep = timestep
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    s.compiler.balanceinertia = True

    flange = s.body("link_6")
    tool = flange.add_body(name="payload", pos=[float(c) for c in payload_cog])
    tool.mass = max(float(payload_kg), 1e-6)
    r = 0.03  # treat the payload as a 3 cm solid sphere about its CoG
    tool.inertia = [max(0.4 * tool.mass * r * r, 1e-9)] * 3
    tool.explicitinertial = True
    flange.add_site(name="tcp", pos=[0, 0, 0])

    for i, j in enumerate(JOINTS):
        jt = s.joint(j)
        m = spec.motors[i]
        jt.frictionloss = m.friction_nm
        # mujoco >= 3.4 stores damping as polynomial coefficients; [c, 0, 0] is linear
        jt.damping = [m.damping_nms, 0.0, 0.0] if np.ndim(jt.damping) else m.damping_nms
        jt.armature = m.armature
    return s.compile()


def model_info(spec: AR4Spec = AR4Spec()) -> dict:
    """Plain-data description of the robot for the UI and reports."""
    return {
        "name": f"Annin Robotics AR4 {spec.model.upper()}",
        "model": spec.model,
        "joints": list(JOINTS),
        "limits_deg": [list(x) for x in LIMITS_DEG[spec.model]],
        "gear_ratio": [round(float(g), 3) for g in spec.ratios],
        "steps_per_deg": list(STEPS_PER_DEG[spec.model]),
        "encoder_counts_per_joint_deg": [
            round(float(c) * math.pi / 180.0, 3) for c in spec.enc_counts_per_joint_rad],
        "max_speed_deg_s": FIRMWARE_MAX_SPEED_DEG_S,
        "max_accel_deg_s2": FIRMWARE_MAX_ACCEL_DEG_S2,
        "jtc": {"rate_hz": JTC_RATE_HZ, **JTC_GAINS, "path_tol_rad": JTC_PATH_TOL_RAD,
                "goal_tol_rad": JTC_GOAL_TOL_RAD},
        "motors": [m.__dict__ for m in spec.motors],
        "sources": [
            "github.com/Annin-Robotics/ar4_ros_driver (MIT): urdf/ar_macro.xacro, "
            "config/mk*.yaml, AR4_teensy.ino, annin_ar4_driver/config/controllers.yaml",
        ],
    }
