"""The arms the Stress Lab knows: kinematics, limits, drive train and controller per robot.

Two drive-train families, simulated differently (``sim.py``):

* ``stepper``: open-loop steppers through belts/planetaries (Annin AR4). See ``ar4.py``.
* ``servo``: brushless servos through harmonic drives with a cascaded position loop,
  model-based feed-forward, a torque limit and a protective stop on following error or
  sustained torque saturation (Universal Robots UR5e, UFACTORY xArm 6).

Kinematics, inertials, joint limits, effort and speed limits come from each maker's ROS
description (sources in ``RobotDef.sources``). Values the makers do not publish (harmonic
drive stiffness, reflected rotor inertia, servo bandwidth, protective-stop thresholds, the
100 % program acceleration) are marked ASSUMED and are what ``world_model.py`` calibrates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from breakpoint_mjx.arm import ar4


@dataclass(frozen=True)
class ServoJoint:
    """One servo joint, all quantities at the joint (output) side."""
    torque_max: float          # Nm, the drive's peak torque limit
    gear: float                # harmonic drive ratio
    motor_inertia: float       # reflected rotor inertia, kg m^2 (ASSUMED)
    stiffness: float           # harmonic drive torsional stiffness, Nm/rad (ASSUMED)
    backlash_deg: float        # lost motion, deg (harmonic drives: well under 1 arcmin)
    friction_nm: float         # Coulomb friction, Nm (ASSUMED)
    damping_nms: float         # viscous friction, Nm s/rad (ASSUMED)
    armature: float = 0.02     # output-stage inertia, kg m^2


@dataclass(frozen=True)
class RobotDef:
    id: str
    name: str
    maker: str
    drive: str                         # "stepper" | "servo"
    joints: tuple
    limits_deg: tuple                  # ((lo, hi),) * 6
    vmax_deg_s: tuple                  # joint speed limit
    v100_deg_s: float                  # what "speed: 100" in a joint move means
    a100_deg_s2: float                 # what "accel: 100" means
    links: tuple                       # (name, parent, xyz, rpy, mass, com, com_rpy, inertia6)
    base: tuple                        # (mass, com, inertia6)
    flange: str                        # link carrying the tool
    control_modes: tuple
    default_control: str
    rated_payload_kg: float
    repeatability_mm: float
    reach_mm: float
    home_deg: tuple
    servo: tuple = ()                  # ServoJoint per joint (servo drive)
    controller: dict = field(default_factory=dict)
    sources: tuple = ()
    stress_space: str = "configs/ar4/theta_space.yaml"
    timestep: float = 1e-3
    aliases: tuple = ()

    @property
    def limits_rad(self) -> np.ndarray:
        return np.deg2rad(np.asarray(self.limits_deg, float))

    @property
    def vmax(self) -> np.ndarray:
        return np.deg2rad(np.asarray(self.vmax_deg_s, float))

    def urdf_text(self) -> str:
        lim = self.limits_rad

        def inertial(mass, com, rpy, I):
            ixx, ixy, ixz, iyy, iyz, izz = I
            return (f'<inertial><origin xyz="{com[0]} {com[1]} {com[2]}" '
                    f'rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/><mass value="{mass}"/>'
                    f'<inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}" iyy="{iyy}" iyz="{iyz}" '
                    f'izz="{izz}"/></inertial>')

        bm, bc, bI = self.base
        parts = [f'<robot name="{self.id}">', '<link name="world"/>',
                 f'<link name="base_link">{inertial(bm, bc, (0, 0, 0), bI)}</link>',
                 '<joint name="base_joint" type="fixed"><parent link="world"/>'
                 '<child link="base_link"/><origin xyz="0 0 0" rpy="0 0 0"/></joint>']
        for i, (name, parent, xyz, rpy, axis, mass, com, crpy, I) in enumerate(self.links):
            parts.append(f'<link name="{name}">{inertial(mass, com, crpy, I)}</link>')
            parts.append(
                f'<joint name="{self.joints[i]}" type="revolute"><parent link="{parent}"/>'
                f'<child link="{name}"/><origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" '
                f'rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/><axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>'
                f'<limit lower="{lim[i, 0]}" upper="{lim[i, 1]}" effort="1000" velocity="10"/>'
                '</joint>')
        parts.append("</robot>")
        return "\n".join(parts)

    def info(self) -> dict:
        out = {"id": self.id, "name": self.name, "maker": self.maker, "drive": self.drive,
               "joints": list(self.joints), "limits_deg": [list(x) for x in self.limits_deg],
               "vmax_deg_s": list(self.vmax_deg_s), "v100_deg_s": self.v100_deg_s,
               "a100_deg_s2": self.a100_deg_s2, "rated_payload_kg": self.rated_payload_kg,
               "repeatability_mm": self.repeatability_mm, "reach_mm": self.reach_mm,
               "control_modes": list(self.control_modes), "controller": self.controller,
               "sources": list(self.sources)}
        if self.drive == "servo":
            out["servo"] = [s.__dict__ for s in self.servo]
        else:
            out["ar4"] = ar4.model_info(ar4.AR4Spec(model=self.id.split("_")[-1]))
        return out


def _ar4(model):
    L = ar4._LINKS
    links = tuple((n, p, xyz, rpy, ax, m, com, (0, 0, 0), I) for (n, p, xyz, rpy, ax, m, com, I) in L)
    return RobotDef(
        id=f"ar4_{model}", name=f"Annin Robotics AR4 {model.upper()}", maker="Annin Robotics",
        drive="stepper", joints=ar4.JOINTS, limits_deg=ar4.LIMITS_DEG[model],
        vmax_deg_s=(ar4.FIRMWARE_MAX_SPEED_DEG_S,) * 6, v100_deg_s=ar4.FIRMWARE_MAX_SPEED_DEG_S,
        a100_deg_s2=ar4.FIRMWARE_MAX_ACCEL_DEG_S2, links=links,
        base=(0.7102, (-4.6941e-06, 0.054174, 0.038824),
              (0.0039943, 3.697e-07, -5.7364e-08, 0.0014946, -0.00036051, 0.0042554)),
        flange="link_6", control_modes=("ros2", "ros2_pid", "ros2_position", "open_loop"),
        default_control="ros2", rated_payload_kg=2.0, repeatability_mm=0.2, reach_mm=629,
        home_deg=(0, 0, 0, 0, 90, 0),
        controller={"rate_hz": ar4.JTC_RATE_HZ, **ar4.JTC_GAINS,
                    "path_tol_rad": ar4.JTC_PATH_TOL_RAD, "goal_tol_rad": ar4.JTC_GOAL_TOL_RAD},
        sources=("github.com/Annin-Robotics/ar4_ros_driver (MIT): ar_macro.xacro, mk*.yaml, "
                 "AR4_teensy.ino, controllers.yaml",),
        stress_space="configs/ar4/theta_space.yaml", aliases=(model,))


H = math.pi / 2
# UR5e: Universal_Robots_ROS2_Description @81f7c9f (BSD-3): default_kinematics.yaml,
# physical_parameters.yaml, joint_limits.yaml. base_link_inertia is rotated pi about z.
_UR_I = {
    "shoulder": (0.00700210, 0.00000073, -0.00001053, 0.00648091, 0.00049994, 0.00657286),
    "upper_arm": (0.01505885, -0.00005400, 0.00000563, 0.33388086, -0.00000181, 0.33247207),
    "forearm": (0.00399632, -0.00001365, 0.00137272, 0.07879254, -0.00000660, 0.07848510),
    "wrist_1": (0.00165491, -0.00000282, -0.00000438, 0.00135962, 0.00010157, 0.00126279),
    "wrist_2": (0.00135617, -0.00000274, 0.00000444, 0.00127827, -0.00005048, 0.00096614),
    "wrist_3": (0.00018694, 0.00000006, -0.00000017, 0.00018908, -0.00000092, 0.00025756),
}
UR5E = RobotDef(
    id="ur5e", name="Universal Robots UR5e", maker="Universal Robots", drive="servo",
    joints=("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
            "wrist_2_joint", "wrist_3_joint"),
    limits_deg=((-360, 360), (-360, 360), (-180, 180), (-360, 360), (-360, 360), (-360, 360)),
    vmax_deg_s=(180,) * 6, v100_deg_s=180.0, a100_deg_s2=300.0,
    links=(
        ("shoulder_link", "base_link", (0, 0, 0.1625), (0, 0, math.pi), (0, 0, 1), 3.761,
         (0.0, -0.00193, -0.02561), (H, 0, 0), _UR_I["shoulder"]),
        ("upper_arm_link", "shoulder_link", (0, 0, 0), (H, 0, 0), (0, 0, 1), 8.058,
         (-0.2125, 0.0, 0.11336), (0, 0, 0), _UR_I["upper_arm"]),
        ("forearm_link", "upper_arm_link", (-0.425, 0, 0), (0, 0, 0), (0, 0, 1), 2.846,
         (-0.2422, 0.0, 0.0265), (0, 0, 0), _UR_I["forearm"]),
        ("wrist_1_link", "forearm_link", (-0.3922, 0, 0.1333), (0, 0, 0), (0, 0, 1), 1.37,
         (0.0, -0.01634, -0.0018), (H, 0, 0), _UR_I["wrist_1"]),
        ("wrist_2_link", "wrist_1_link", (0, -0.0997, 0), (H, 0, 0), (0, 0, 1), 1.3,
         (0.0, 0.01634, -0.0018), (-H, 0, 0), _UR_I["wrist_2"]),
        ("wrist_3_link", "wrist_2_link", (0, 0.0996, 0), (H, math.pi, math.pi), (0, 0, 1), 0.365,
         (0.0, 0.0, -0.001159), (0, 0, 0), _UR_I["wrist_3"]),
    ),
    base=(4.0, (0, 0, 0.025), (0.0044, 0, 0, 0.0044, 0, 0.0072)),
    flange="wrist_3_link",
    control_modes=("servo",), default_control="servo",
    rated_payload_kg=5.0, repeatability_mm=0.03, reach_mm=850,
    home_deg=(0, -90, 90, -90, -90, 0),
    servo=(  # sizes 3/3/3/1/1/1: 150 Nm and 28 Nm peak (UR "max joint torques")
        ServoJoint(150.0, 101, 0.60, 30000.0, 0.01, 2.0, 1.5, 0.05),
        ServoJoint(150.0, 101, 0.60, 30000.0, 0.01, 2.0, 1.5, 0.05),
        ServoJoint(150.0, 101, 0.60, 25000.0, 0.01, 1.5, 1.0, 0.05),
        ServoJoint(28.0, 101, 0.08, 4500.0, 0.015, 0.4, 0.2, 0.02),
        ServoJoint(28.0, 101, 0.08, 4500.0, 0.015, 0.4, 0.2, 0.02),
        ServoJoint(28.0, 101, 0.08, 4500.0, 0.015, 0.4, 0.2, 0.02),
    ),
    controller={"rate_hz": 500.0, "bandwidth_hz": 12.0, "zeta": 0.9,
                "following_stop_rad": 0.05, "saturation_stop_ms": 40,
                "note": "500 Hz real-time loop (UR). Bandwidth and protective-stop thresholds "
                        "ASSUMED; UR does not publish them."},
    sources=("github.com/UniversalRobots/Universal_Robots_ROS2_Description@81f7c9f (BSD-3): "
             "config/ur5e/*.yaml; UR 'max joint torques' article for effort limits",),
    stress_space="configs/servo_arm/theta_space.yaml")

# xArm 6: xarm_ros2 @ac736c9 (BSD-3): xarm6_default_kinematics.yaml,
# link_inertial/xarm6_type6_HT_BR2.yaml, joint_limits.yaml (torque constant x 100:1 x 2.55).
XARM6 = RobotDef(
    id="xarm6", name="UFACTORY xArm 6", maker="UFACTORY", drive="servo",
    joints=tuple(f"joint{i}" for i in range(1, 7)),
    limits_deg=((-178.2, 178.2), (-118.0, 120.0), (-178.2, 11.0), (-178.2, 178.2),
                (-97.0, 178.2), (-178.2, 178.2)),
    vmax_deg_s=(180,) * 6, v100_deg_s=180.0, a100_deg_s2=573.0,
    links=(
        ("link1", "base_link", (0, 0, 0.267), (0, 0, 0), (0, 0, 1), 2.3814,
         (0.00022, 0.02951, -0.0124), (0, 0, 0),
         (0.0058562, -1.79e-05, 3.55e-06, 0.0050316, -0.000888336, 0.003536652)),
        ("link2", "link1", (0, 0, 0), (-1.5708, 0, 0), (0, 0, 1), 2.2675,
         (0.03881, -0.22783, 0.03496), (0, 0, 0),
         (0.028315776, 0.005, 0.00066546, 0.0058, -0.0045741, 0.0273447)),
        ("link3", "link2", (0.0535, -0.2845, 0), (0, 0, 0), (0, 0, 1), 1.875,
         (0.07041, 0.11631, 0.0107), (0, 0, 0),
         (0.0063483, -0.0015397, 0.00096858, 0.00379758, 0.00186567, 0.00595768)),
        ("link4", "link3", (0.0775, 0.3425, 0), (-1.5708, 0, 0), (0, 0, 1), 1.3192,
         (-0.00018, 0.01798, -0.02291), (0, 0, 0),
         (0.004896, -6.925e-06, -1.418e-05, 0.00445694, -0.00023186, 0.00134332)),
        ("link5", "link4", (0, 0, 0), (1.5708, 0, 0), (0, 0, 1), 1.33854,
         (0.0651, 0.03096, 0.00315), (0, 0, 0),
         (0.00146378, -0.000450624, 0.000284306, 0.00184192, 0.000130866, 0.002333524)),
        ("link6", "link5", (0.076, 0.097, 0), (-1.5708, 0, 0), (0, 0, 1), 0.17,
         (0, -0.00677, -0.01098), (0, 0, 0),
         (9.3e-05, 0, 0, 5.87e-05, -3.6e-06, 0.000132)),
    ),
    base=(2.7, (0.0, 0.0, 0.09103), (0.00494, 0, 0, 0.00494, 0, 0.0036)),
    flange="link6",
    control_modes=("servo",), default_control="servo",
    rated_payload_kg=5.0, repeatability_mm=0.1, reach_mm=700,
    home_deg=(0, -30, -60, 0, 90, 0),
    servo=(
        ServoJoint(163.2, 100, 0.40, 20000.0, 0.02, 1.5, 1.0, 0.04),
        ServoJoint(163.2, 100, 0.40, 20000.0, 0.02, 1.5, 1.0, 0.04),
        ServoJoint(81.6, 100, 0.20, 12000.0, 0.02, 1.0, 0.6, 0.03),
        ServoJoint(81.6, 100, 0.20, 6000.0, 0.025, 0.6, 0.3, 0.02),
        ServoJoint(30.6, 100, 0.06, 3000.0, 0.025, 0.3, 0.15, 0.02),
        ServoJoint(30.6, 100, 0.06, 3000.0, 0.025, 0.3, 0.15, 0.02),
    ),
    controller={"rate_hz": 250.0, "bandwidth_hz": 10.0, "zeta": 0.9,
                "following_stop_rad": 0.06, "saturation_stop_ms": 40,
                "note": "Servo bandwidth and stop thresholds ASSUMED; peak torques are "
                        "UFACTORY's (torque constant x 100:1 x 2.55 overload)."},
    sources=("github.com/xArm-Developer/xarm_ros2@ac736c9 (BSD-3): xarm_description/config/"
             "kinematics, link_inertial/xarm6_type6_HT_BR2.yaml, xarm6/joint_limits.yaml",),
    stress_space="configs/servo_arm/theta_space.yaml")

ROBOTS = {r.id: r for r in (_ar4("mk5"), _ar4("mk3"), UR5E, XARM6)}


def get_robot(rid: str) -> RobotDef:
    rid = str(rid)
    if rid in ROBOTS:
        return ROBOTS[rid]
    for r in ROBOTS.values():
        if rid in r.aliases:
            return r
    raise ValueError(f"unknown robot {rid!r} (known: {', '.join(ROBOTS)})")


def build_model(robot: RobotDef, *, payload_kg: float = 0.0, payload_cog=(0.0, 0.0, 0.0),
                timestep: float | None = None):
    """Compile ``robot`` with a payload on its flange and per-joint friction/armature. For
    the AR4 this is ``ar4.build_model`` (kept bit-identical)."""
    import mujoco

    if robot.drive == "stepper":
        spec = ar4.AR4Spec(model=robot.id.split("_")[-1])
        return ar4.build_model(spec, payload_kg=payload_kg, payload_cog=payload_cog,
                               timestep=timestep or robot.timestep)
    s = mujoco.MjSpec.from_string(robot.urdf_text())
    s.option.timestep = timestep or robot.timestep
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    s.compiler.balanceinertia = True
    fl = s.body(robot.flange)
    tool = fl.add_body(name="payload", pos=[float(c) for c in payload_cog])
    tool.mass = max(float(payload_kg), 1e-6)
    tool.inertia = [max(0.4 * tool.mass * 0.04 ** 2, 1e-9)] * 3
    tool.explicitinertial = True
    fl.add_site(name="tcp", pos=[0, 0, 0])
    for i, j in enumerate(robot.joints):
        jt = s.joint(j)
        sj = robot.servo[i]
        jt.frictionloss = sj.friction_nm
        jt.damping = [sj.damping_nms, 0.0, 0.0] if np.ndim(jt.damping) else sj.damping_nms
        jt.armature = sj.armature
    return s.compile()
