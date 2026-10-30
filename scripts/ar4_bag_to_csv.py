"""Turn a rosbag2 recording of a ROS 2 arm into a Stress Lab log (CSV).

Records the joint_trajectory_controller's state topic, which carries the commanded
reference and the encoder feedback in one message, so no time alignment is needed:

    ros2 bag record /joint_trajectory_controller/controller_state     # then run the program
    python scripts/ar4_bag_to_csv.py <bag_dir> --out log.csv --payload-kg 0.5
    breakpoint arm calibrate log.csv --out world_model.json

Works for any robot driven by ros2_control's JointTrajectoryController (AR4, UR, xArm).
Optional: --tcp CSV with columns t,x,y,z in mm from an external measurement (dial
indicator, camera, laser tracker); rows are matched to the nearest log sample.
Needs ``pip install rosbags`` (pure Python, no ROS install).
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np


# Leading fields of control_msgs/JointTrajectoryControllerState. Bags from Iron and later
# embed their own definitions; Humble bags do not, so these are registered as a fallback.
JTC_STATE = {
    "humble": "std_msgs/Header header\nstring[] joint_names\n"
              "trajectory_msgs/JointTrajectoryPoint reference\n"
              "trajectory_msgs/JointTrajectoryPoint feedback\n",
    "legacy": "std_msgs/Header header\nstring[] joint_names\n"
              "trajectory_msgs/JointTrajectoryPoint desired\n"
              "trajectory_msgs/JointTrajectoryPoint actual\n",
}


def typestore(layout="humble"):
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    ts.register(get_types_from_msg(JTC_STATE[layout],
                                   "control_msgs/msg/JointTrajectoryControllerState"))
    return ts


def read_bag(bag: Path, topic: str, joints=None, layout="humble"):
    from rosbags.highlevel import AnyReader

    t, ref, fb, names = [], [], [], None
    with AnyReader([bag], default_typestore=typestore(layout)) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            have = ", ".join(sorted({c.topic for c in reader.connections}))
            raise SystemExit(f"topic {topic} not in the bag (it has: {have})")
        for conn, stamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            names = names or list(msg.joint_names)
            # Humble+: reference/feedback; older JTC: desired/actual
            r = msg.reference if hasattr(msg, "reference") else msg.desired
            f = msg.feedback if hasattr(msg, "feedback") else msg.actual
            t.append(stamp * 1e-9)
            ref.append(list(r.positions))
            fb.append(list(f.positions))
    order = [names.index(j) for j in joints] if joints else list(range(6))
    ref = np.asarray(ref)[:, order]
    fb = np.asarray(fb)[:, order]
    t = np.asarray(t) - t[0]
    return t, ref, fb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--topic", default="/joint_trajectory_controller/controller_state")
    ap.add_argument("--joints", nargs=6, default=None, help="joint names in robot order")
    ap.add_argument("--tcp", type=Path, default=None, help="CSV t,x,y,z [s, mm]")
    ap.add_argument("--payload-kg", type=float, default=None)
    ap.add_argument("--model", default=None, help="robot id: mk5, ur5e, xarm6, ...")
    ap.add_argument("--control", default=None, help="ros2 | servo | ... (as in the program)")
    ap.add_argument("--layout", choices=("humble", "legacy"), default="humble",
                    help="controller-state layout for bags without embedded definitions: "
                         "humble (reference/feedback) or legacy (desired/actual, pre-Humble)")
    ap.add_argument("--out", type=Path, default=Path("log.csv"))
    a = ap.parse_args()
    t, ref, fb = read_bag(a.bag, a.topic, a.joints, a.layout)
    tcp = np.full((len(t), 3), math.nan)
    if a.tcp:
        rows = np.loadtxt(a.tcp, delimiter=",", skiprows=1, ndmin=2)
        for tt, x, y, z in rows:
            tcp[int(np.argmin(np.abs(t - tt)))] = (x, y, z)
    with open(a.out, "w", newline="") as fh:
        for k, v in (("source", f"rosbag {a.bag.name}"), ("payload_kg", a.payload_kg),
                     ("model", a.model), ("control", a.control)):
            if v is not None:
                fh.write(f"#{k}={v}\n")
        w = csv.writer(fh)
        w.writerow(["t"] + [f"cmd_j{i}" for i in range(1, 7)] + [f"fb_j{i}" for i in range(1, 7)]
                   + ["tcp_x", "tcp_y", "tcp_z"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.4f}"] + [f"{x:.5f}" for x in np.rad2deg(ref[i])]
                       + [f"{x:.5f}" for x in np.rad2deg(fb[i])]
                       + ["" if math.isnan(x) else f"{x:.4f}" for x in tcp[i]])
    print(f"wrote {a.out}: {len(t)} samples over {t[-1]:.1f} s")


if __name__ == "__main__":
    main()
