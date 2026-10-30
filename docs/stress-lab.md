# Stress Lab: stress-testing programmed robot arms

BreakPoint's policy engine searches for the conditions where a learned controller fails.
Stress Lab does the same for arms that run conventional programs (taught points, joint and
linear moves, inverse kinematics, joint-level motion control): it searches payload,
centre of gravity, motion profile, drive-train and fixture conditions for the ones where a
program stops doing its job, isolates the smallest change that causes each failure, and
calibrates the simulator against the robot's own logs.

CPU only. No jax, no GPU.

```bash
python -m pip install -e ".[arm]"          # mujoco, scipy, flask
breakpoint arm serve                       # the web app, http://127.0.0.1:8740
```

## Robots

| Robot | Drive train modelled | Controller modelled | Sources |
|---|---|---|---|
| Annin AR4 MK5 / MK3 | stepper + belt/planetary, motor-shaft encoders | AR4 ROS 2 driver: velocity mode (as shipped), velocity PID, position mode; ideal step generator | `ar4_ros_driver` (MIT): URDF macro, `mk*.yaml`, Teensy firmware, `controllers.yaml` |
| Universal Robots UR5e | brushless servo + harmonic drive | 1 kHz drive loop, 500 Hz setpoints, gravity feed-forward, torque limit, protective stop | `Universal_Robots_ROS2_Description` (BSD-3): kinematics, inertials, effort and speed limits |
| UFACTORY xArm 6 | brushless servo + harmonic drive | 1 kHz drive loop, 250 Hz setpoints, same structure | `xarm_ros2` (BSD-3): kinematics, inertials, torque constants x 100:1 |

Kinematics are checked against each maker's published zero pose (`tests/test_arm.py`).
Values the makers do not publish (stepper torque curves, harmonic-drive stiffness,
reflected rotor inertia, servo bandwidth, protective-stop thresholds) are typical values,
marked ASSUMED in `breakpoint_mjx/arm/robots.py` and `ar4.py`. They are what calibration
replaces.

### How the drive trains are simulated

* **Stepper (AR4).** The rotor feels `T(w) sin(50 (theta_cmd - theta_rotor))`. Pull-out
  torque falls with speed. When the load needs more than that, the rotor slips a tooth
  and the axis loses 4 full steps: missed steps come from the physics, not a threshold.
  Between rotor and link there is backlash and torsional compliance. The encoders sit on
  the motor shafts, as on the AR4, so they see lost steps but not backlash or wind-up.
  The shipped driver (`velocity_control_enabled: true`, position and velocity command
  interfaces) streams the trajectory's velocity with no position correction, so a lost
  step is never made up. Position mode (`MT`) re-targets from the encoders and does make
  it up.
* **Servo (UR5e, xArm 6).** PID on the motor-side encoder at 1 kHz, setpoints held at the
  robot's command rate, model-based gravity feed-forward (`ff_scale` is the payload as
  configured in the controller versus the real one), a peak-torque limit with
  anti-windup, harmonic-drive compliance, and a protective stop on following error or
  40 ms of torque saturation.

## What it tests

| Tab / command | Question it answers |
|---|---|
| Stress test (`breakpoint arm stress`) | Which conditions make this program fail, grouped into families named by cause (missed steps, protective stop, torque, singularity, envelope, sag the encoders cannot see, drift)? |
| Isolate cause (`--isolate N`) | Which axes does a failure actually need, and from what value? Greedy pruning to nominal, then bisection, keeping the same failure family. |
| Envelope (`breakpoint arm envelope`) | Pass/fail over two axes, e.g. payload × acceleration, with the plausible band marked. |
| Repeatability | Error at every taught point over N back-to-back cycles: accumulated error. |
| Sim vs real (`breakpoint arm calibrate`) | Fit the simulator to a robot log, then test on the calibrated simulator. |

Errors are measured against the *taught* position: the nominal run, because points on a
real robot are taught by jogging, so the robot's static sag is already in them.

## Calibration (the world model)

Two layers, fitted to a log of commanded vs encoder positions (plus optional external TCP
measurements):

1. **Physics.** Friction, motor torque, backlash and gearbox stiffness scales, fitted by
   cross-entropy search where every candidate is a full replay of the logged command.
2. **Residual.** What physics does not explain (latency, velocity-dependent lag),
   regressed on the commanded velocity and applied as a correction on the commanded motion
   with the `world_model_scale` axis. This is the contract of the engine's
   `env._bp_action_correction` hook.

The report says per parameter whether the log identified it. Motor-shaft encoders cannot
see backlash or wind-up, so those need TCP measurements at a few taught points (a dial
indicator is enough).

**Measured on synthetic data only so far.** Fitted on one synthetic log of the AR4 (hidden
drive-train values plus an unmodelled 17 ms latency), then tested on a different program
on the same hidden robot (heavier payload, other poses and speeds):

| Held-out log | Joint feedback RMS | TCP RMS |
|---|---|---|
| raw simulator | 0.165° | 0.53 mm |
| physics calibration | 0.163° | 0.15 mm |
| physics + residual | 0.006° | 0.17 mm |

It reproduces the robot, but it does not recover the true parameters uniquely (friction
2.5 vs 1.8 true; the report marks it weak). The synthetic robot shares the simulator's
structure, so a real arm's unmodelled effects are the open question: the first real log is
the test that matters.

### Getting a log from a robot

```bash
ros2 bag record /joint_trajectory_controller/controller_state   # run the program meanwhile
python scripts/ar4_bag_to_csv.py <bag_dir> --out log.csv --payload-kg 0.5 --model mk5
breakpoint arm calibrate log.csv --out world_model.json
```

Any ros2_control JointTrajectoryController works (AR4, UR, xArm). Add `--tcp tcp.csv`
(columns `t,x,y,z`, mm) for external measurements.

## Demos for prospects

```bash
breakpoint arm demo-pack --out demo_pack \
    --policy-report g1=out/g1/report.html --policy-report spot=out/spot/report.html
```

This writes one folder: `index.html` (a hub) plus one self-contained interactive page per
arm (stress test with replays, minimal cause, envelope, repeatability, calibration) and
the legged-policy reports from the engine. It takes about 6 minutes on a laptop. Zip it,
attach it or host it as static files. G1 and Spot need a trained policy and a GPU run
(`scripts/train.py`, the Colab notebook). The Go1 card uses the bundled CPU smoke-budget
report.

Programs are YAML (`breakpoint_mjx/arm/programs/`). Write one for a prospect's own cell
and upload it in the app, or pass the path to any `breakpoint arm` command.

## Limits

* Simulation only. A clean result is a search result under a budget, not a proof.
* Contacts are off: no collision checking against the cell or self-collision, only a
  table plane check on the flange.
* One payload body, a 4 cm sphere at the given CoG. No grasp dynamics.
* Plausibility is the engine's transparent band heuristic, not a measured probability.
* `ros2_pid` (the AR4's velocity-only interface with the yaml gains) oscillates in
  simulation at 60 Hz. That is a modelling result to check on hardware, not a claim.
