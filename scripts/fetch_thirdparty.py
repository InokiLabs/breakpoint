"""Fetch a real, public, third-party locomotion checkpoint and stage it for `breakpoint`.

The checkpoint is NOT vendored in this repo. This script downloads it, checks its sha256,
exports the actor MLP to an npz of weights (``params:`` policy spec), and writes the
training config next to it in Isaac Lab's ``params/env.yaml`` / ``agent.yaml`` dump
layout so `breakpoint init --from rsl_rl` can read it.

Checkpoint: Isaac Lab's pretrained RSL-RL policy for ``Isaac-Velocity-Flat-Unitree-Go2-v0``
(the Unitree Go2 flat-terrain velocity task), as served to Isaac Lab's
``scripts/reinforcement_learning/rsl_rl/play.py --use_pretrained_checkpoint``:

    https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/
        Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/
        checkpoint.pt

Licence: Isaac Lab's source is BSD-3-Clause (github.com/isaac-sim/IsaacLab/LICENSE). No
licence file is published next to the checkpoint on NVIDIA's asset bucket, so its terms are
not verified here; that is why it is downloaded at run time instead of committed.

Why not unitree_rl_gym's ``deploy/pre_train/go2/motion.pt``: that file does not exist.
Checked 2026-09-27 against every commit on every branch of
github.com/unitreerobotics/unitree_rl_gym (59 commits): the shipped pre-trained policies
are g1, h1 and h1_2 (humanoids), and there is no go2 deploy config. The Isaac Lab Go2
checkpoint is the nearest real public quadruped policy with a readable training config.

The env.yaml written here is TRANSCRIBED by hand from the Isaac Lab source at commit
b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8 (main, 2026-09-27):
  source/isaaclab_assets/isaaclab_assets/robots/unitree.py        UNITREE_GO2_CFG
  source/isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py   obs / actions / events
  source/isaaclab_tasks/.../locomotion/velocity/config/go2/rough_env_cfg.py, flat_env_cfg.py
  source/isaaclab_tasks/.../locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py
The checkpoint was trained with whatever Isaac Lab version produced it (its ``iter`` is 299,
matching the flat task's max_iterations 300); if the config changed between that version
and b0542fe, the transcription can differ from what the policy really saw.

    python scripts/fetch_thirdparty.py                     # -> thirdparty/isaaclab_go2_flat/
    python scripts/fetch_thirdparty.py --out some/dir

Needs torch (installed in .venv-cpu) to read the checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np

URL = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/"
       "Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/"
       "checkpoint.pt")
SHA256 = "47ccb2046ec3064257997f879c62d64217b937713c30d827883b2c2f40c50439"
ISAACLAB_COMMIT = "b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8"

# params/env.yaml in Isaac Lab's dump layout, transcribed from the files named above.
# Only the fields breakpoint init reads are included.
ENV_YAML = f"""\
# TRANSCRIBED (not dumped by Isaac Lab) from IsaacLab@{ISAACLAB_COMMIT[:7]} for
# Isaac-Velocity-Flat-Unitree-Go2-v0. Written by scripts/fetch_thirdparty.py.
sim:
  dt: 0.005                      # velocity_env_cfg.py __post_init__
decimation: 4
scene:
  robot:
    init_state:                  # unitree.py UNITREE_GO2_CFG
      joint_pos:
        .*L_hip_joint: 0.1
        .*R_hip_joint: -0.1
        F[L,R]_thigh_joint: 0.8
        R[L,R]_thigh_joint: 1.0
        .*_calf_joint: -1.5
    actuators:
      base_legs:
        class_type: isaaclab.actuators.actuator_pd:DCMotor
        joint_names_expr: [.*_hip_joint, .*_thigh_joint, .*_calf_joint]
        effort_limit: 23.5
        saturation_effort: 23.5
        velocity_limit: 30.0
        stiffness: 25.0
        damping: 0.5
        friction: 0.0
observations:
  policy:                        # velocity_env_cfg.py PolicyCfg (height_scan removed by flat)
    concatenate_terms: true
    enable_corruption: true
    history_length: null
    base_lin_vel: {{func: isaaclab.envs.mdp.observations:base_lin_vel, scale: null, clip: null}}
    base_ang_vel: {{func: isaaclab.envs.mdp.observations:base_ang_vel, scale: null, clip: null}}
    projected_gravity: {{func: isaaclab.envs.mdp.observations:projected_gravity, scale: null, clip: null}}
    velocity_commands:
      func: isaaclab.envs.mdp.observations:generated_commands
      params: {{command_name: base_velocity}}
    joint_pos: {{func: isaaclab.envs.mdp.observations:joint_pos_rel, scale: null, clip: null}}
    joint_vel: {{func: isaaclab.envs.mdp.observations:joint_vel_rel, scale: null, clip: null}}
    actions: {{func: isaaclab.envs.mdp.observations:last_action, scale: null, clip: null}}
    height_scan: null
actions:
  joint_pos:                     # velocity_env_cfg.py ActionsCfg; go2 rough sets scale 0.25
    class_type: isaaclab.envs.mdp.actions.joint_actions:JointPositionAction
    joint_names: [.*]
    scale: 0.25
    offset: 0.0
    preserve_order: false
    use_default_offset: true
events:                          # velocity_env_cfg.py EventCfg with the go2 rough overrides
  physics_material:
    func: isaaclab.envs.mdp.events:randomize_rigid_body_material
    mode: startup
    params:
      asset_cfg: {{name: robot, body_names: .*}}
      static_friction_range: [0.8, 0.8]
      dynamic_friction_range: [0.6, 0.6]
      restitution_range: [0.0, 0.0]
  add_base_mass:
    func: isaaclab.envs.mdp.events:randomize_rigid_body_mass
    mode: startup
    params:
      asset_cfg: {{name: robot, body_names: base}}
      mass_distribution_params: [-1.0, 3.0]
      operation: add
  base_external_force_torque:
    func: isaaclab.envs.mdp.events:apply_external_force_torque
    mode: reset
    params: {{force_range: [0.0, 0.0], torque_range: [0.0, 0.0]}}
  reset_base:
    func: isaaclab.envs.mdp.events:reset_root_state_uniform
    mode: reset
  reset_robot_joints:
    func: isaaclab.envs.mdp.events:reset_joints_by_scale
    mode: reset
    params: {{position_range: [1.0, 1.0], velocity_range: [0.0, 0.0]}}
"""

AGENT_YAML = f"""\
# TRANSCRIBED from IsaacLab@{ISAACLAB_COMMIT[:7]} go2/agents/rsl_rl_ppo_cfg.py
# (UnitreeGo2FlatPPORunnerCfg). Written by scripts/fetch_thirdparty.py.
policy:
  actor_hidden_dims: [128, 128, 128]
  activation: elu
  actor_obs_normalization: false
empirical_normalization: false
"""


def _download(url, dst):
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 - fixed https URL
        data = r.read()
    dst.write_bytes(data)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="thirdparty/isaaclab_go2_flat",
                    help="staging directory (default thirdparty/isaaclab_go2_flat)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "params").mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint.pt"
    data = ckpt.read_bytes() if ckpt.exists() else _download(URL, ckpt)
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256:
        print(f"sha256 mismatch: got {got}, expected {SHA256}. The upstream file changed; "
              f"re-check it before trusting any result.", file=sys.stderr)
        return 2
    print(f"sha256 ok ({SHA256[:12]}...)")

    import torch

    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"]
    # the actor MLP only: actor.{0,2,4,6}.{weight,bias}; torch Linear weights are
    # (out, in), which the params: path reads with its default weight_layout out_in.
    actor = {k: v.numpy().astype(np.float32) for k, v in sd.items() if k.startswith("actor.")}
    shapes = [tuple(actor[k].shape) for k in actor if k.endswith("weight")]
    np.savez(out / "actor.npz", **actor)
    (out / "params" / "env.yaml").write_text(ENV_YAML, encoding="utf-8")
    (out / "params" / "agent.yaml").write_text(AGENT_YAML, encoding="utf-8")
    print(f"wrote {out / 'actor.npz'} (layers {shapes}, activation elu), "
          f"{out / 'params' / 'env.yaml'}, {out / 'params' / 'agent.yaml'}")
    print("next:\n"
          f"  breakpoint init --from rsl_rl {out} --robot go1 --out {out / 'contract.yaml'}\n"
          f"  BP_ACTIVATION=elu breakpoint check-policy --robot go1 --policy "
          f"params:{out / 'actor.npz'} --obs-spec {out / 'contract.yaml'} --gap-check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
