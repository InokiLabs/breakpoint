# HAND-WRITTEN FIXTURE mirroring legged_gym/envs/go2/go2_config.py's override layout.
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class GO2RoughCfg(LeggedRobotCfg):
    class control(LeggedRobotCfg.control):
        stiffness = {'joint': 20.}
        damping = {'joint': 0.5}
        action_scale = 0.25

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        randomize_motor_strength = True
