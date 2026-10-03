# HAND-WRITTEN FIXTURE mirroring the class layout of legged_gym's LeggedRobotCfg
# (legged_gym/envs/base/legged_robot_config.py). Values illustrative.
import os  # never executed by breakpoint init; only literals are read


class LeggedRobotCfg:
    class control:
        control_type = 'P'
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}
        damping = {'joint_a': 1.0, 'joint_b': 1.5}
        action_scale = 0.5
        decimation = 4

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        not_a_literal = os.getcwd()
