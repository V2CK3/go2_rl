# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import os
import cv2
import numpy as np
from isaacgym import gymapi
from legged_gym import LEGGED_GYM_ROOT_DIR

# import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from datetime import datetime



def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False     
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False 
    env_cfg.domain_rand.push_robots = False 
    env_cfg.domain_rand.continuous_push = False 
    env_cfg.domain_rand.randomize_base_mass = False 
    env_cfg.domain_rand.randomize_base_com = False 
    env_cfg.domain_rand.randomize_pd_gains = False 
    env_cfg.domain_rand.randomize_calculated_torque = False 
    env_cfg.domain_rand.randomize_link_mass = False 
    env_cfg.domain_rand.randomize_motor_zero_offset = False 
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_joint_armature = False
    env_cfg.domain_rand.randomize_cmd_action_latency = False
    env_cfg.domain_rand.range_cmd_action_latency = [5, 5]
    env_cfg.domain_rand.add_obs_latency = True
    env_cfg.domain_rand.randomize_obs_motor_latency = True
    env_cfg.domain_rand.range_obs_motor_latency = [5, 5]
    env_cfg.domain_rand.randomize_obs_imu_latency = True
    env_cfg.domain_rand.range_obs_imu_latency = [5, 5]
    env_cfg.noise.curriculum = False
    env_cfg.commands.heading_command = False

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)
    print(f"play terrain.mesh_type={env_cfg.terrain.mesh_type}")

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    # load policy (no new log dir during play)
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    current_time_str = datetime.now().strftime('%H-%M-%S')
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, '0_exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env_cfg.sim.dt)
    robot_index = 0
    stop_state_log = 1000  # steps to record before plot_states
    plots_done = False
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION,
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S') + args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    obs = env.get_observations()
    i = 0
    while True:
        actions = policy(obs.detach())
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])

        if PLOT_STATES and i < stop_state_log:
            # One GPU sync batch for the logged robot (avoids dozens of .item() stalls).
            ri = robot_index
            act = actions[ri].detach().cpu().numpy()
            dof_pos = env.dof_pos[ri].detach().cpu().numpy()
            dof_vel = env.dof_vel[ri].detach().cpu().numpy()
            torques = env.torques[ri].detach().cpu().numpy()
            commands = env.commands[ri, :3].detach().cpu().numpy()
            base_lin = env.base_lin_vel[ri].detach().cpu().numpy()
            base_ang_yaw = env.base_ang_vel[ri, 2].detach().cpu().item()
            contact_z = env.contact_forces[ri, env.feet_indices, 2].detach().cpu().numpy()
            joint_idx = 0
            logger.log_states(
                {
                    'dof_pos_target': float(act[joint_idx] * env.cfg.control.action_scale),
                    'dof_pos': float(dof_pos[joint_idx]),
                    'dof_vel': float(dof_vel[joint_idx]),
                    'dof_torque': float(torques[joint_idx]),
                    'command_x': float(commands[0]),
                    'command_y': float(commands[1]),
                    'command_yaw': float(commands[2]),
                    'base_vel_x': float(base_lin[0]),
                    'base_vel_y': float(base_lin[1]),
                    'base_vel_z': float(base_lin[2]),
                    'base_vel_yaw': float(base_ang_yaw),
                    'contact_forces_z': contact_z,
                }
            )
        elif PLOT_STATES and (not plots_done) and i == stop_state_log:
            # Shared Logger.plot_states (async window so Isaac Gym keeps running).
            logger.plot_states(show=True, async_show=True)
            plots_done = True

        i += 1

    if RENDER:
        video.release()


if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = False
    PLOT_STATES = False  # True: record then Logger.plot_states(); avoid during interactive play if possible
    args = get_args()
    play(args)
