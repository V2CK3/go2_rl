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
import re
import cv2
import numpy as np
from isaacgym import gymapi
from legged_gym import LEGGED_GYM_ROOT_DIR

# import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 64)
    # Play: do not time-out healthy robots — only fall/tip/void should reset.
    env_cfg.env.episode_length_s = 1e6
    # Terrain map: keep curriculum layout (difficulty = row/num_rows).
    # randomized_terrain() (curriculum=False) samples H∈{14,18.5,21}cm — far harder
    # than a typical stairs run (~L3) and causes "some envs stuck, some fine".
    is_stairs = "stairs" in str(args.task).lower()
    if is_stairs:
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.num_rows = 10  # match training: H = 0.05 + 0.18*(level/10)
        env_cfg.terrain.num_cols = 5
        # Match this run's skill (~terrain_level≈3.5): L0–L1 OK, L2+ often 卡脚.
        # Evaluate mostly what was trained; raise after retrain clears taller risers.
        env_cfg.terrain.max_init_terrain_level = 2  # L0..L2 → H≈5–8.6 cm
    else:
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.max_init_terrain_level = 2
    # Stairs center platform is ~3×3 m; spawn randomly on it (leave margin off the risers).
    if hasattr(env_cfg.init_state, "xy_spawn_noise"):
        env_cfg.init_state.xy_spawn_noise = 1.2
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
    env_cfg.domain_rand.add_cmd_action_latency = False
    # Match training: stairs cfg trains WITHOUT obs latency; enabling it here
    # made play robots freeze / tip (policy never saw delayed obs).
    env_cfg.domain_rand.add_obs_latency = False
    env_cfg.domain_rand.randomize_obs_motor_latency = False
    env_cfg.domain_rand.randomize_obs_imu_latency = False
    if hasattr(env_cfg.noise, "curriculum"):
        env_cfg.noise.curriculum = False
    env_cfg.commands.heading_command = False
    # Slower climb in play: 0.5–0.7 + 碎步易撞更高立面；降速减卡脚观感噪声.
    if is_stairs:
        env_cfg.commands.ranges.lin_vel_x = [0.25, 0.45]
        env_cfg.commands.ranges.lin_vel_y = [-0.1, 0.1]
        env_cfg.commands.ranges.ang_vel_yaw = [-0.3, 0.3]
    else:
        env_cfg.commands.ranges.lin_vel_x = [0.3, 0.7]
        env_cfg.commands.ranges.lin_vel_y = [-0.1, 0.1]
        env_cfg.commands.ranges.ang_vel_yaw = [-0.4, 0.4]
    # Less frequent command changes while watching.
    env_cfg.commands.resampling_time = 20.

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)
    print(f"play terrain.mesh_type={env_cfg.terrain.mesh_type}")
    if is_stairs:
        n_rows = env_cfg.terrain.num_rows
        max_lv = env_cfg.terrain.max_init_terrain_level
        h0 = 0.05 + 0.18 * (0 / n_rows)
        h1 = 0.05 + 0.18 * (max_lv / n_rows)
        print(
            f"play stairs curriculum map rows={n_rows} spawn_levels=0..{max_lv} "
            f"step_height≈{h0*100:.1f}–{h1*100:.1f}cm (W=31cm)"
        )

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # Freeze level promotion after the curriculum heightfield is built.
    if is_stairs:
        env.cfg.terrain.curriculum = False
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    # load policy (no new log dir during play)
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    resume_path = getattr(ppo_runner, "resume_path", None)
    if resume_path:
        ckpt_dir = os.path.dirname(resume_path)
        if os.path.basename(ckpt_dir) == "model":
            ckpt_dir = os.path.dirname(ckpt_dir)
        play_run_name = os.path.basename(ckpt_dir)
        # Prefer filename iter (model_12300.pt); saved dict['iter'] can lag behind.
        m = re.search(r"model_(\d+)\.pt$", os.path.basename(resume_path))
        play_iteration = int(m.group(1)) if m else int(getattr(ppo_runner, "current_learning_iteration", -1))
    else:
        play_run_name = str(train_cfg.runner.load_run)
        play_iteration = int(getattr(ppo_runner, "current_learning_iteration", -1))
    print(f"Play checkpoint: run={play_run_name}  iter={play_iteration}")

    play_run_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, play_run_name
    )
    # export policy as a jit module: logs/<exp>/<RUNS>/policies/{RUNS}_{iter}.pt
    if EXPORT_POLICY:
        path = os.path.join(play_run_dir, "policies")
        jit_name = f"{play_run_name}_{play_iteration}.pt"
        jit_path = export_policy_as_jit(ppo_runner.alg.actor_critic, path, filename=jit_name)
        print(f"Exported JIT: {jit_path}")

    logger = Logger(env_cfg.sim.dt)
    log_robot_index = 0
    stop_log_steps = 1000  # steps to record before plot_states

    obs = env.get_observations()
    # Force a clear forward command on all envs at start (resample may still zero some).
    play_vx = 0.35 if is_stairs else 0.5
    env.commands[:, 0] = play_vx
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    i = 0
    while True:
        actions = policy(obs.detach())
        # Keep locomotion command alive (env may zero cmds with norm<0.2 on resample).
        if i % 50 == 0:
            moving = torch.norm(env.commands[:, :2], dim=1) < 0.15
            env.commands[moving, 0] = play_vx
            env.commands[moving, 1] = 0.0
        obs, _, _, _, _ = env.step(actions.detach())

        if PLOT_STATES and i < stop_log_steps:
            ri = min(log_robot_index, env.num_envs - 1)
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
        elif PLOT_STATES and i == stop_log_steps:
            path = os.path.join(play_run_dir, 'sim2play')
            # Save only — plt.show() steals focus from Isaac viewer (mouse camera dies).
            saved = logger.plot_states(
                show=False,
                async_show=False,
                save_dir=path,
                run_name=play_run_name,
                iteration=play_iteration,
            )
            print(f"Play curves saved (viewer keeps focus): {saved}")
        i += 1


if __name__ == '__main__':
    EXPORT_POLICY = True
    PLOT_STATES = True

    args = get_args()
    args.task = "go2_stairs"
    args.load_run = "2026-08-12_19-00-12_stairs"
    # args.load_run = -1
    args.checkpoint = -1
    # args.num_envs = 

    play(args)
