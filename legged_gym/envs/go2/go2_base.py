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
# Adapted from My_unitree_go2_gym GO2_Trot for Go2 locomotion skills.

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.terrain import Terrain


class Go2Base(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.trot = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=False)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = (self.episode_length_buf * self.dt) % cycle_time / cycle_time
        return phase

    def _get_gait_phase(self):
        # float mask: 1 stance, 0 swing; diagonal pairs for trot
        phase = self._get_phase()
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = phase < 0.5  # FL-RR
        stance_mask[:, 1] = phase > 0.5  # FR-RL
        return stance_mask

    def _get_noise_scale_vec(self, cfg):
        """Noise scales aligned with single-frame actor obs (47-D)."""
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0:5] = 0.  # phase + commands
        noise_vec[5:8] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[8:11] = noise_scales.quat
        noise_vec[11:23] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[23:35] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[35:47] = 0.  # previous actions
        return noise_vec

    def compute_observations(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)

        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 5
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
            self.dof_pos * self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,  # 12
            self.actions,  # 12
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
            stance_mask,  # 2
            contact_mask,  # 2
        ), dim=-1)  # 68

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        if self.cfg.domain_rand.randomize_obs_motor_latency:
            self.obs_motor = self.obs_motor_latency_buffer[
                torch.arange(self.num_envs), :, self.obs_motor_latency_simstep.long()]
        else:
            self.obs_motor = torch.cat((q, dq), 1)

        if self.cfg.domain_rand.randomize_obs_imu_latency:
            self.obs_imu = self.obs_imu_latency_buffer[
                torch.arange(self.num_envs), :, self.obs_imu_latency_simstep.long()]
        else:
            self.obs_imu = torch.cat(
                (self.base_ang_vel * self.obs_scales.ang_vel,
                 self.base_euler_xyz * self.obs_scales.quat), 1)

        obs_buf = torch.cat((
            self.command_input,  # 5
            self.obs_imu,  # 6
            self.obs_motor,  # 24
            self.actions,  # 12
        ), dim=-1)  # 47

        if self.add_noise:
            obs_now = obs_buf.clone() + (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def _resample_commands(self, env_ids):
        """Resample velocity commands; occasionally force stand-still."""
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), 
            device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), 
            device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), 
                device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), 
                device=self.device).squeeze(1)

        all_zero_mask = torch.rand(len(env_ids), device=self.device) < 0.05
        self.commands[env_ids[all_zero_mask]] = 0.0

        xy_zero_mask = torch.rand(len(env_ids), device=self.device) < 0.05
        self.commands[env_ids[xy_zero_mask], :2] = 0.0
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    # ================================================ Rewards ================================================== #
    def _reward_trot(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        TROT = (contact[:, 0] == contact[:, 3]) & \
               (contact[:, 1] == contact[:, 2]) & \
               (contact[:, 0] == stance_mask[:, 0]) & \
               (contact[:, 1] == stance_mask[:, 1])
        cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        self.trot = TROT.to(torch.float32).mean() * (cmd_norm > 0.1) + \
            (torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 0.1), dim=1) == 4) * (cmd_norm < 0.1)
        return TROT * (cmd_norm > 0.1)

    def _reward_default_hip_pos(self):
        joint_diff = torch.abs(self.dof_pos[:, 0]) + torch.abs(self.dof_pos[:, 3]) + \
                     torch.abs(self.dof_pos[:, 6]) + torch.abs(self.dof_pos[:, 9])
        return joint_diff

    def _reward_feet_clearance(self):
        self.feet_height = self.rigid_state[:, self.feet_indices, 2] - 0.02
        left_feet_height = self.feet_height[:, ::3]
        right_feet_height = self.feet_height[:, 1:3]
        swing_mask = 1 - self._get_gait_phase()
        phase = self._get_phase()
        target_height = (torch.abs(torch.sin(2 * torch.pi * phase)) * self.cfg.rewards.target_foot_height).unsqueeze(1).repeat(1, 2)
        rew = torch.exp(-torch.sum(torch.abs(left_feet_height - target_height) * swing_mask[:, 0].unsqueeze(1).repeat(1, 2), dim=1) * 10)
        rew += torch.exp(-torch.sum(torch.abs(right_feet_height - target_height) * swing_mask[:, 1].unsqueeze(1).repeat(1, 2), dim=1) * 10)
        return rew * (torch.norm(self.commands[:, :3], dim=1) > 0.1)

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_vel_limits(self):
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma) * (self.trot > 0.7)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma) * (self.trot > 0.7)

    def _reward_contact_without_command(self):
        return (torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 0.1), dim=1) == 4) * \
               (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_stand_still(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * \
               (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        return torch.sum(
            (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.),
            dim=1)

    def _reward_default_pos(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
