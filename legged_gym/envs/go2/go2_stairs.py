# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from My_unitree_go2_gym GO2_Stairs for Go2 stair climbing.

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.math import quat_apply_yaw


class Go2Stairs(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.base_height_points = self._init_base_height_points()
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()

    def _get_noise_scale_vec(self, cfg):
        """Noise scales for single-frame actor obs (45-D, no gait phase)."""
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.quat
        noise_vec[9:21] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.  # previous actions
        return noise_vec

    def compute_observations(self):
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

        # 3 + 6 + 24 + 12 = 45 (no gait phase vs flat trot)
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.obs_imu,
            self.obs_motor,
            self.actions,
        ), dim=-1)

        if self.add_noise:
            obs_now = obs_buf.clone() + (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        # Privileged: actor obs + base lin vel + height scan (187)
        heights = self.measured_heights
        if not torch.is_tensor(heights):
            heights = torch.zeros(
                self.num_envs, self.num_height_points, device=self.device, requires_grad=False)

        self.privileged_obs_buf = torch.cat((
            obs_buf,
            self.base_lin_vel * self.obs_scales.lin_vel,
            heights,
        ), dim=-1)

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat(
            [self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def _init_base_height_points(self):
        """Points under the base used for terrain-relative base-height reward."""
        y = torch.tensor(
            [-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2],
            device=self.device, requires_grad=False)
        x = torch.tensor(
            [-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15],
            device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs, self.num_base_height_points, 3,
            device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_base_heights(self, env_ids=None):
        """Mean base height above local terrain."""
        if self.cfg.terrain.mesh_type == 'plane':
            if env_ids is not None:
                return self.root_states[env_ids, 2].clone()
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids is not None:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_base_height_points),
                self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
            root_z = self.root_states[env_ids, 2]
            n = len(env_ids)
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_base_height_points),
                self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)
            root_z = self.root_states[:, 2]
            n = self.num_envs

        points = points + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[:, :, 0].view(-1), 0, self.height_samples.shape[0] - 2)
        py = torch.clip(points[:, :, 1].view(-1), 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(torch.min(heights1, heights2), heights3)
        base_height = heights.view(n, -1) * self.terrain.cfg.vertical_scale
        return torch.mean(root_z.unsqueeze(1) - base_height, dim=1)

    # ================================================ Rewards ================================================== #
    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        base_height = self._get_base_heights()
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
        return torch.sum(
            1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1)

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_default_pos(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_air_time = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        rew_air_time *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_air_time
