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
        # Swing-foot lift trackers (stair-aware clearance; sparse credit on landing).
        nfeet = len(self.feet_indices)
        self.last_feet_z = torch.zeros(self.num_envs, nfeet, device=self.device)
        self.feet_air_clearance = torch.zeros(self.num_envs, nfeet, device=self.device)
        self.prev_foot_contact_clr = torch.zeros(
            self.num_envs, nfeet, dtype=torch.bool, device=self.device
        )
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()

    def check_termination(self):
        """Reset on base contact, bad attitude, or falling through terrain."""
        super().check_termination()
        # Side/back on ground: projected gravity z is -1 upright; near 0 or + means tipped.
        fallen_orient = self.projected_gravity[:, 2] > -0.3
        # Fell off mesh / through stairs (common cause of "disappear").
        fallen_height = self.root_states[:, 2] < (self.env_origins[:, 2] - 0.8)
        # Extreme tumble velocity (sim blow-up).
        blown = torch.norm(self.root_states[:, 7:10], dim=1) > 8.0
        self.reset_buf |= fallen_orient | fallen_height | blown

    def _reset_root_states(self, env_ids):
        """Reset on the flat center platform of each terrain tile (not stacked)."""
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # Pyramid stairs platform_size≈3m; keep spawn on the flat middle.
            xy_noise = getattr(self.cfg.init_state, "xy_spawn_noise", 1.2)
            n = len(env_ids)
            # Independent x/y uniform → spread out instead of clustering at origin.
            self.root_states[env_ids, 0] += torch_rand_float(-xy_noise, xy_noise, (n, 1), device=self.device).squeeze(1)
            self.root_states[env_ids, 1] += torch_rand_float(-xy_noise, xy_noise, (n, 1), device=self.device).squeeze(1)
            self.root_states[env_ids, 2] = self.env_origins[env_ids, 2] + self.cfg.init_state.pos[2]
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids, 7:13] = 0
            self.root_states[env_ids, 2] += 1.8
        else:
            self.root_states[env_ids, 7:13] = 0.
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _update_terrain_curriculum(self, env_ids):
        """Terrain curriculum gated by lin-vel tracking (avoid level-up after collapse)."""
        if not self.init_done:
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # Default 0.3 * length (~2.4m): slower cmds made 0.4*length hard to hit.
        promote_frac = getattr(self.cfg.rewards, "terrain_promote_distance_frac", 0.3)
        move_up = distance > self.terrain.env_length * promote_frac
        # Soft demotion: only if barely left the spawn (harsh 0.5*cmd*T pinned everyone at L0).
        demote_frac = getattr(self.cfg.rewards, "terrain_demote_distance_frac", 0.12)
        cmd_xy = torch.norm(self.commands[env_ids, :2], dim=1)
        move_down = (distance < cmd_xy * self.max_episode_length_s * demote_frac) * ~move_up

        track_scale = self.reward_scales.get("tracking_lin_vel", 0.0)
        if track_scale > 0 and "tracking_lin_vel" in self.episode_sums:
            thr = getattr(self.cfg.rewards, "terrain_track_up_threshold", 0.2)
            track_mean = self.episode_sums["tracking_lin_vel"][env_ids] / self.max_episode_length
            good_track = track_mean > thr * track_scale
            # Require yaw tracking too — distance alone rewards walking off the stair tile.
            ang_scale = self.reward_scales.get("tracking_ang_vel", 0.0)
            if ang_scale > 0 and "tracking_ang_vel" in self.episode_sums:
                ang_thr = getattr(self.cfg.rewards, "terrain_ang_track_up_threshold", 0.12)
                ang_mean = self.episode_sums["tracking_ang_vel"][env_ids] / self.max_episode_length
                good_track = good_track & (ang_mean > ang_thr * ang_scale)
            move_up = move_up & good_track
            if getattr(self.cfg.rewards, "terrain_demote_on_poor_track", False):
                commanded = cmd_xy > 0.1
                move_down = move_down | ((~good_track) & commanded & ~move_up)

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

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
        if hasattr(self, "feet_air_clearance"):
            self.feet_air_clearance[env_ids] = 0.0
            self.last_feet_z[env_ids] = self.rigid_state[env_ids][:, self.feet_indices, 2]
            self.prev_foot_contact_clr[env_ids] = False

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
        clip = getattr(self.cfg.rewards, "lin_vel_z_clip", 2.0)
        return torch.square(self.base_lin_vel[:, 2]).clip(max=clip)

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_pitch_forward(self):
        """Penalize nose-down pitch only (head low / tail high)."""
        # Body +x gravity component > 0 when pitched forward.
        return torch.square(self.projected_gravity[:, 0].clip(min=0.0))

    def _reward_nose_plant(self):
        """Penalize front-down / rear-up stance (rear feet unloaded while commanded)."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        # feet_names are sorted: FL, FR, RL, RR
        front = contact[:, 0] | contact[:, 1]
        rear = contact[:, 2] | contact[:, 3]
        moving = torch.norm(self.commands[:, :2], dim=1) > 0.15
        pitched = self.projected_gravity[:, 0] > 0.12
        rear_unload = front & (~rear)
        return ((pitched | rear_unload) & moving).float()

    def _reward_base_height(self):
        base_height = self._get_base_heights()
        clip = getattr(self.cfg.rewards, "base_height_clip", 0.25)
        return torch.square(base_height - self.cfg.rewards.base_height_target).clip(max=clip)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_calf_acc(self):
        """Extra penalty on calf joint acceleration (play 小腿高频抖动)."""
        # DOF order: FL/FR/RL/RR × (hip, thigh, calf) → calves at 2,5,8,11.
        acc = (self.last_dof_vel - self.dof_vel) / self.dt
        return torch.sum(torch.square(acc[:, [2, 5, 8, 11]]), dim=1)

    def _reward_action_rate(self):
        clip = getattr(self.cfg.rewards, "action_rate_clip", 8.0)
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1).clip(max=clip)

    def _reward_action_smoothness(self):
        """Penalize second-order action jerk (anti bang-bang chatter)."""
        clip = getattr(self.cfg.rewards, "action_smoothness_clip", 8.0)
        return torch.sum(
            torch.square(self.actions - 2.0 * self.last_actions + self.last_last_actions),
            dim=1,
        ).clip(max=clip)

    def _reward_lin_vel_smooth(self):
        """Penalize forward base-velocity jerk (stop-go / 一阵一阵)."""
        last_lin = quat_rotate_inverse(self.base_quat, self.last_root_vel[:, 0:3])
        dvx = (self.base_lin_vel[:, 0] - last_lin[:, 0]) / self.dt
        moving = torch.abs(self.commands[:, 0]) > 0.2
        clip = getattr(self.cfg.rewards, "lin_vel_smooth_clip", 40.0)
        return (torch.square(dvx) * moving.float()).clip(max=clip)

    def _reward_collision(self):
        return torch.sum(
            1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1)

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        rew = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        dragging = contact.all(dim=1) & (torch.norm(self.commands[:, :2], dim=1) > 0.15)
        scale = getattr(self.cfg.rewards, "drag_tracking_scale", 0.6)
        return rew * torch.where(dragging, torch.full_like(rew, scale), torch.ones_like(rew))

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        dragging = contact.all(dim=1) & (torch.norm(self.commands[:, :2], dim=1) > 0.15)
        scale = getattr(self.cfg.rewards, "drag_tracking_scale", 0.6)
        return rew * torch.where(dragging, torch.full_like(rew, scale), torch.ones_like(rew))

    def _reward_default_pos(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_default_hip_pos(self):
        # Penalize hip abduction/adduction away from 0 (indices: FL, FR, RL, RR hips).
        return (
            torch.abs(self.dof_pos[:, 0])
            + torch.abs(self.dof_pos[:, 3])
            + torch.abs(self.dof_pos[:, 6])
            + torch.abs(self.dof_pos[:, 9])
        )

    def _reward_thigh_overflex(self):
        """Penalize thighs folding past a soft limit into the base (esp. front on drop-in)."""
        # DOF order: FL/FR/RL/RR × (hip, thigh, calf) → thighs at 1,4,7,10.
        thighs = self.dof_pos[:, [1, 4, 7, 10]]
        thr = getattr(self.cfg.rewards, "thigh_overflex_threshold", 1.05)
        return torch.sum((thighs - thr).clip(min=0.0), dim=1)

    def _reward_front_rear_thigh_amp(self):
        """Penalize rear thighs swinging much farther from default than front thighs."""
        front = torch.abs(self.dof_pos[:, [1, 4]] - self.default_dof_pos[:, [1, 4]]).sum(dim=1)
        rear = torch.abs(self.dof_pos[:, [7, 10]] - self.default_dof_pos[:, [7, 10]]).sum(dim=1)
        return (rear - front).clip(min=0.0)

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical faces (common on stair edges / tip-over).
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _sample_terrain_height_xy(self, pos_xy: torch.Tensor) -> torch.Tensor:
        """Sample heightfield at world XY. pos_xy: (N, K, 2) → (N, K)."""
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(pos_xy.shape[:2], device=self.device, dtype=torch.float)
        scale = self.terrain.cfg.horizontal_scale
        border = self.terrain.cfg.border_size
        px = ((pos_xy[..., 0] + border) / scale).long()
        py = ((pos_xy[..., 1] + border) / scale).long()
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        h = torch.minimum(self.height_samples[px, py], self.height_samples[px + 1, py])
        h = torch.minimum(h, self.height_samples[px, py + 1])
        return h.float() * self.terrain.cfg.vertical_scale

    def _reward_feet_clearance(self):
        """Sparse swing-lift credit on landing (dense per-step swing reward caused rear-high farming)."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        foot_z = self.rigid_state[:, self.feet_indices, 2]
        swing = ~contact
        delta_z = foot_z - self.last_feet_z
        self.feet_air_clearance += delta_z.clip(min=0.0) * swing.float()
        self.last_feet_z = foot_z.clone()

        # One-shot credit when a foot first touches down after swinging.
        first_contact = contact & (~self.prev_foot_contact_clr) & (self.feet_air_clearance > 0.01)
        tgt = self.cfg.rewards.target_foot_height
        peak = (self.feet_air_clearance / tgt).clip(max=1.0)
        short = (tgt - self.feet_air_clearance).clip(min=0.0, max=0.10) / tgt
        rew = torch.sum((peak - 0.5 * short) * first_contact.float(), dim=1)

        self.feet_air_clearance = torch.where(
            contact, torch.zeros_like(self.feet_air_clearance), self.feet_air_clearance
        )
        self.prev_foot_contact_clr = contact.clone()
        moving = torch.norm(self.commands[:, :2], dim=1) > 0.1
        return rew * moving

    def _reward_feet_clearance_terrain(self):
        """Penalize swing feet too close to local / ahead terrain (L3+ riser clearance)."""
        if self.cfg.terrain.mesh_type not in ("trimesh", "heightfield"):
            return torch.zeros(self.num_envs, device=self.device)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        swing = (~contact).float()
        foot_pos = self.rigid_state[:, self.feet_indices, :3]
        forward = quat_apply(self.base_quat, self.forward_vec)
        # Probe at foot and slightly ahead (next riser lip).
        look = getattr(self.cfg.rewards, "clearance_look_ahead", 0.10)
        margin = getattr(self.cfg.rewards, "clearance_terrain_margin", 0.04)
        xy0 = foot_pos[..., :2]
        xy1 = xy0 + look * forward[:, None, :2]
        h = torch.maximum(
            self._sample_terrain_height_xy(xy0),
            self._sample_terrain_height_xy(xy1),
        )
        # Positive when foot is below terrain+margin while swinging → clip to penalty.
        gap = (h + margin) - foot_pos[..., 2]
        short = gap.clip(min=0.0, max=0.12) * swing
        moving = torch.norm(self.commands[:, :2], dim=1) > 0.1
        return torch.sum(short, dim=1) * moving

    def _reward_feet_contact_forces(self):
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
                - self.cfg.rewards.max_contact_force
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_foot_slip(self):
        """Penalize horizontal foot speed while in contact (anti slide / dog-paddle)."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_vel_xy = self.rigid_state[:, self.feet_indices, 7:9]
        slip = torch.norm(foot_vel_xy, dim=2) * contact.float()
        return torch.sum(slip, dim=1)

    def _resample_commands(self, env_ids):
        """Resample cmds; keep ||v_xy|| above deadzone; bias straight (match play/sim2sim)."""
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return
        dead = getattr(self.cfg.commands, "deadzone", 0.2)
        # Re-draw tiny cmds (base zeros ||v||<0.2) until they clear the deadzone.
        for _ in range(4):
            small = torch.norm(self.commands[env_ids, :2], dim=1) <= dead
            if not torch.any(small):
                break
            ids = env_ids[small]
            self.commands[ids, 0] = torch_rand_float(
                self.command_ranges["lin_vel_x"][0],
                self.command_ranges["lin_vel_x"][1],
                (len(ids), 1),
                device=self.device,
            ).squeeze(1)
            self.commands[ids, 1] = torch_rand_float(
                self.command_ranges["lin_vel_y"][0],
                self.command_ranges["lin_vel_y"][1],
                (len(ids), 1),
                device=self.device,
            ).squeeze(1)
            self.commands[ids, :2] *= (
                torch.norm(self.commands[ids, :2], dim=1) > dead
            ).unsqueeze(1)
        still_small = torch.norm(self.commands[env_ids, :2], dim=1) <= dead
        if torch.any(still_small):
            ids = env_ids[still_small]
            self.commands[ids, 0] = 0.4
            self.commands[ids, 1] = 0.0

        # Most play/sim2sim runs use vy=0, yaw=0 — train that case heavily.
        p_straight = getattr(self.cfg.commands, "straight_command_prob", 0.7)
        straight = torch.rand(len(env_ids), device=self.device) < p_straight
        if torch.any(straight):
            ids = env_ids[straight]
            self.commands[ids, 1] = 0.0
            self.commands[ids, 2] = 0.0

    def _process_dof_props(self, props, env_id):
        """Assign absolute joint damping/friction (URDF has ~0; multiply would no-op)."""
        props = super()._process_dof_props(props, env_id)
        # Match MuJoCo-like viscous friction for sim2sim robustness (train-side only).
        if self.cfg.domain_rand.randomize_joint_damping:
            d = float(self.joint_damping_coeffs[env_id, 0].item())
            for i in range(len(props)):
                props["damping"][i] = d
        if self.cfg.domain_rand.randomize_joint_friction:
            f = float(self.joint_friction_coeffs[env_id, 0].item())
            for i in range(len(props)):
                props["friction"][i] = f
        return props

    def _reward_commanded_still(self):
        """Penalize near-zero base speed while a locomotion command is active."""
        cmd = torch.norm(self.commands[:, :2], dim=1)
        speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        thr = getattr(self.cfg.rewards, "commanded_still_speed", 0.12)
        return ((cmd > 0.2) & (speed < thr)).float()

    def _reward_uncommanded_yaw(self):
        """Penalize yaw rate when yaw cmd≈0 (play/sim2sim 机头右偏)."""
        yaw_thr = getattr(self.cfg.rewards, "uncommanded_yaw_cmd_thr", 0.1)
        straight = (torch.abs(self.commands[:, 2]) < yaw_thr) & (torch.abs(self.commands[:, 0]) > 0.2)
        # Square + mild linear term so small drift is still discouraged.
        yaw = self.base_ang_vel[:, 2]
        return (torch.square(yaw) + 0.5 * torch.abs(yaw)) * straight.float()

    def _reward_uncommanded_vy(self):
        """Penalize lateral velocity when vy cmd≈0 (anti side-exit / curve walking)."""
        vy_thr = getattr(self.cfg.rewards, "uncommanded_vy_cmd_thr", 0.05)
        straight = (torch.abs(self.commands[:, 1]) < vy_thr) & (torch.abs(self.commands[:, 0]) > 0.2)
        vy = self.base_lin_vel[:, 1]
        return (torch.square(vy) + 0.5 * torch.abs(vy)) * straight.float()

    def _reward_drag_gait(self):
        """Penalize all-four stance while a locomotion command is active."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        moving = torch.norm(self.commands[:, :2], dim=1) > 0.2
        return (contact.all(dim=1) & moving).float()

    def _reward_feet_air_time(self):
        """Reward longer strides; little credit for scurrying hops."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        first_contact = (self.feet_air_time > 0.0) & contact & (~self.last_contacts)
        self.last_contacts = contact
        self.feet_air_time += self.dt
        min_air = getattr(self.cfg.rewards, "min_feet_air_time", 0.08)
        # Credit only air beyond scurry threshold (encourages ~0.12–0.30s swings).
        excess = (self.feet_air_time - min_air).clamp(min=0.0, max=0.25)
        # Soft penalty for landings shorter than min_air (front feet weighted).
        short = (min_air - self.feet_air_time).clamp(min=0.0, max=min_air) / min_air
        w = torch.ones(self.feet_indices.shape[0], device=self.device)
        w[0:2] = 1.5  # FL, FR
        rew = torch.sum(
            (excess - 0.6 * short * w) * first_contact.float(), dim=1
        )
        rew *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew
