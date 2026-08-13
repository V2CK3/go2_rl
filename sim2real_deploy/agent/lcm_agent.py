"""硬件 Agent：组 47 维观测、叠 frame_stack、经 LCM 发 PD（对齐 sim2sim_go2_base）。"""

from __future__ import annotations

import math
import time
from collections import deque

import lcm
import numpy as np
import torch

from sim2real_deploy.agent.deploy_cfg import Go2BaseDeployCfg
from sim2real_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt

LCM_URL = "udpm://239.255.76.67:7667?ttl=255"
lc = lcm.LCM(LCM_URL)


class LCMAgent:
    """组观测、叠历史、发 PD。get_obs/reset/step 返回含 obs_history 的 dict。"""

    def __init__(self, cfg: Go2BaseDeployCfg, state):
        self.deploy_cfg = cfg
        self.cfg = cfg.as_legacy_dict()
        self.state = state

        self.dt = cfg.policy_dt
        self.timestep = 0
        self.device = "cpu"

        self.num_envs = 1
        self.num_obs = cfg.num_single_obs
        self.num_actions = cfg.num_actions
        self.num_privileged_obs = None
        self.num_commands = 3

        self.default_dof_pos = np.asarray(cfg.default_dof_pos, dtype=np.float64)
        self.p_gains = np.full(12, cfg.kp, dtype=np.float64)
        self.d_gains = np.full(12, cfg.kd, dtype=np.float64)
        print(f"[LCMAgent] kp={cfg.kp} kd={cfg.kd} action_scale={cfg.action_scale}")
        print(f"[LCMAgent] default_dof_pos={self.default_dof_pos}")

        self.joint_idxs = self.state.joint_idxs
        self.actions = torch.zeros(12)
        self.last_actions = torch.zeros(12)
        self.commands = np.zeros(3, dtype=np.float64)

        self.dof_pos = np.zeros(12)
        self.dof_vel = np.zeros(12)
        self.body_linear_vel = np.zeros(3)
        self.body_angular_vel = np.zeros(3)
        self.joint_pos_target = np.zeros(12)
        self.joint_vel_target = np.zeros(12)
        self.torques = np.zeros(12)
        self.contact_state = np.ones(4)
        self.is_currently_probing = False
        self.time = time.time()
        self._reset_hist()

    def set_probing(self, is_currently_probing):
        self.is_currently_probing = is_currently_probing

    def get_privileged_observations(self):
        return None

    def _reset_hist(self):
        self.hist = deque(maxlen=self.deploy_cfg.frame_stack)
        for _ in range(self.deploy_cfg.frame_stack):
            self.hist.append(np.zeros((1, self.deploy_cfg.num_single_obs), dtype=np.float32))

    def _pack(self, single_obs: torch.Tensor):
        frame = single_obs.detach().cpu().numpy().astype(np.float32)
        if frame.ndim == 1:
            frame = frame.reshape(1, -1)
        self.hist.append(frame)
        stacked = np.concatenate(list(self.hist), axis=1)
        return {
            "obs": single_obs,
            "privileged_obs": None,
            "obs_history": torch.tensor(stacked, dtype=torch.float32, device=self.device),
        }

    def get_obs(self):
        self.commands[:] = self.state.get_command(probe=self.is_currently_probing)[:3]

        self.dof_pos = self.state.get_dof_pos()
        self.dof_vel = self.state.get_dof_vel()
        self.body_linear_vel = self.state.get_body_linear_vel()
        self.body_angular_vel = self.state.get_body_angular_vel()
        self.contact_state = self.state.get_contact_state()

        eu_ang = np.asarray(self.state.get_rpy(), dtype=np.float64).copy()
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        action_np = (torch.clip(self.actions, -self.deploy_cfg.clip_actions, self.deploy_cfg.clip_actions)
            .detach().cpu().numpy())

        cfg = self.deploy_cfg
        scales = cfg.obs_scales
        t = self.timestep * self.dt
        obs = np.zeros((1, cfg.num_single_obs), dtype=np.float32)
        phase = (t % cfg.cycle_time) / cfg.cycle_time
        obs[0, 0] = math.sin(2 * math.pi * phase)
        obs[0, 1] = math.cos(2 * math.pi * phase)
        obs[0, 2] = float(self.commands[0]) * scales.lin_vel
        obs[0, 3] = float(self.commands[1]) * scales.lin_vel
        obs[0, 4] = float(self.commands[2]) * scales.ang_vel
        obs[0, 5:8] = np.asarray(self.body_angular_vel, dtype=np.float32) * scales.ang_vel
        obs[0, 8:11] = np.asarray(eu_ang, dtype=np.float32) * scales.quat
        obs[0, 11:23] = (np.asarray(self.dof_pos, dtype=np.float32) - np.asarray(cfg.default_dof_pos, dtype=np.float32)) * scales.dof_pos
        obs[0, 23:35] = np.asarray(self.dof_vel, dtype=np.float32) * scales.dof_vel
        obs[0, 35:47] = np.asarray(action_np, dtype=np.float32)
        obs = np.clip(obs, -cfg.clip_observations, cfg.clip_observations)
        return self._pack(torch.tensor(obs, device=self.device, dtype=torch.float32))

    def publish_action(self, action, hard_reset=False):
        command_for_robot = pd_tau_targets_lcmt()
        act = action[0, :12].detach().cpu().numpy().flatten()
        self.joint_pos_target = act * self.deploy_cfg.action_scale
        self.joint_pos_target[[0, 3, 6, 9]] *= self.deploy_cfg.hip_scale_reduction
        self.joint_pos_target = self.joint_pos_target + self.default_dof_pos

        joint_pos_target = self.joint_pos_target[self.joint_idxs]
        self.joint_vel_target = np.zeros(12)

        command_for_robot.q_des = joint_pos_target
        command_for_robot.qd_des = self.joint_vel_target
        command_for_robot.kp = self.p_gains
        command_for_robot.kd = self.d_gains
        command_for_robot.tau_ff = np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10**6)
        command_for_robot.id = -1 if hard_reset else 0

        self.torques = (self.joint_pos_target - self.dof_pos) * self.p_gains + (
            self.joint_vel_target - self.dof_vel
        ) * self.d_gains
        lc.publish("pd_plustau_targets", command_for_robot.encode())

    def reset(self):
        self.actions = torch.zeros(12)
        self.last_actions = torch.zeros(12)
        self.time = time.time()
        self.timestep = 0
        self._reset_hist()
        return self.get_obs()

    def step(self, actions, hard_reset=False):
        clip_actions = self.deploy_cfg.clip_actions
        self.last_actions = self.actions[:]
        self.actions = torch.clip(actions[0:1, :], -clip_actions, clip_actions)
        self.publish_action(self.actions, hard_reset=hard_reset)

        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0:
            print(f"frq: {1 / max(time.time() - self.time, 1e-6):.1f} Hz")
        self.time = time.time()

        obs = self.get_obs()
        infos = {
            "joint_pos": self.dof_pos[np.newaxis, :],
            "joint_vel": self.dof_vel[np.newaxis, :],
            "joint_pos_target": self.joint_pos_target[np.newaxis, :],
            "joint_vel_target": self.joint_vel_target[np.newaxis, :],
            "body_linear_vel": self.body_linear_vel[np.newaxis, :],
            "body_angular_vel": self.body_angular_vel[np.newaxis, :],
            "contact_state": self.contact_state[np.newaxis, :],
            "body_linear_vel_cmd": self.commands[0:2],
            "body_angular_vel_cmd": self.commands[2:3],
            "privileged_obs": None,
        }
        self.timestep += 1
        return obs, None, None, infos
