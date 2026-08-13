"""
go2_base 真机部署配置（与 sim2sim_go2_base / Go2BaseCfg 对齐）。

策略输入维数以本配置为准（与训练 / sim2sim 对齐）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ObsScales:
    lin_vel: float = 2.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    quat: float = 1.0


@dataclass
class Go2BaseDeployCfg:
    """与 sim2sim_go2_base.Sim2simCfg 同构的部署配置。"""

    # env
    frame_stack: int = 10
    num_single_obs: int = 47
    num_actions: int = 12

    # control
    action_scale: float = 0.25
    hip_scale_reduction: float = 1.0  # go2_base 无此项；保持 1.0 以兼容校准反解
    kp: float = 20.0
    kd: float = 0.5
    decimation: int = 4
    sim_dt: float = 0.005

    # rewards / gait clock
    cycle_time: float = 0.5

    # normalization
    obs_scales: ObsScales = field(default_factory=ObsScales)
    clip_observations: float = 100.0
    clip_actions: float = 100.0

    # default stand pose: FL/FR/RL/RR × [hip, thigh, calf]
    default_dof_pos: List[float] = field(
        default_factory=lambda: [0.0, 0.8, -1.5] * 4
    )

    @property
    def num_observations(self) -> int:
        return self.frame_stack * self.num_single_obs

    @property
    def policy_dt(self) -> float:
        """策略控制周期 = decimation * sim_dt（训练侧 self.dt）。"""
        return self.decimation * self.sim_dt

    def as_legacy_dict(self) -> dict:
        """
        转成 logger 期望的嵌套 dict 形态。
        仅含校准与日志需要的字段。
        """
        return {
            "env": {
                "num_observations": self.num_observations,
                "num_single_obs": self.num_single_obs,
                "frame_stack": self.frame_stack,
                "num_actions": self.num_actions,
                "num_privileged_obs": None,
            },
            "control": {
                "action_scale": self.action_scale,
                "hip_scale_reduction": self.hip_scale_reduction,
                "decimation": self.decimation,
                "stiffness": {"joint": self.kp},
                "damping": {"joint": self.kd},
                "control_type": "P",
            },
            "sim": {"dt": self.sim_dt},
            "rewards": {"cycle_time": self.cycle_time},
            "normalization": {
                "obs_scales": {
                    "lin_vel": self.obs_scales.lin_vel,
                    "ang_vel": self.obs_scales.ang_vel,
                    "dof_pos": self.obs_scales.dof_pos,
                    "dof_vel": self.obs_scales.dof_vel,
                    "quat": self.obs_scales.quat,
                },
                "clip_observations": self.clip_observations,
                "clip_actions": self.clip_actions,
            },
            "init_state": {
                "default_joint_angles": {
                    "FL_hip_joint": self.default_dof_pos[0],
                    "FL_thigh_joint": self.default_dof_pos[1],
                    "FL_calf_joint": self.default_dof_pos[2],
                    "FR_hip_joint": self.default_dof_pos[3],
                    "FR_thigh_joint": self.default_dof_pos[4],
                    "FR_calf_joint": self.default_dof_pos[5],
                    "RL_hip_joint": self.default_dof_pos[6],
                    "RL_thigh_joint": self.default_dof_pos[7],
                    "RL_calf_joint": self.default_dof_pos[8],
                    "RR_hip_joint": self.default_dof_pos[9],
                    "RR_thigh_joint": self.default_dof_pos[10],
                    "RR_calf_joint": self.default_dof_pos[11],
                }
            },
            "commands": {"num_commands": 3},
        }
