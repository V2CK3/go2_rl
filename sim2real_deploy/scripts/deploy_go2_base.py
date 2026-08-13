"""
go2_base 真机部署入口：加载 JIT、校准站立、闭环控制
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

import lcm
from sim2real_deploy.agent.deploy_cfg import Go2BaseDeployCfg
from sim2real_deploy.agent.lcm_agent import LCMAgent
from sim2real_deploy.agent.state_builder import StateBuilder
from sim2real_deploy.scripts.logger import DeployLogger
from sim2sim_deploy.utils.utils import resolve_exported_jit, resolve_plot_meta, run_artifact_dir

LCM_URL = "udpm://239.255.76.67:7667?ttl=255"


def load_jit_policy(path: str, expected_obs_dim: int):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JIT policy not found: {path}")

    policy_mod = torch.jit.load(path, map_location="cpu")
    policy_mod.eval()

    try:
        first = next(policy_mod.parameters())
        in_features = int(first.shape[1]) if first.ndim == 2 else None
        if in_features is not None and in_features != expected_obs_dim:
            raise RuntimeError(f"Policy expects obs dim {in_features}, deploy builds {expected_obs_dim}.")
        print(f"[deploy] JIT input dim OK: {in_features}")
    except StopIteration:
        out = policy_mod(torch.zeros(1, expected_obs_dim))
        print(f"[deploy] JIT forward OK: in={expected_obs_dim} out={tuple(out.shape)}")

    def policy(obs, info):
        x = obs["obs_history"] if isinstance(obs, dict) else obs
        with torch.no_grad():
            return policy_mod(x.to("cpu"))

    return policy

class DeploymentRunner:
    """校准站立 + 策略闭环。R2：校准确认 / 运行中暂停再校准。"""

    def __init__(self, agent, policy, log_root="."):
        self.agent = agent
        self.policy = policy
        self.logger = DeployLogger(agent.cfg, log_root)

    @property
    def state(self):
        return self.agent.state

    def _calibrate(self, wait=True, low=False):
        """插值到站立（或低蹲），wait=True 时两次等 R2。"""
        agent = self.agent
        agent.get_obs()
        joint_pos = agent.dof_pos
        if low:
            final_goal = np.array([0., 0.3, -0.7] * 4)
        else:
            final_goal = np.zeros(12)
        nominal_joint_pos = agent.default_dof_pos

        print("About to calibrate; the robot will stand [Press R2 to calibrate]")
        while wait:
            if self.state.right_lower_right_switch_pressed:
                print(">>>>>>>>>>>>>>> R2 is pressed <<<<<<<<<<<<<")
                self.state.right_lower_right_switch_pressed = False
                break

        hip_reduction = agent.cfg["control"].get("hip_scale_reduction", 1.0)
        action_scale = agent.cfg["control"]["action_scale"]
        cal_action = np.zeros((agent.num_envs, agent.num_actions))
        target = joint_pos - nominal_joint_pos
        while np.max(np.abs(target - final_goal)) > 0.01:
            target -= np.clip((target - final_goal), -0.05, 0.05)
            next_target = target.copy()
            next_target[[0, 3, 6, 9]] /= max(float(hip_reduction), 1e-6)
            cal_action[:, 0:12] = next_target / action_scale
            agent.step(torch.from_numpy(cal_action))
            agent.get_obs()
            time.sleep(0.05)

        print("Starting pose calibrated [Press R2 to start controller]")
        while True:
            if self.state.right_lower_right_switch_pressed:
                print(">>>>>>>>>>>>>>> R2 is pressed again <<<<<<<<<<<<<")
                self.state.right_lower_right_switch_pressed = False
                break

        return agent.reset()

    def run(self, max_steps=100000000, logging=True):
        control_obs = self.agent.reset()
        control_obs = self._calibrate(wait=True)

        try:
            for i in range(max_steps):
                policy_info = {}
                action = self.policy(control_obs, policy_info)
                obs, ret, done, info = self.agent.step(action)

                info.update(policy_info)
                info.update({
                    "observation": obs, "reward": ret, "done": done, "timestep": i,
                    "time": i * self.agent.dt, "action": action,
                    "rpy": self.state.get_rpy(), "torques": self.agent.torques,
                })
                if logging:
                    self.logger.log(info)
                control_obs = obs

                rpy = self.state.get_rpy()
                if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
                    self._calibrate(wait=False, low=True)

                if self.state.right_lower_right_switch_pressed:
                    control_obs = self._calibrate(wait=False)
                    time.sleep(1)
                    self.state.right_lower_right_switch_pressed = False
                    while not self.state.right_lower_right_switch_pressed:
                        time.sleep(0.01)
                    self.state.right_lower_right_switch_pressed = False

            self._calibrate(wait=False)
            self.logger.save()
        except KeyboardInterrupt:
            self.logger.save()


def main():
    experiment = "go2_stairs"
    run = None                    # None = latest JIT; or e.g. '2026-08-13_09-27-22_stairs'
    iteration = None              # None = highest iter for that run
    policy_path = None            # explicit jit path; else logs/<exp>/<run>/policies/{run}_{iter}.pt

    cfg = Go2BaseDeployCfg()
    jit_path = policy_path or resolve_exported_jit(experiment, run, iteration)
    if not run:
        inferred, _ = resolve_plot_meta(experiment, None, jit_path)
        if inferred and inferred != experiment:
            run = inferred
    if not run:
        raise ValueError("set run= (training RUNS) or policy_path named {RUNS}_{iter}.pt")
    log_path = run_artifact_dir(experiment, run, "sim2real")

    print(f"[deploy] Loading JIT: {jit_path}")
    print(f"[deploy] Logs: {log_path}")
    print(
        f"[deploy] obs: single={cfg.num_single_obs} stack={cfg.frame_stack} "
        f"-> {cfg.num_observations}"
    )
    policy = load_jit_policy(jit_path, cfg.num_observations)

    lcm_client = lcm.LCM(LCM_URL)
    state = StateBuilder(
        lcm_client,
        x_scale=0.6,
        y_scale=0.4,
        yaw_scale=0.8
    )
    agent = LCMAgent(cfg, state)
    state.spin()
    print("[deploy] Agent ready")

    runner = DeploymentRunner(agent, policy, log_root=str(log_path))
    print("[deploy] Press R2 twice: stand calibrate -> start policy")
    runner.run(logging=True)


if __name__ == "__main__":
    main()
