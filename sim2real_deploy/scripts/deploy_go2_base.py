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
from sim2sim_deploy.utils.utils import (
    LEGGED_GYM_ROOT_DIR,
    resolve_exported_jit,
    resolve_plot_meta,
    run_artifact_dir,
)

LCM_URL = "udpm://239.255.76.67:7667?ttl=255"


def resolve_jit_path(path: str) -> str:
    """Accept cwd-relative or repo-root-relative JIT paths."""
    if not path:
        raise FileNotFoundError("empty JIT path")
    path = os.path.expanduser(path)
    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        candidates.append(os.path.abspath(path))
        candidates.append(os.path.join(LEGGED_GYM_ROOT_DIR, path))
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"JIT policy not found: {path} (cwd={os.getcwd()}; tried {candidates})"
    )


def load_jit_policy(path: str, expected_obs_dim: int):
    path = resolve_jit_path(path)

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

    def _send_q(self, agent, q_abs):
        """Send an absolute joint target via the usual action → PD path."""
        hip_reduction = float(agent.cfg["control"].get("hip_scale_reduction", 1.0))
        action_scale = float(agent.cfg["control"]["action_scale"])
        offset = np.asarray(q_abs, dtype=np.float64) - agent.default_dof_pos
        offset[[0, 3, 6, 9]] /= max(hip_reduction, 1e-6)
        cal_action = (offset / action_scale).reshape(1, 12)
        agent.step(torch.from_numpy(cal_action))

    def _calibrate(self, wait=True, low=False):
        """插值到站立（或低蹲）。必须在 R2 之后重新读关节，禁止用按键前的旧值/全零。"""
        agent = self.agent
        print("About to calibrate; the robot will stand [Press R2 to calibrate]")
        agent.get_obs()
        self.logger.snapshot_agent(agent, tag="pre-calibrate")
        if wait:
            self._wait_r2("Press R2 to calibrate")

        for _ in range(8):
            agent.get_obs()
            time.sleep(0.02)
        q = np.asarray(agent.dof_pos, dtype=np.float64).copy()
        if not self.state.received_first_legdata or np.max(np.abs(q)) < 0.05:
            raise RuntimeError(
                "Refuse to stand: joint angles are still ~0 (no legdata). "
                "Keep lcm_bridge running and wait until q is not all zeros."
            )

        if low:
            q_goal = agent.default_dof_pos + np.array([0.0, 0.3, -0.7] * 4)
        else:
            q_goal = np.asarray(agent.default_dof_pos, dtype=np.float64).copy()

        max_delta = float(np.max(np.abs(q - q_goal)))
        print(
            f"[calibrate] hold current pose then stand. "
            f"max|q-q_goal|={max_delta:.2f} rad  q={np.round(q, 2)}"
        )
        self.logger.snapshot_agent(agent, tag="calibrate-start")

        q_cmd = q.copy()
        for _ in range(20):
            self._send_q(agent, q_cmd)

        step = 0.008
        while np.max(np.abs(q_cmd - q_goal)) > 0.015:
            q_cmd = q_cmd + np.clip(q_goal - q_cmd, -step, step)
            self._send_q(agent, q_cmd)

        print("Starting pose calibrated [Press R2 to start controller]")
        self.logger.snapshot_agent(agent, tag="post-calibrate")
        self._wait_r2("Press R2 to start controller")

        return agent.reset()

    def _wait_r2(self, what: str):
        """Block until R2 edge; print link status so a dead bridge is obvious."""
        last = 0.0
        while True:
            if self.state.right_lower_right_switch_pressed:
                print(">>>>>>>>>>>>>>> R2 is pressed <<<<<<<<<<<<<")
                self.state.right_lower_right_switch_pressed = False
                return
            now = time.time()
            if now - last >= 2.0:
                self.agent.get_obs()
                q = self.agent.dof_pos
                rpy = np.asarray(self.state.get_rpy(), dtype=np.float64)
                print(
                    f"[deploy] waiting {what} | "
                    f"legdata={'OK' if self.state.received_first_legdata else 'NONE'} "
                    f"rc={'OK' if self.state.received_first_rc else 'NONE'} "
                    f"R2={int(bool(self.state.right_lower_right_switch))} "
                    f"q0={q[0]:.2f} rpy=({rpy[0]:.2f},{rpy[1]:.2f},{rpy[2]:.2f})",
                    flush=True,
                )
                if not self.state.received_first_rc:
                    print(
                        "[deploy] no joystick yet. Start bridge first: "
                        "`sudo ./lcm_bridge enp109s0` (no colon after the NIC), "
                        "press Enter, wait until 'Press L2+B', then press R2.",
                        flush=True,
                    )
                last = now
            time.sleep(0.02)

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


def run_deploy(
    experiment: str,
    cfg,
    *,
    run=None,
    iteration=None,
    policy_path=None,
    x_scale=0.6,
    y_scale=0.4,
    yaw_scale=0.8,
):
    jit_path = resolve_jit_path(policy_path) if policy_path else resolve_exported_jit(experiment, run, iteration)
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
        f"-> {cfg.num_observations} gait_phase={cfg.use_gait_phase}"
    )
    policy = load_jit_policy(jit_path, cfg.num_observations)

    lcm_client = lcm.LCM(LCM_URL)
    state = StateBuilder(
        lcm_client,
        x_scale=x_scale,
        y_scale=y_scale,
        yaw_scale=yaw_scale,
    )
    agent = LCMAgent(cfg, state)
    state.spin()
    print("[deploy] Agent ready")

    runner = DeploymentRunner(agent, policy, log_root=str(log_path))
    print("[deploy] Press R2 twice: stand calibrate -> start policy")
    runner.run(logging=True)


def main():
    experiment = "go2_base"
    run = None                    # None = latest JIT; or e.g. '2026-08-07_18-34-12_base'
    iteration = None              # None = highest iter for that run
    policy_path = None            # explicit jit path; else logs/<exp>/<run>/policies/{run}_{iter}.pt

    run_deploy(
        experiment,
        Go2BaseDeployCfg(),
        run=run,
        iteration=iteration,
        policy_path=policy_path,
        x_scale=0.6,
        y_scale=0.4,
        yaw_scale=0.8,
    )


if __name__ == "__main__":
    main()
