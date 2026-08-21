"""
go2_stairs 真机部署：45 维、无步态相位、对齐 sim2sim_go2_stairs。

只加载 play 导出的 JIT：logs/go2_stairs/<RUNS>/policies/{RUNS}_{iter}.pt
"""

from __future__ import annotations

from sim2real_deploy.agent.deploy_cfg import Go2StairsDeployCfg
from sim2real_deploy.scripts.deploy_go2_base import run_deploy


def main():
    experiment = "go2_stairs"
    run = None                    # None = latest JIT; or e.g. '2026-08-18_09-31-50_stairs'
    iteration = None              # None = highest iter for that run
    policy_path = "logs/go2_stairs/2026-08-13_09-27-22_stairs/policies/2026-08-13_09-27-22_stairs_5000.pt"            # explicit jit path; else logs/<exp>/<run>/policies/{run}_{iter}.pt

    # Stick max ≈ training command ranges (vx 0.30–0.55, vy ±0.08, yaw ±0.20).
    run_deploy(
        experiment,
        Go2StairsDeployCfg(),
        run=run,
        iteration=iteration,
        policy_path=policy_path,
        x_scale=0.35,
        y_scale=0.08,
        yaw_scale=0.20,
    )


if __name__ == "__main__":
    main()
