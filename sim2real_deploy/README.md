# sim2real_deploy

lcm_types → bridge → agent → scripts

Go2 真机部署（对齐本仓库 `go2_base` / `sim2sim_go2_base`）。

| 项 | 值 |
|----|----|
| 策略 | TorchScript JIT `logs/<exp>/<RUNS>/policies/{RUNS}_{iter}.pt` |
| 观测 | 47 × `frame_stack=10` = 470 |
| 指令 | `(vx, vy, yaw)` |
| PD | kp=20, kd=0.5, `action_scale=0.25` |

## 目录

```text
lcm_types/     LCM 消息（*.lcm 手写；*.py/*.hpp 由 lcm-gen 生成）
bridge/        C++ DDS↔LCM（unitree_sdk2/）
agent/         StateBuilder + LCMAgent + deploy_cfg
scripts/       deploy_go2_base.py（入口）+ logger.py
```

## 编译桥接

依赖：系统已安装 LCM；`bridge/unitree_sdk2` 已就绪。

```bash
cd sim2real_deploy
mkdir -p build && cd build
cmake ..
make -j
```

产物：`build/lcm_bridge`、`build/lcm_receive_msgs_test`

## 运行

```bash
# 终端 1
cd sim2real_deploy/build
sudo ./lcm_bridge eth0

# 终端 2
cd sim2real_deploy/scripts
python deploy_go2_base.py
```

JIT / 日志在对应 run 下：`logs/<exp>/<RUNS>/policies/`、`logs/<exp>/<RUNS>/sim2real/`。在 `deploy_go2_base.py` 的 `main()` 里设 `run` / `iteration`（都不填则用最新 JIT）。

上机：吊起 → 两次 R2（校准 → 开策略）→ 小摇杆试走。

1. 加载 JIT 与配置
2. `lcm.LCM(LCM_URL)` 加入组播 `239.255.76.67:7667`
3. `StateBuilder` 订阅传感器/遥控，`get_command()` 得到 (vx, vy, yaw)
4. `LCMAgent` 组观测并发 PD
5. `DeploymentRunner`：校准站立 → 闭环；R2 暂停再校准
