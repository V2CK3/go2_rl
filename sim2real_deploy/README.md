# sim2real_deploy

lcm_types → bridge → agent → scripts

Go2 真机部署。楼梯策略用 `deploy_go2_stairs.py`（45 维）；平地 gait 策略用 `deploy_go2_base.py`（470 维）。

| 项 | go2_stairs | go2_base |
|----|------------|----------|
| 入口 | `deploy_go2_stairs.py` | `deploy_go2_base.py` |
| 策略 | TorchScript JIT `logs/<exp>/<RUNS>/policies/{RUNS}_{iter}.pt` | 同左 |
| 观测 | 45 × `frame_stack=1`，无相位 | 47 × `frame_stack=10` = 470 |
| 站立角 | `[0, 0.75, -1.5] × 4` | `[0, 0.8, -1.5] × 4` |
| 指令上限 | vx 0.35 / vy 0.08 / yaw 0.20 | vx 0.6 / vy 0.4 / yaw 0.8 |
| PD | kp=20, kd=0.5, `action_scale=0.25` | 同左 |

不要把训练 checkpoint `model/model_*.pt` 直接上机。

## 目录

```text
lcm_types/     LCM 消息（*.lcm 手写；*.py/*.hpp 由 lcm-gen 生成）
bridge/        C++ DDS↔LCM（unitree_sdk2/）
agent/         StateBuilder + LCMAgent + deploy_cfg
scripts/       deploy_go2_stairs.py / deploy_go2_base.py + logger.py
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

在仓库根目录（保证 `sim2real_deploy` 可 import）：

```bash
# 终端 1：网口改成连狗的那块（例如 eth0 / enp109s0）
cd sim2real_deploy/build
sudo ./lcm_bridge eth0

# 终端 2：楼梯策略
python -m sim2real_deploy.scripts.deploy_go2_stairs

# 或平地 gait 策略
python -m sim2real_deploy.scripts.deploy_go2_base
```

JIT / 日志在对应 run 下：`logs/<exp>/<RUNS>/policies/`、`logs/<exp>/<RUNS>/sim2real/`。在入口 `main()` 里设 `run` / `iteration`（都不填则用最新 JIT）。

部署进程会每隔约 0.5 s 打印姿态 / 指令 / 关节；同时写入：

| 文件 | 内容 |
|------|------|
| `status.log` | 与终端相同的可读状态 |
| `state.csv` | 每控制周期一行（cmd、RPY、q、qdes、力矩） |
| `log.pkl` | 结束时完整 dump |

另开终端可 `tail -f logs/go2_stairs/<RUNS>/sim2real/<时间戳>/status.log`。Ctrl+C 会保存 CSV/pickle。

上机：吊起 → 两次 R2（校准 → 开策略）→ 小摇杆试走。急停：手柄 **L2+B** 阻尼。

1. 加载 JIT 与配置
2. `lcm.LCM(LCM_URL)` 加入组播 `239.255.76.67:7667`
3. `StateBuilder` 订阅传感器/遥控，`get_command()` 得到 (vx, vy, yaw)
4. `LCMAgent` 组观测并发 PD
5. `DeploymentRunner`：校准站立 → 闭环；R2 暂停再校准
