# bridge — DDS ↔ LCM

| 路径 | 说明 |
|------|------|
| `lcm_position_go2.cpp` | 主桥：Unitree SDK2 (DDS) ↔ LCM |
| `lcm_receive.cpp` | LCM 收包自检 |
| `unitree_sdk2/` | 官方 SDK2 |

## 编译

```bash
cd sim2real_deploy
mkdir -p build && cd build
cmake .. && make -j
```

## 运行

```bash
sudo ./lcm_receive
sudo ./lcm_position_go2 eth0
```

若缺 `libddsc.so`：

```bash
export LD_LIBRARY_PATH=$PWD/../bridge/unitree_sdk2/thirdparty/lib/$(uname -m):$LD_LIBRARY_PATH
```
