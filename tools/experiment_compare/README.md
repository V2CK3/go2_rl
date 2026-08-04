# Experiment Compare（实验横向对比）

把多次训练的 **配置参数 / TensorBoard 曲线 / Sim2Sim / Sim2Real** 放在一起横向对比。

## 启动

```bash
conda activate rl_lab
bash tools/experiment_compare/run.sh
# 或
streamlit run tools/experiment_compare/app.py
```

浏览器打开提示的本地地址（默认 `http://localhost:8501`）。

## 功能

| Tab | 内容 |
|-----|------|
| 总览 | 每个 run 的迭代数、checkpoint、sim/real 状态、最终 reward、解决办法速览 |
| 配置对比 | 关键超参 / reward scale 横向表格，高亮差异 |
| 训练曲线 | TensorBoard 标量叠加对比 |
| Sim / Real 对比 | 结果、问题、解决办法、图片 / 视频并排；问题清单汇总表 |
| **人工复盘** | **训练后人工录入本轮 Sim / Real 结果与对应解决办法** |
| 高级编辑 | 兼容旧的 status / 路径编辑 |

## 人工复盘（推荐流程）

1. 训练结束 → 在左侧选中该 run（或任意 run）
2. 打开 **人工复盘** Tab
3. 填写：
   - 本轮总体结论
   - **Sim 结果 / 问题 / 解决办法**
   - **Real 结果 / 问题 / 解决办法**
   - 可选：问题清单多行（现象 → 解决办法）
4. 点「保存本轮复盘」→ 写入 `eval_results.json`
5. 在 **Sim / Real 对比** 里和其他 run 横向对照

`eval_results.json` 示例：

```json
{
  "summary": "平地可用，真机偏软",
  "sim2sim": {
    "status": "pass",
    "date": "2026-08-01",
    "result": "MuJoCo 跟踪正常",
    "problems": "高速 yaw 略抖",
    "solutions": "提高 tracking_ang_vel 权重",
    "plots": ["sim2sim/result_xxx.png"],
    "metrics": {"vx_rmse": 0.12}
  },
  "sim2real": {
    "status": "fail",
    "date": "2026-08-02",
    "result": "真机站立发软，前进易后坐",
    "problems": "仿真刚度偏高 / 缺延迟",
    "solutions": "提高 PD；加 latency rand；降低 action_scale",
    "videos": [],
    "images": [],
    "metrics": {}
  },
  "findings": [
    {
      "domain": "sim2real",
      "phenomenon": "真机比仿真更软",
      "solution": "提高刚度并加 actuator delay 随机",
      "status": "open",
      "date": "2026-08-02"
    }
  ]
}
```

## 回填已有 run

旧训练目录没有 `config.json` 时，可从当前源码配置回填（会标注 `source_config`）：

```bash
python tools/experiment_compare/backfill.py --eval-template
```

## Sim2Sim 写入结果

```bash
python sim2sim_deploy/sim2sim_go2.py \
  --experiment go2_base \
  --run 2026-08-01_12-46-35_demo
```

会把曲线图存到该 run 的 `sim2sim/`，并更新 `eval_results.json`。

## 数据约定

```
logs/<experiment>/<YYYY-MM-DD_HH-MM-SS>_<run_name>/
  config.json            # 训练开始时自动保存
  eval_results.json      # 人工复盘 + sim2sim/sim2real 结果
  events.out.tfevents.*
  model_*.pt
  sim2sim/*.png          # 可选：该 run 专属 sim2sim 图
```
