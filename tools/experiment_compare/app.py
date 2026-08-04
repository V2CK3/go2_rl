"""
Horizontal experiment comparison tool.

Launch:
  conda activate rl_lab
  streamlit run tools/experiment_compare/app.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# Allow `streamlit run tools/experiment_compare/app.py` without installing the package.
_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from experiment_compare.config_util import (  # noqa: E402
    diff_flat_configs,
    format_value,
)
from experiment_compare.scanner import (  # noqa: E402
    DEFAULT_LOGS_DIR,
    DEFAULT_METRIC_TAGS,
    discover_runs,
    load_run,
    save_eval_results,
)

st.set_page_config(
    page_title="Go2 Experiment Compare",
    layout="wide",
)

STATUS_LABEL = {
    "pending": "pending",
    "pass": "PASS",
    "fail": "FAIL",
    "skip": "skip",
}


@st.cache_data(show_spinner=False)
def _list_runs(logs_dir: str) -> List[Dict[str, Any]]:
    runs = discover_runs(logs_dir, load_metrics=False)
    return [
        {
            "run_id": r.run_id,
            "experiment": r.experiment,
            "run_name": r.run_name,
            "final_iter": r.final_iter,
            "has_tfevents": r.has_tfevents,
            "config_source": r.config_source,
            "sim2sim": r.eval_results.get("sim2sim", {}).get("status", "pending"),
            "sim2real": r.eval_results.get("sim2real", {}).get("status", "pending"),
            "n_plots": len(r.sim2sim_plots),
        }
        for r in runs
    ]


@st.cache_data(show_spinner="Loading selected runs…")
def _load_selected(logs_dir: str, run_ids: tuple, metric_tags: tuple) -> List[Dict[str, Any]]:
    loaded = []
    for run_id in run_ids:
        experiment, run_name = run_id.split("/", 1)
        info = load_run(
            experiment,
            run_name,
            logs_dir=logs_dir,
            load_metrics=True,
            metric_tags=list(metric_tags),
        )
        loaded.append(
            {
                "run_id": info.run_id,
                "label": info.label,
                "experiment": info.experiment,
                "run_name": info.run_name,
                "run_dir": info.run_dir,
                "config": info.config,
                "config_source": info.config_source,
                "checkpoints": info.checkpoints,
                "final_iter": info.final_iter,
                "has_tfevents": info.has_tfevents,
                "eval_results": info.eval_results,
                "sim2sim_plots": info.sim2sim_plots,
                "exported_policy": info.exported_policy,
                "metrics_summary": info.metrics_summary,
                "metric_series": info.metric_series,
            }
        )
    return loaded


def _status_badge(status: str) -> str:
    return f"`{STATUS_LABEL.get(status, status)}`"


def _run_column_labels(runs: List[Dict[str, Any]]) -> List[str]:
    """Unique short column labels for tables."""
    names = [r["run_name"] for r in runs]
    if len(set(names)) == len(names):
        return names
    return [r["run_id"] for r in runs]


def render_overview(runs: List[Dict[str, Any]]) -> None:
    cols = st.columns(len(runs))
    for col, run in zip(cols, runs):
        with col:
            st.subheader(run["run_name"])
            st.caption(run["experiment"])
            st.markdown(
                f"""
| | |
|---|---|
| **iter** | `{run['final_iter'] if run['final_iter'] is not None else '—'}` |
| **ckpt** | `{len(run['checkpoints'])}` |
| **TB** | `{'yes' if run['has_tfevents'] else 'no'}` |
| **config** | `{run['config_source']}` |
| **sim2sim** | {_status_badge(run['eval_results']['sim2sim']['status'])} |
| **sim2real** | {_status_badge(run['eval_results']['sim2real']['status'])} |
"""
            )
            rew = run["metrics_summary"].get("Train/mean_reward")
            if rew:
                st.metric("mean reward (final)", f"{rew['final']:.3f}", help=f"best={rew['best']:.3f}")
            summary = (run["eval_results"].get("summary") or "").strip()
            if summary:
                st.caption(summary)
            s2s_sol = (run["eval_results"]["sim2sim"].get("solutions") or "").strip()
            s2r_sol = (run["eval_results"]["sim2real"].get("solutions") or "").strip()
            if s2s_sol:
                st.markdown(f"**Sim 办法：** {s2s_sol}")
            if s2r_sol:
                st.markdown(f"**Real 办法：** {s2r_sol}")


def render_config_compare(runs: List[Dict[str, Any]], only_highlight: bool, only_diff: bool) -> None:
    labels = _run_column_labels(runs)
    configs = {lab: (r["config"] or {}) for lab, r in zip(labels, runs)}

    rows = diff_flat_configs(configs, only_highlight=only_highlight, only_diff=only_diff)
    if not rows:
        st.info("所选 run 在当前过滤条件下没有差异（或没有可用配置）。")
        return

    table = {"param": [row["key"] for row in rows]}
    for lab in labels:
        table[lab] = [format_value(row.get(lab)) for row in rows]
    table["diff"] = ["≠" if row["differs"] else "=" for row in rows]
    df = pd.DataFrame(table)

    def _highlight(row):
        if row["diff"] == "≠":
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(df.style.apply(_highlight, axis=1), use_container_width=True, height=min(560, 40 + 28 * len(df)))
    sources = ", ".join(f"{r['run_name']}={r['config_source']}" for r in runs)
    st.caption(f"Config source: {sources}. `source_config` 表示从当前源码回填，可能与训练时不完全一致。")


def render_metrics(runs: List[Dict[str, Any]], selected_tags: List[str]) -> None:
    import plotly.graph_objects as go

    # Summary table
    summary_rows = []
    for tag in selected_tags:
        row = {"metric": tag}
        any_present = False
        for r in runs:
            s = r["metrics_summary"].get(tag)
            if s:
                row[r["run_name"]] = f"{s['final']:.4g} (best {s['best']:.4g})"
                any_present = True
            else:
                row[r["run_name"]] = "—"
        if any_present:
            summary_rows.append(row)
    if summary_rows:
        st.markdown("**Final / best scalars**")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    for tag in selected_tags:
        fig = go.Figure()
        has_data = False
        for r in runs:
            series = r["metric_series"].get(tag) or []
            if not series:
                continue
            has_data = True
            xs = [p[0] for p in series]
            ys = [p[1] for p in series]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=r["run_name"]))
        if not has_data:
            continue
        fig.update_layout(
            title=tag,
            xaxis_title="iteration",
            yaxis_title=tag.split("/")[-1],
            height=320,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_eval(runs: List[Dict[str, Any]]) -> None:
    st.markdown("### Sim / Real 结果与解决办法（横向对比）")
    # Summary row
    if any((r["eval_results"].get("summary") or "").strip() for r in runs):
        st.markdown("**本轮总结**")
        cols = st.columns(len(runs))
        for col, run in zip(cols, runs):
            with col:
                st.markdown(f"**{run['run_name']}**")
                st.write(run["eval_results"].get("summary") or "—")

    cols = st.columns(len(runs))
    for col, run in zip(cols, runs):
        with col:
            s2s = run["eval_results"]["sim2sim"]
            s2r = run["eval_results"]["sim2real"]
            st.markdown(f"**{run['run_name']}**")

            st.markdown(f"#### Sim2Sim {_status_badge(s2s.get('status', 'pending'))}")
            if s2s.get("date"):
                st.caption(f"date: {s2s['date']}")
            _render_result_block(s2s)
            metrics = s2s.get("metrics") or {}
            if metrics:
                st.json(metrics)
            for plot in run["sim2sim_plots"][:3]:
                st.image(plot, caption=os.path.basename(plot), use_container_width=True)
            if len(run["sim2sim_plots"]) > 3:
                st.caption(f"+{len(run['sim2sim_plots']) - 3} more plots")

            st.divider()
            st.markdown(f"#### Sim2Real {_status_badge(s2r.get('status', 'pending'))}")
            if s2r.get("date"):
                st.caption(f"date: {s2r['date']}")
            _render_result_block(s2r)
            metrics = s2r.get("metrics") or {}
            if metrics:
                st.json(metrics)
            for img in (s2r.get("images") or [])[:3]:
                path = img if os.path.isabs(img) else os.path.normpath(os.path.join(run["run_dir"], img))
                if os.path.isfile(path):
                    st.image(path, caption=os.path.basename(path), use_container_width=True)
            for vid in (s2r.get("videos") or [])[:2]:
                path = vid if os.path.isabs(vid) else os.path.normpath(os.path.join(run["run_dir"], vid))
                if os.path.isfile(path):
                    st.video(path)

    # Findings table across runs
    finding_rows = []
    for run in runs:
        for i, item in enumerate(run["eval_results"].get("findings") or []):
            finding_rows.append(
                {
                    "run": run["run_name"],
                    "domain": item.get("domain", ""),
                    "phenomenon": item.get("phenomenon", ""),
                    "solution": item.get("solution", ""),
                    "status": item.get("status", "open"),
                    "date": item.get("date", ""),
                }
            )
    if finding_rows:
        st.markdown("### 问题 → 解决办法一览")
        st.dataframe(pd.DataFrame(finding_rows), use_container_width=True, height=min(360, 40 + 32 * len(finding_rows)))


def _render_result_block(section: Dict[str, Any]) -> None:
    if section.get("result"):
        st.markdown("**结果**")
        st.write(section["result"])
    if section.get("problems"):
        st.markdown("**问题**")
        st.write(section["problems"])
    if section.get("solutions"):
        st.markdown("**解决办法**")
        st.write(section["solutions"])
    if section.get("notes") and not section.get("result"):
        st.write(section["notes"])
    elif section.get("notes"):
        st.caption(section["notes"])


FINDING_DOMAINS = ["sim2sim", "sim2real", "train", "other"]
FINDING_STATUSES = ["open", "done", "wontfix"]


def _findings_to_editor_rows(findings: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows = []
    for item in findings or []:
        rows.append(
            {
                "domain": item.get("domain") or "sim2sim",
                "phenomenon": item.get("phenomenon") or "",
                "solution": item.get("solution") or "",
                "status": item.get("status") or "open",
                "date": item.get("date") or "",
            }
        )
    return rows


def _editor_rows_to_findings(df: pd.DataFrame) -> List[Dict[str, str]]:
    findings = []
    for _, row in df.iterrows():
        phenomenon = str(row.get("phenomenon") or "").strip()
        solution = str(row.get("solution") or "").strip()
        if not phenomenon and not solution:
            continue
        findings.append(
            {
                "domain": str(row.get("domain") or "other").strip() or "other",
                "phenomenon": phenomenon,
                "solution": solution,
                "status": str(row.get("status") or "open").strip() or "open",
                "date": str(row.get("date") or "").strip(),
            }
        )
    return findings


def render_review_editor(logs_dir: str, compare_runs: List[Dict[str, Any]], catalog: List[Dict[str, Any]]) -> None:
    """人工复盘环节：录入本轮 sim / real 结果与解决办法。"""
    st.markdown("### 人工复盘")
    st.caption(
        "训练结束后在此记录本轮 **Sim 结果 / Real 结果**，以及对应的 **问题与解决办法**。"
        " 内容写入 run 目录的 `eval_results.json`，可在「Sim2Sim / Sim2Real」页横向对比。"
    )

    all_ids = [r["run_id"] for r in catalog]
    default_id = compare_runs[0]["run_id"] if compare_runs else (all_ids[0] if all_ids else None)
    if not default_id:
        st.warning("没有可编辑的 run。")
        return

    target = st.selectbox(
        "选择本轮训练 run",
        all_ids,
        index=all_ids.index(default_id) if default_id in all_ids else 0,
        key="review_target_run",
    )
    experiment, run_name = target.split("/", 1)

    # Prefer already-loaded compare run; otherwise load on demand
    run = next((r for r in compare_runs if r["run_id"] == target), None)
    if run is None:
        info = load_run(experiment, run_name, logs_dir=logs_dir, load_metrics=False)
        run = {
            "run_id": info.run_id,
            "run_name": info.run_name,
            "run_dir": info.run_dir,
            "eval_results": info.eval_results,
            "sim2sim_plots": info.sim2sim_plots,
        }

    data = run["eval_results"]
    s2s = data.get("sim2sim") or {}
    s2r = data.get("sim2real") or {}

    summary = st.text_area(
        "本轮总体结论",
        data.get("summary") or "",
        height=80,
        key=f"review_summary_{target}",
        placeholder="例如：平地策略 sim 可用，真机偏软需提高刚度 / 降低 action scale",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Sim（Sim2Sim）")
        s2s_status = st.selectbox(
            "Sim 状态",
            ["pending", "pass", "fail", "skip"],
            index=["pending", "pass", "fail", "skip"].index(s2s.get("status", "pending")),
            key=f"review_s2s_status_{target}",
        )
        s2s_date = st.text_input("Sim 日期", s2s.get("date", ""), key=f"review_s2s_date_{target}")
        s2s_result = st.text_area(
            "Sim 结果",
            s2s.get("result") or s2s.get("notes") or "",
            height=100,
            key=f"review_s2s_result_{target}",
            placeholder="本轮仿真表现：跟踪是否稳定、步态、晃动、摔倒等",
        )
        s2s_problems = st.text_area(
            "Sim 问题",
            s2s.get("problems") or "",
            height=80,
            key=f"review_s2s_problems_{target}",
            placeholder="例如：高速 yaw 跟踪滞后；站立时抖腿",
        )
        s2s_solutions = st.text_area(
            "Sim 解决办法",
            s2s.get("solutions") or "",
            height=100,
            key=f"review_s2s_solutions_{target}",
            placeholder="例如：提高 tracking_ang_vel 权重；增大 sym_coef；加站立惩罚",
        )
        s2s_metrics_raw = st.text_area(
            "Sim metrics (JSON，可选)",
            json.dumps(s2s.get("metrics") or {}, indent=2, ensure_ascii=False),
            height=90,
            key=f"review_s2s_metrics_{target}",
        )
        s2s_plots_raw = st.text_area(
            "Sim 图片路径（一行一个）",
            "\n".join(s2s.get("plots") or []),
            height=70,
            key=f"review_s2s_plots_{target}",
        )

    with c2:
        st.markdown("#### Real（Sim2Real）")
        s2r_status = st.selectbox(
            "Real 状态",
            ["pending", "pass", "fail", "skip"],
            index=["pending", "pass", "fail", "skip"].index(s2r.get("status", "pending")),
            key=f"review_s2r_status_{target}",
        )
        s2r_date = st.text_input("Real 日期", s2r.get("date", ""), key=f"review_s2r_date_{target}")
        s2r_result = st.text_area(
            "Real 结果",
            s2r.get("result") or s2r.get("notes") or "",
            height=100,
            key=f"review_s2r_result_{target}",
            placeholder="真机表现：能否站稳、走直线、抖动、过热、摔机等",
        )
        s2r_problems = st.text_area(
            "Real 问题",
            s2r.get("problems") or "",
            height=80,
            key=f"review_s2r_problems_{target}",
            placeholder="例如：真机比仿真更软；启动瞬间后坐；单侧摩擦不一致",
        )
        s2r_solutions = st.text_area(
            "Real 解决办法",
            s2r.get("solutions") or "",
            height=100,
            key=f"review_s2r_solutions_{target}",
            placeholder="例如：提高 PD 刚度；加 delay randomization；降低 action_scale；补质量随机",
        )
        s2r_metrics_raw = st.text_area(
            "Real metrics (JSON，可选)",
            json.dumps(s2r.get("metrics") or {}, indent=2, ensure_ascii=False),
            height=90,
            key=f"review_s2r_metrics_{target}",
        )
        s2r_images_raw = st.text_area(
            "Real 图片路径（一行一个）",
            "\n".join(s2r.get("images") or []),
            height=70,
            key=f"review_s2r_images_{target}",
        )
        s2r_videos_raw = st.text_area(
            "Real 视频路径（一行一个）",
            "\n".join(s2r.get("videos") or []),
            height=70,
            key=f"review_s2r_videos_{target}",
        )

    st.markdown("#### 问题清单（现象 → 解决办法）")
    st.caption("可增删多行；适合把本轮发现拆成可跟踪的条目，横向对比时会汇总成表。")
    base_rows = _findings_to_editor_rows(data.get("findings") or [])
    if not base_rows:
        base_rows = [{"domain": "sim2sim", "phenomenon": "", "solution": "", "status": "open", "date": ""}]
    edited = st.data_editor(
        pd.DataFrame(base_rows),
        num_rows="dynamic",
        use_container_width=True,
        key=f"review_findings_editor_{target}",
        column_config={
            "domain": st.column_config.SelectboxColumn("领域", options=FINDING_DOMAINS, required=True),
            "phenomenon": st.column_config.TextColumn("现象 / 结果", width="large"),
            "solution": st.column_config.TextColumn("解决办法", width="large"),
            "status": st.column_config.SelectboxColumn("状态", options=FINDING_STATUSES, required=True),
            "date": st.column_config.TextColumn("日期"),
        },
    )

    if st.button("保存本轮复盘", type="primary", key=f"review_save_{target}"):
        try:
            s2s_metrics = json.loads(s2s_metrics_raw or "{}")
            s2r_metrics = json.loads(s2r_metrics_raw or "{}")
        except json.JSONDecodeError as e:
            st.error(f"metrics JSON 解析失败: {e}")
            return
        payload = {
            "summary": summary.strip(),
            "sim2sim": {
                "status": s2s_status,
                "date": s2s_date.strip(),
                "result": s2s_result.strip(),
                "problems": s2s_problems.strip(),
                "solutions": s2s_solutions.strip(),
                "notes": s2s.get("notes") or "",
                "plots": [p.strip() for p in s2s_plots_raw.splitlines() if p.strip()],
                "metrics": s2s_metrics,
            },
            "sim2real": {
                "status": s2r_status,
                "date": s2r_date.strip(),
                "result": s2r_result.strip(),
                "problems": s2r_problems.strip(),
                "solutions": s2r_solutions.strip(),
                "notes": s2r.get("notes") or "",
                "images": [p.strip() for p in s2r_images_raw.splitlines() if p.strip()],
                "videos": [p.strip() for p in s2r_videos_raw.splitlines() if p.strip()],
                "metrics": s2r_metrics,
            },
            "findings": _editor_rows_to_findings(edited),
        }
        path = save_eval_results(run["run_dir"], payload)
        st.success(f"已保存复盘: {path}")
        st.cache_data.clear()


def render_eval_editor(runs: List[Dict[str, Any]]) -> None:
    st.markdown("### 编辑评测结果（兼容旧入口）")
    st.caption("建议改用「人工复盘」页。此处仍可快速改 status / 路径。")
    target = st.selectbox("选择 run", [r["run_id"] for r in runs], key="legacy_edit_target")
    run = next(r for r in runs if r["run_id"] == target)
    data = run["eval_results"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sim2Sim**")
        s2s_status = st.selectbox(
            "sim2sim status",
            ["pending", "pass", "fail", "skip"],
            index=["pending", "pass", "fail", "skip"].index(data["sim2sim"].get("status", "pending")),
            key="s2s_status",
        )
        s2s_date = st.text_input("sim2sim date", data["sim2sim"].get("date", ""), key="s2s_date")
        s2s_notes = st.text_area("sim2sim notes", data["sim2sim"].get("notes", ""), height=120, key="s2s_notes")
        s2s_metrics_raw = st.text_area(
            "sim2sim metrics (JSON)",
            json.dumps(data["sim2sim"].get("metrics") or {}, indent=2, ensure_ascii=False),
            height=120,
            key="s2s_metrics",
        )
        s2s_plots_raw = st.text_area(
            "sim2sim plots (一行一个路径，相对 run 目录或绝对路径)",
            "\n".join(data["sim2sim"].get("plots") or []),
            height=80,
            key="s2s_plots",
        )
    with c2:
        st.markdown("**Sim2Real**")
        s2r_status = st.selectbox(
            "sim2real status",
            ["pending", "pass", "fail", "skip"],
            index=["pending", "pass", "fail", "skip"].index(data["sim2real"].get("status", "pending")),
            key="s2r_status",
        )
        s2r_date = st.text_input("sim2real date", data["sim2real"].get("date", ""), key="s2r_date")
        s2r_notes = st.text_area("sim2real notes", data["sim2real"].get("notes", ""), height=120, key="s2r_notes")
        s2r_metrics_raw = st.text_area(
            "sim2real metrics (JSON)",
            json.dumps(data["sim2real"].get("metrics") or {}, indent=2, ensure_ascii=False),
            height=120,
            key="s2r_metrics",
        )
        s2r_images_raw = st.text_area(
            "sim2real images (一行一个)",
            "\n".join(data["sim2real"].get("images") or []),
            height=80,
            key="s2r_images",
        )
        s2r_videos_raw = st.text_area(
            "sim2real videos (一行一个)",
            "\n".join(data["sim2real"].get("videos") or []),
            height=80,
            key="s2r_videos",
        )

    if st.button("保存 eval_results.json", type="primary", key="legacy_save"):
        try:
            s2s_metrics = json.loads(s2s_metrics_raw or "{}")
            s2r_metrics = json.loads(s2r_metrics_raw or "{}")
        except json.JSONDecodeError as e:
            st.error(f"metrics JSON 解析失败: {e}")
            return
        # Preserve richer fields written by 人工复盘
        s2s_old = data.get("sim2sim") or {}
        s2r_old = data.get("sim2real") or {}
        payload = {
            "summary": data.get("summary") or "",
            "findings": data.get("findings") or [],
            "sim2sim": {
                "status": s2s_status,
                "date": s2s_date,
                "result": s2s_old.get("result") or "",
                "problems": s2s_old.get("problems") or "",
                "solutions": s2s_old.get("solutions") or "",
                "notes": s2s_notes,
                "plots": [p.strip() for p in s2s_plots_raw.splitlines() if p.strip()],
                "metrics": s2s_metrics,
            },
            "sim2real": {
                "status": s2r_status,
                "date": s2r_date,
                "result": s2r_old.get("result") or "",
                "problems": s2r_old.get("problems") or "",
                "solutions": s2r_old.get("solutions") or "",
                "notes": s2r_notes,
                "images": [p.strip() for p in s2r_images_raw.splitlines() if p.strip()],
                "videos": [p.strip() for p in s2r_videos_raw.splitlines() if p.strip()],
                "metrics": s2r_metrics,
            },
        }
        path = save_eval_results(run["run_dir"], payload)
        st.success(f"已保存: {path}")
        st.cache_data.clear()


def main() -> None:
    st.title("Go2 实验横向对比")
    st.caption("对比训练配置、TensorBoard 曲线、Sim / Real 结果与解决办法")

    with st.sidebar:
        st.header("Runs")
        logs_dir = st.text_input("logs 目录", DEFAULT_LOGS_DIR)
        if st.button("刷新列表"):
            st.cache_data.clear()

        catalog = _list_runs(logs_dir)
        if not catalog:
            st.warning("未找到训练 run。请确认 logs 目录。")
            return

        experiments = sorted({r["experiment"] for r in catalog})
        picked_exps = st.multiselect("实验", experiments, default=experiments)
        filtered = [r for r in catalog if r["experiment"] in picked_exps]

        st.dataframe(
            pd.DataFrame(filtered)[
                ["run_id", "final_iter", "sim2sim", "sim2real", "config_source"]
            ],
            use_container_width=True,
            height=220,
        )

        default_ids = [r["run_id"] for r in filtered[:2]]
        selected_ids = st.multiselect(
            "选择要对比的 run（建议 2–4 个）",
            [r["run_id"] for r in filtered],
            default=default_ids,
        )

        st.divider()
        only_highlight = st.checkbox("只显示关键配置项", value=True)
        only_diff = st.checkbox("只显示有差异的配置", value=True)
        metric_tags = st.multiselect(
            "曲线指标",
            DEFAULT_METRIC_TAGS,
            default=[
                "Train/mean_reward",
                "Train/mean_episode_length",
                "Episode/rew_tracking_lin_vel",
                "Loss/surrogate",
            ],
        )

    # 人工复盘不依赖对比选择；对比页需要至少 1 个 run
    tab_overview, tab_cfg, tab_metrics, tab_eval, tab_review, tab_edit = st.tabs(
        ["总览", "配置对比", "训练曲线", "Sim / Real 对比", "人工复盘", "高级编辑"]
    )

    runs: List[Dict[str, Any]] = []
    if selected_ids:
        runs = _load_selected(logs_dir, tuple(selected_ids), tuple(metric_tags))

    with tab_overview:
        if not runs:
            st.info("请在左侧至少选择一个 run。")
        else:
            render_overview(runs)
            # Quick glance of solutions on overview
            st.markdown("#### 解决办法速览")
            quick = []
            for r in runs:
                s2s = r["eval_results"]["sim2sim"]
                s2r = r["eval_results"]["sim2real"]
                quick.append(
                    {
                        "run": r["run_name"],
                        "sim_solutions": (s2s.get("solutions") or "")[:120],
                        "real_solutions": (s2r.get("solutions") or "")[:120],
                        "findings": len(r["eval_results"].get("findings") or []),
                    }
                )
            st.dataframe(pd.DataFrame(quick), use_container_width=True)
    with tab_cfg:
        if not runs:
            st.info("请在左侧至少选择一个 run。")
        else:
            render_config_compare(runs, only_highlight=only_highlight, only_diff=only_diff)
    with tab_metrics:
        if not runs:
            st.info("请在左侧至少选择一个 run。")
        else:
            render_metrics(runs, selected_tags=metric_tags)
    with tab_eval:
        if not runs:
            st.info("请在左侧至少选择一个 run。")
        else:
            render_eval(runs)
    with tab_review:
        render_review_editor(logs_dir, runs, filtered)
    with tab_edit:
        if not runs:
            st.info("请在左侧至少选择一个 run。")
        else:
            render_eval_editor(runs)


if __name__ == "__main__":
    main()
