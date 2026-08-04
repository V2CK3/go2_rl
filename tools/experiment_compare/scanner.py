"""Scan training runs under logs/ and load configs, metrics, eval results."""

from __future__ import annotations

import copy
import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config_util import REPO_ROOT, load_config_json, save_config_json, snapshot_from_source

DEFAULT_LOGS_DIR = os.path.join(REPO_ROOT, "logs")

DEFAULT_EVAL_RESULTS: Dict[str, Any] = {
    "summary": "",  # 本轮训练总体结论
    "sim2sim": {
        "status": "pending",  # pending | pass | fail | skip
        "date": "",
        "result": "",  # 本轮 sim 结果（现象 / 结论）
        "problems": "",  # 发现的问题
        "solutions": "",  # 对应解决办法
        "notes": "",
        "plots": [],
        "metrics": {},
    },
    "sim2real": {
        "status": "pending",
        "date": "",
        "result": "",
        "problems": "",
        "solutions": "",
        "notes": "",
        "videos": [],
        "images": [],
        "metrics": {},
    },
    # 多条「现象 → 解决办法」记录，便于横向对比与跟踪
    "findings": [
        # {
        #   "domain": "sim2sim" | "sim2real" | "train" | "other",
        #   "phenomenon": "观察到的现象 / 结果",
        #   "solution": "对应解决办法",
        #   "status": "open" | "done" | "wontfix",
        #   "date": "YYYY-MM-DD",
        # }
    ],
}

# Preferred TensorBoard tags for overlays / summary cards.
DEFAULT_METRIC_TAGS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Loss/value_function",
    "Loss/surrogate",
    "Loss/symmetric",
    "Policy/mean_noise_std",
    "Perf/total_fps",
    "Episode/rew_tracking_lin_vel",
    "Episode/rew_tracking_ang_vel",
    "Episode/rew_orientation",
    "Episode/rew_base_height",
    "Episode/rew_collision",
    "Episode/terrain_level",
    "Episode/max_command_x",
]

RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


@dataclass
class RunInfo:
    experiment: str
    run_name: str
    run_dir: str
    label: str
    config: Optional[Dict[str, Any]] = None
    config_source: str = "missing"  # run_json | source_config | missing
    checkpoints: List[int] = field(default_factory=list)
    final_iter: Optional[int] = None
    has_tfevents: bool = False
    eval_results: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_EVAL_RESULTS))
    sim2sim_plots: List[str] = field(default_factory=list)
    exported_policy: Optional[str] = None
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    metric_series: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return f"{self.experiment}/{self.run_name}"


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in (overlay or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def list_experiments(logs_dir: str = DEFAULT_LOGS_DIR) -> List[str]:
    if not os.path.isdir(logs_dir):
        return []
    names = []
    for name in sorted(os.listdir(logs_dir)):
        path = os.path.join(logs_dir, name)
        if os.path.isdir(path):
            names.append(name)
    return names


def list_run_dirs(experiment: str, logs_dir: str = DEFAULT_LOGS_DIR) -> List[str]:
    exp_dir = os.path.join(logs_dir, experiment)
    if not os.path.isdir(exp_dir):
        return []
    runs = []
    for name in sorted(os.listdir(exp_dir)):
        path = os.path.join(exp_dir, name)
        if not os.path.isdir(path):
            continue
        if name.startswith("0_"):
            continue
        if RUN_DIR_RE.match(name) or os.path.isfile(os.path.join(path, "config.json")):
            runs.append(name)
            continue
        # Fallback: treat dirs with model_*.pt or tfevents as runs
        if glob.glob(os.path.join(path, "model_*.pt")) or glob.glob(
            os.path.join(path, "events.out.tfevents.*")
        ):
            runs.append(name)
    return runs


def _checkpoint_iters(run_dir: str) -> List[int]:
    iters = []
    for path in glob.glob(os.path.join(run_dir, "model_*.pt")):
        name = os.path.basename(path)
        m = re.match(r"model_(\d+)\.pt$", name)
        if m:
            iters.append(int(m.group(1)))
    return sorted(iters)


def _find_tfevents(run_dir: str) -> Optional[str]:
    matches = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    return matches[-1] if matches else None


def _exported_dir(experiment: str, logs_dir: str) -> str:
    return os.path.join(logs_dir, experiment, "0_exported")


def _collect_sim2sim_plots(experiment: str, run_name: str, logs_dir: str, eval_results: Dict) -> List[str]:
    plots: List[str] = []
    # From eval_results (relative to run dir or absolute)
    run_dir = os.path.join(logs_dir, experiment, run_name)
    for p in eval_results.get("sim2sim", {}).get("plots") or []:
        if os.path.isabs(p):
            if os.path.isfile(p):
                plots.append(p)
        else:
            cand = os.path.normpath(os.path.join(run_dir, p))
            if os.path.isfile(cand):
                plots.append(cand)
    # Per-run sim2sim folder
    per_run = os.path.join(run_dir, "sim2sim")
    if os.path.isdir(per_run):
        plots.extend(sorted(glob.glob(os.path.join(per_run, "*.png"))))
    # Shared export folder (legacy)
    shared = os.path.join(_exported_dir(experiment, logs_dir), "sim2sim")
    if os.path.isdir(shared):
        plots.extend(sorted(glob.glob(os.path.join(shared, "*.png"))))
    # Deduplicate preserving order
    seen = set()
    unique = []
    for p in plots:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _find_exported_policy(experiment: str, logs_dir: str) -> Optional[str]:
    path = os.path.join(_exported_dir(experiment, logs_dir), "policies", "policy_1.pt")
    return path if os.path.isfile(path) else None


def load_eval_results(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "eval_results.json")
    if not os.path.isfile(path):
        return copy.deepcopy(DEFAULT_EVAL_RESULTS)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _deep_merge(DEFAULT_EVAL_RESULTS, data)


def save_eval_results(run_dir: str, data: Dict[str, Any]) -> str:
    path = os.path.join(run_dir, "eval_results.json")
    merged = _deep_merge(DEFAULT_EVAL_RESULTS, data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    return path


def read_tb_scalars(
    run_dir: str,
    tags: Optional[List[str]] = None,
    max_points: int = 400,
) -> Dict[str, List[Tuple[int, float]]]:
    """Read scalar series from TensorBoard events. Returns {tag: [(step, value), ...]}."""
    event_file = _find_tfevents(run_dir)
    if event_file is None:
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    wanted = tags or sorted(available)
    out: Dict[str, List[Tuple[int, float]]] = {}
    for tag in wanted:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        if not events:
            continue
        if len(events) > max_points:
            step = max(1, len(events) // max_points)
            events = events[::step]
        out[tag] = [(int(e.step), float(e.value)) for e in events]
    return out


def summarize_series(series: Dict[str, List[Tuple[int, float]]]) -> Dict[str, Any]:
    summary = {}
    for tag, points in series.items():
        if not points:
            continue
        vals = [v for _, v in points]
        summary[tag] = {
            "final": vals[-1],
            "best": max(vals),
            "worst": min(vals),
            "last_step": points[-1][0],
            "n": len(points),
        }
    return summary


def discover_runs(logs_dir: str = DEFAULT_LOGS_DIR, load_metrics: bool = False) -> List[RunInfo]:
    runs: List[RunInfo] = []
    for experiment in list_experiments(logs_dir):
        for run_name in list_run_dirs(experiment, logs_dir):
            runs.append(load_run(experiment, run_name, logs_dir, load_metrics=load_metrics))
    return runs


def load_run(
    experiment: str,
    run_name: str,
    logs_dir: str = DEFAULT_LOGS_DIR,
    load_metrics: bool = True,
    metric_tags: Optional[List[str]] = None,
) -> RunInfo:
    run_dir = os.path.join(logs_dir, experiment, run_name)
    label = f"{experiment}/{run_name}"

    config = load_config_json(os.path.join(run_dir, "config.json"))
    config_source = "run_json" if config else "missing"
    if config is None:
        config = snapshot_from_source(experiment)
        if config is not None:
            config_source = "source_config"

    checkpoints = _checkpoint_iters(run_dir)
    eval_results = load_eval_results(run_dir)
    plots = _collect_sim2sim_plots(experiment, run_name, logs_dir, eval_results)

    info = RunInfo(
        experiment=experiment,
        run_name=run_name,
        run_dir=run_dir,
        label=label,
        config=config,
        config_source=config_source,
        checkpoints=checkpoints,
        final_iter=checkpoints[-1] if checkpoints else None,
        has_tfevents=_find_tfevents(run_dir) is not None,
        eval_results=eval_results,
        sim2sim_plots=plots,
        exported_policy=_find_exported_policy(experiment, logs_dir),
    )

    if load_metrics and info.has_tfevents:
        tags = metric_tags or DEFAULT_METRIC_TAGS
        series = read_tb_scalars(run_dir, tags=tags)
        # Also pull any Episode/rew_* that exist
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                if tag.startswith("Episode/") and tag not in series:
                    series.update(read_tb_scalars(run_dir, tags=[tag]))
        except Exception:
            pass
        info.metric_series = series
        info.metrics_summary = summarize_series(series)
    return info


def backfill_config(experiment: str, run_name: str, logs_dir: str = DEFAULT_LOGS_DIR, force: bool = False) -> str:
    """Write config.json for an existing run from current source config."""
    run_dir = os.path.join(logs_dir, experiment, run_name)
    out_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(out_path) and not force:
        return out_path
    cfg = snapshot_from_source(experiment)
    if cfg is None:
        raise ValueError(f"No source config mapping for experiment '{experiment}'")
    save_config_json(out_path, cfg)
    return out_path


def backfill_all(logs_dir: str = DEFAULT_LOGS_DIR, force: bool = False) -> List[str]:
    written = []
    for experiment in list_experiments(logs_dir):
        for run_name in list_run_dirs(experiment, logs_dir):
            try:
                written.append(backfill_config(experiment, run_name, logs_dir, force=force))
            except ValueError:
                continue
    return written
