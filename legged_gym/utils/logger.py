# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from __future__ import annotations

import math
import os
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FOOT_LABELS = ("FL", "FR", "RL", "RR")


TB_PLOT_GROUPS = (
    ("Episode", "tb_reward"),
    ("Loss", "tb_loss"),
    ("Train", "tb_train"),
)

# TensorBoard UI default smoothing weight (~0.6).
TB_SMOOTH_WEIGHT = 0.6


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _has(log: Dict[str, Any], key: str) -> bool:
    return bool(log.get(key))

def _as_1d(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)

def _apply_plot_rc(font_size: int = 11) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size + 1,
            "axes.titlesize": font_size + 1,
            "legend.fontsize": max(font_size - 2, 8),
            "xtick.labelsize": max(font_size - 1, 8),
            "ytick.labelsize": max(font_size - 1, 8),
        }
    )

def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", length=4, width=1)

def _make_axes_grid(n: int, ncols: int = 3, cell_size: Tuple[float, float] = (4.2, 2.8)):
    """Create a nrows x ncols axes grid; unused cells are turned off by caller."""
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))
    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(cell_size[0] * ncols, max(cell_size[1], cell_size[1] * nrows)),
        squeeze=False,
    )
    return fig, axs, nrows, ncols

def _hide_unused_axes(axs, n_used: int, nrows: int, ncols: int) -> None:
    for j in range(n_used, nrows * ncols):
        r, c = divmod(j, ncols)
        axs[r, c].axis("off")

def save_figure(
    fig,
    save_dir: str,
    filename: str,
    dpi: int = 150,
    layout: bool = True,
    tight_rect: Optional[Sequence[float]] = None,
) -> str:
    """Shared PNG export used by play/sim2sim state plots and TB curve dumps."""
    os.makedirs(save_dir, exist_ok=True)
    if layout:
        if tight_rect is not None:
            fig.tight_layout(rect=list(tight_rect))
        else:
            fig.tight_layout()
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


def _plot_series(
    ax,
    x,
    y_keys: Sequence[Tuple[str, str, str]],
    log: Dict[str, Any],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """Plot optional (key, style, label) series from ``log`` onto ``ax``."""
    for key, style, label in y_keys:
        if _has(log, key):
            ax.plot(x, log[key], style, label=label)
    _style(ax, xlabel, ylabel, title)


# ---------------------------------------------------------------------------
# Play / sim2sim state figure
# ---------------------------------------------------------------------------

def build_state_figure(
    log: Dict[str, list],
    dt: float,
    foot_labels: Sequence[str] = DEFAULT_FOOT_LABELS,
    suptitle: Optional[str] = None,
):
    """Build the shared 3x3 state figure used by play and sim2sim."""
    if not log or not _has(log, "base_vel_x"):
        return None

    n = len(log["base_vel_x"])
    t = np.linspace(0.0, n * dt, n)
    _apply_plot_rc(11)
    fig, axs, _, _ = _make_axes_grid(9, ncols=3, cell_size=(14 / 3, 10 / 3))

    # Row 0: base velocity tracking
    _plot_series(
        axs[0, 0], t,
        (("base_vel_x", "-", "measured"), ("command_x", "--", "command")),
        log, "time [s]", "vx [m/s]", "Base vel X",
    )
    _plot_series(
        axs[0, 1], t,
        (("base_vel_y", "-", "measured"), ("command_y", "--", "command")),
        log, "time [s]", "vy [m/s]", "Base vel Y",
    )
    _plot_series(
        axs[0, 2], t,
        (("base_vel_yaw", "-", "measured"), ("command_yaw", "--", "command")),
        log, "time [s]", "yaw [rad/s]", "Base vel Yaw",
    )

    # Row 1: DOF / torque
    _plot_series(
        axs[1, 0], t,
        (("dof_pos", "-", "measured"), ("dof_pos_target", "--", "target")),
        log, "time [s]", "pos [rad]", "DOF position",
    )
    _plot_series(
        axs[1, 1], t,
        (("dof_vel", "-", "measured"), ("dof_vel_target", "--", "target")),
        log, "time [s]", "vel [rad/s]", "DOF velocity",
    )
    _plot_series(
        axs[1, 2], t,
        (("dof_torque", "-", "measured"),),
        log, "time [s]", "torque [N·m]", "Torque",
    )

    # Row 2: contacts / torque-vel / vz
    if _has(log, "contact_forces_z"):
        forces = np.asarray(log["contact_forces_z"], dtype=np.float64)
        if forces.ndim == 1:
            forces = forces.reshape(-1, 1)
        for i in range(forces.shape[1]):
            label = foot_labels[i] if i < len(foot_labels) else f"foot_{i}"
            axs[2, 0].plot(t[: forces.shape[0]], forces[:, i], label=label)
    _style(axs[2, 0], "time [s]", "Fz [N]", "Contact Fz")

    if _has(log, "dof_vel") and _has(log, "dof_torque"):
        axs[2, 1].plot(log["dof_vel"], log["dof_torque"], "x", markersize=2, label="measured")
    _style(axs[2, 1], "joint vel [rad/s]", "torque [N·m]", "Torque / velocity")

    _plot_series(
        axs[2, 2], t,
        (("base_vel_z", "-", "measured"),),
        log, "time [s]", "vz [m/s]", "Base vel Z",
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# TensorBoard curve figures
# ---------------------------------------------------------------------------

def load_tb_scalars(
    log_dir: str,
    tags: Optional[Sequence[str]] = None,
    max_points: int = 2000,
) -> Dict[str, List[Tuple[int, float]]]:
    """Load TensorBoard scalar series from a training run directory."""
    if not log_dir or not os.path.isdir(log_dir):
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not installed; skip TB plots.")
        return {}

    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    wanted = list(tags) if tags is not None else sorted(available)
    out: Dict[str, List[Tuple[int, float]]] = {}
    for tag in wanted:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        if not events:
            continue
        if len(events) > max_points:
            stride = max(1, len(events) // max_points)
            events = events[::stride]
        out[tag] = [(int(e.step), float(e.value)) for e in events]
    return out


def _filter_tb_group(
    series: Dict[str, List[Tuple[int, float]]],
    prefix: str,
) -> Dict[str, List[Tuple[int, float]]]:
    """Pick tags under ``prefix/``; drop ``*/time`` duplicates for Train."""
    key = prefix.rstrip("/") + "/"
    selected = {
        tag: points
        for tag, points in series.items()
        if tag.startswith(key) and not tag.endswith("/time")
    }
    return dict(sorted(selected.items()))


def tb_smooth(values: Sequence[float], weight: float = TB_SMOOTH_WEIGHT) -> np.ndarray:
    """
    TensorBoard scalar smoothing (EMA + debias).

    Matches vz_line_chart2:
    last = last * weight + (1 - weight) * x
    smoothed = last / (1 - weight ** n)
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return vals
    weight = float(np.clip(weight, 0.0, 0.999))
    if weight <= 0.0:
        return vals.copy()

    last = 0.0
    num_acc = 0
    out = np.empty_like(vals)
    for i, x in enumerate(vals):
        last = last * weight + (1.0 - weight) * x
        num_acc += 1
        debias = 1.0 - (weight ** num_acc)
        out[i] = last / debias if debias > 1e-12 else last
    return out


def build_tb_group_figure(
    series: Dict[str, List[Tuple[int, float]]],
    title: str,
    ncols: int = 3,
    smooth_weight: float = TB_SMOOTH_WEIGHT,
    show_raw: bool = True,
):
    """Build a multi-subplot figure for one TB tag group (TB-style EMA smooth)."""
    if not series:
        return None

    tags = list(series.keys())
    is_reward = title.startswith("Episode")
    # Thin crisp lines (raw faint, smooth slightly stronger).
    raw_lw = 0.35 if is_reward else 0.3
    smooth_lw = 1.1 if is_reward else 0.95
    raw_alpha = 0.18 if is_reward else 0.15
    smooth_color = "#1a5fb4" if is_reward else "#3584e4"
    raw_color = "#62a0ea"

    _apply_plot_rc(11 if is_reward else 10)
    fig, axs, nrows, ncols = _make_axes_grid(
        len(tags),
        ncols=ncols,
        cell_size=(4.5, 3.1) if is_reward else (4.2, 2.8),
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, tag in enumerate(tags):
        r, c = divmod(i, ncols)
        ax = axs[r, c]
        steps = np.asarray([p[0] for p in series[tag]], dtype=np.float64)
        values = np.asarray([p[1] for p in series[tag]], dtype=np.float64)
        short = tag.split("/", 1)[-1]
        smoothed = tb_smooth(values, weight=smooth_weight)

        if show_raw and len(values) > 1:
            ax.plot(
                steps, values,
                color=raw_color, linewidth=raw_lw, alpha=raw_alpha,
                label=None, zorder=1,
            )
        ax.plot(
            steps, smoothed,
            color=smooth_color,
            linewidth=smooth_lw,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=f"{short} (smooth={smooth_weight:g})",
            zorder=2,
        )
        ax.set_xlabel("iteration")
        ax.set_ylabel(short)
        ax.set_title(short, fontweight="bold" if is_reward else "normal")
        ax.legend(loc="best", framealpha=0.92, fontsize=8)
        ax.grid(True, alpha=0.35, linewidth=0.8)
        ax.tick_params(axis="both", which="major", length=4, width=1.1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.15 if is_reward else 1.0)

    _hide_unused_axes(axs, len(tags), nrows, ncols)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def save_tb_plots(
    log_dir: str,
    save_dir: Optional[str] = None,
    groups: Sequence[Tuple[str, str]] = TB_PLOT_GROUPS,
    dpi: int = 150,
    max_points: int = 2000,
    smooth_weight: float = TB_SMOOTH_WEIGHT,
) -> List[str]:
    """Export TensorBoard Episode / Loss / Train curves as PNGs into the run dir."""
    save_dir = save_dir or log_dir
    if not log_dir or not os.path.isdir(log_dir):
        print(f"No TB log dir: {log_dir}")
        return []

    series = load_tb_scalars(log_dir, max_points=max_points)
    if not series:
        print(f"No TB scalars in: {log_dir}")
        return []

    saved: List[str] = []
    for prefix, stem in groups:
        group = _filter_tb_group(series, prefix)
        fig = build_tb_group_figure(
            group,
            title=f"{prefix} ({os.path.basename(log_dir)})",
            smooth_weight=smooth_weight,
        )
        if fig is None:
            print(f"No TB tags for group '{prefix}'; skip.")
            continue
        saved.append(save_figure(fig, save_dir, f"{stem}.png", dpi=dpi, layout=False))
    return saved


def export_training_plots(
    runner,
    log_dir: Optional[str],
    save_dir: Optional[str] = None,
    dpi: int = 150,
) -> List[str]:
    """Flush TB writer and dump Episode/Loss/Train PNGs (default: ``log_dir/tb_curves``)."""
    if not log_dir:
        return []
    writer = getattr(runner, "writer", None)
    if writer is not None:
        writer.flush()
    out_dir = save_dir or os.path.join(log_dir, "tb_curves")
    return save_tb_plots(log_dir, save_dir=out_dir, dpi=dpi)


def _plot_process_main(
    log: Dict[str, list],
    dt: float,
    foot_labels: tuple,
    suptitle: Optional[str] = None,
) -> None:
    fig = build_state_figure(log, dt, foot_labels=foot_labels, suptitle=suptitle)
    if fig is None:
        return
    plt.show()


# ---------------------------------------------------------------------------
# Logger facade (play / sim2sim / optional TB export)
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, dt: float):
        self.state_log: Dict[str, list] = defaultdict(list)
        self.rew_log: Dict[str, list] = defaultdict(list)
        self.dt = float(dt)
        self.num_episodes = 0
        self.plot_process: Optional[Process] = None

    def log_state(self, key: str, value: Any) -> None:
        self.state_log[key].append(value)

    def log_states(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            self.log_state(key, value)

    def log_rewards(self, data: Dict[str, Any], num_episodes: int) -> None:
        for key, value in data.items():
            if "rew" in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self) -> None:
        self.state_log.clear()
        self.rew_log.clear()
        self.num_episodes = 0

    def snapshot(self) -> Dict[str, list]:
        """Copy state log so plotting can run without racing the main loop."""
        return {k: list(v) for k, v in self.state_log.items()}

    def num_state_steps(self) -> int:
        if not self.state_log.get("base_vel_x"):
            return 0
        return len(self.state_log["base_vel_x"])

    def compute_tracking_metrics(self) -> Dict[str, float]:
        """RMSE of measured base velocity vs command (used by sim2sim eval)."""
        log = self.state_log
        pairs = (
            ("base_vel_x", "command_x", "vx_rmse"),
            ("base_vel_y", "command_y", "vy_rmse"),
            ("base_vel_yaw", "command_yaw", "yaw_rmse"),
        )
        if not (_has(log, "base_vel_x") and _has(log, "command_x")):
            return {}

        out: Dict[str, float] = {"n_steps": float(len(log["base_vel_x"]))}
        for measured, command, name in pairs:
            if _has(log, measured) and _has(log, command):
                err = _as_1d(log[measured]) - _as_1d(log[command])
                out[name] = float(np.sqrt(np.mean(err ** 2)))
        return out

    def plot_states(
        self,
        save_dir: Optional[str] = None,
        show: bool = False,
        async_show: bool = True,
        foot_labels: Sequence[str] = DEFAULT_FOOT_LABELS,
        filename: Optional[str] = None,
        dpi: int = 150,
        run_name: Optional[str] = None,
        iteration: Optional[int] = None,
        prefix: str = "play",
        suptitle: Optional[str] = None,
    ) -> Optional[str]:
        """Plot logged robot states (shared by play / sim2sim)."""
        log = self.snapshot()
        if not _has(log, "base_vel_x"):
            print("No state log to plot; skip.")
            return None

        if suptitle is None and (run_name is not None or iteration is not None):
            run_txt = run_name if run_name is not None else "?"
            iter_txt = str(iteration) if iteration is not None else "exported"
            suptitle = f"run={run_txt}  |  iter={iter_txt}"

        fig = build_state_figure(log, self.dt, foot_labels=foot_labels, suptitle=suptitle)
        if fig is None:
            return None

        saved_path = None
        if save_dir is not None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if filename is None:
                run_part = (run_name or "unknown").replace(os.sep, "_")
                iter_part = f"iter{iteration}" if iteration is not None else "exported"
                filename = f"{prefix}_{run_part}_{iter_part}_{stamp}.png"
            saved_path = save_figure(fig, save_dir, filename, dpi=dpi, layout=False)
            fig = None

        if show:
            labels = tuple(foot_labels)
            if async_show:
                if self.plot_process is not None and self.plot_process.is_alive():
                    self.plot_process.terminate()
                self.plot_process = Process(
                    target=_plot_process_main,
                    args=(log, self.dt, labels, suptitle),
                )
                self.plot_process.start()
            else:
                if fig is None:
                    fig = build_state_figure(
                        log, self.dt, foot_labels=foot_labels, suptitle=suptitle
                    )
                if fig is not None:
                    plt.show()

        elif fig is not None:
            plt.close(fig)

        return saved_path

    def plot_tb_curves(
        self,
        log_dir: str,
        save_dir: Optional[str] = None,
        dpi: int = 150,
    ) -> List[str]:
        """Save TB reward / loss / train PNGs."""
        return save_tb_plots(log_dir=log_dir, save_dir=save_dir, dpi=dpi)

    def print_rewards(self) -> None:
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.asarray(values)) / max(self.num_episodes, 1)
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None and self.plot_process.is_alive():
            self.plot_process.terminate()
