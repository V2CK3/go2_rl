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

import os
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FOOT_LABELS = ("FL", "FR", "RL", "RR")


def _has(log: Dict[str, list], key: str) -> bool:
    return bool(log.get(key))

def _as_1d(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)

def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", length=4, width=1)

def build_state_figure( log: Dict[str, list], dt: float, foot_labels: Sequence[str] = DEFAULT_FOOT_LABELS):
    """Build the shared 3x3 state figure used by play and sim2sim."""
    if not log or not _has(log, "base_vel_x"):
        return None

    n = len(log["base_vel_x"])
    t = np.linspace(0.0, n * dt, n)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axs = plt.subplots(3, 3, figsize=(14, 10))

    # Row 0: base velocity tracking
    if _has(log,"base_vel_x"):
        axs[0, 0].plot(t, log["base_vel_x"], label="measured")
    if _has(log, "command_x"):
        axs[0, 0].plot(t, log["command_x"], "--", label="command")
    _style(axs[0, 0], "time [s]", "vx [m/s]", "Base vel X")

    if _has(log, "base_vel_y"):
        axs[0, 1].plot(t, log["base_vel_y"], label="measured")
    if _has(log, "command_y"):
        axs[0, 1].plot(t, log["command_y"], "--", label="command")
    _style(axs[0, 1], "time [s]", "vy [m/s]", "Base vel Y")

    if _has(log, "base_vel_yaw"):
        axs[0, 2].plot(t, log["base_vel_yaw"], label="measured")
    if _has(log, "command_yaw"):
        axs[0, 2].plot(t, log["command_yaw"], "--", label="command")
    _style(axs[0, 2], "time [s]", "yaw [rad/s]", "Base vel Yaw")

    # Row 1: DOF / torque / vz
    if _has(log, "dof_pos"):
        axs[1, 0].plot(t, log["dof_pos"], label="measured")
    if _has(log, "dof_pos_target"):
        axs[1, 0].plot(t, log["dof_pos_target"], "--", label="target")
    _style(axs[1, 0], "time [s]", "pos [rad]", "DOF position")

    if _has(log, "dof_vel"):
        axs[1, 1].plot(t, log["dof_vel"], label="measured")
    if _has(log, "dof_vel_target"):
        axs[1, 1].plot(t, log["dof_vel_target"], "--", label="target")
    _style(axs[1, 1], "time [s]", "vel [rad/s]", "DOF velocity")

    if _has(log, "dof_torque"):
        axs[1, 2].plot(t, log["dof_torque"], label="measured")
    _style(axs[1, 2], "time [s]", "torque [N·m]", "Torque")

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

    if _has(log, "base_vel_z"):
        axs[2, 2].plot(t, log["base_vel_z"], label="measured")
    _style(axs[2, 2], "time [s]", "vz [m/s]", "Base vel Z")

    fig.tight_layout()
    return fig


def _plot_process_main(log: Dict[str, list], dt: float, foot_labels: tuple) -> None:
    fig = build_state_figure(log, dt, foot_labels=foot_labels)
    if fig is None:
        return
    plt.show()


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
        if not _has(log, "base_vel_x") or not _has(log, "command_x"):
            return {}

        def _rmse(a, b) -> float:
            return float(np.sqrt(np.mean((_as_1d(a) - _as_1d(b)) ** 2)))

        out = {
            "vx_rmse": _rmse(log["base_vel_x"], log["command_x"]),
            "n_steps": float(len(log["base_vel_x"])),
        }
        if _has(log, "base_vel_y") and _has(log, "command_y"):
            out["vy_rmse"] = _rmse(log["base_vel_y"], log["command_y"])
        if _has(log, "base_vel_yaw") and _has(log, "command_yaw"):
            out["yaw_rmse"] = _rmse(log["base_vel_yaw"], log["command_yaw"])
        return out

    def plot_states(
        self, save_dir: Optional[str] = None, show: bool = False, async_show: bool = True,
        foot_labels: Sequence[str] = DEFAULT_FOOT_LABELS, filename: Optional[str] = None, dpi: int = 150,) -> Optional[str]:
        """
        Plot logged robot states (shared by play / sim2sim).

        Args:
            save_dir: If set, save a PNG under this directory and return its path.
            show: If True, display the figure (blocks unless async_show=True).
            async_show: Show in a background process so Isaac Gym / MuJoCo keep running.
            foot_labels: Legend labels for contact force curves.
            filename: Optional PNG filename; default result_YYYYmmdd_HHMMSS.png.
            dpi: Save resolution.

        Returns:
            Saved PNG path, or None if nothing was saved.
        """
        log = self.snapshot()
        if not _has(log, "base_vel_x"):
            print("No state log to plot; skip.")
            return None

        saved_path = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig = build_state_figure(log, self.dt, foot_labels=foot_labels)
            if fig is None:
                return None
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = filename or f"result_{stamp}.png"
            saved_path = os.path.join(save_dir, name)
            fig.savefig(saved_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved state plot: {saved_path}")

        if show:
            labels = tuple(foot_labels)
            if async_show:
                if self.plot_process is not None and self.plot_process.is_alive():
                    self.plot_process.terminate()
                self.plot_process = Process(
                    target=_plot_process_main,
                    args=(log, self.dt, labels),
                )
                self.plot_process.start()
            else:
                fig = build_state_figure(log, self.dt, foot_labels=foot_labels)
                if fig is not None:
                    plt.show()

        return saved_path

    def print_rewards(self) -> None:
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.asarray(values)) / max(self.num_episodes, 1)
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None and self.plot_process.is_alive():
            self.plot_process.terminate()
