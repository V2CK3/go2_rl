"""部署日志：终端状态 + CSV + pickle。"""

from __future__ import annotations

import csv
import os
import time
import pickle as pkl

import numpy as np
import torch

JOINT_NAMES = (
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
)
FOOT_NAMES = ("FL", "FR", "RL", "RR")


def _as_1d(val) -> np.ndarray:
    if val is None:
        return np.zeros(0, dtype=np.float64)
    if isinstance(val, torch.Tensor):
        val = val.detach().cpu().numpy()
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    return arr


def _fmt_legs(arr, prec=2) -> str:
    q = _as_1d(arr)
    if q.size < 12:
        return str(q)
    parts = []
    for i, name in enumerate(FOOT_NAMES):
        a, b, c = q[3 * i : 3 * i + 3]
        parts.append(f"{name}[{a:.{prec}f} {b:.{prec}f} {c:.{prec}f}]")
    return " ".join(parts)


class DeployLogger:
    """
    每步写入 CSV；按 print_every 打印终端并写 status.log；结束时再 dump pickle。
    跳过 dict（如 obs_history）以免体积过大。
    """

    def __init__(self, cfg, log_root: str, print_every: int = 25):
        self.cfg = cfg
        self.infos = []
        self.print_every = max(int(print_every), 1)
        self.log_path = self._make_log_path(log_root)
        self.log_dir = os.path.dirname(self.log_path)
        self.csv_path = os.path.join(self.log_dir, "state.csv")
        self.status_path = os.path.join(self.log_dir, "status.log")

        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(self._csv_header())
        self._status_file = open(self.status_path, "w")
        self._n = 0
        self._t_wall0 = time.time()
        self._last_wall = self._t_wall0

        print(f"[deploy] live status every {self.print_every} steps")
        print(f"[deploy] CSV: {self.csv_path}")
        print(f"[deploy] text: {self.status_path}")

    @staticmethod
    def _make_log_path(log_root: str) -> str:
        stamp = time.strftime("%Y/%m_%d/%H_%M_%S")
        for i in range(100):
            path = os.path.join(log_root, f"{stamp}_{i}")
            try:
                os.makedirs(path)
                return os.path.join(path, "log.pkl")
            except FileExistsError:
                continue
        raise RuntimeError(f"cannot create log dir under {log_root}")

    @staticmethod
    def _csv_header():
        cols = [
            "t", "step", "hz",
            "cmd_vx", "cmd_vy", "cmd_yaw",
            "roll", "pitch", "yaw",
            "lin_vx", "lin_vy", "lin_vz",
            "ang_wx", "ang_wy", "ang_wz",
            "c_FL", "c_FR", "c_RL", "c_RR",
        ]
        for prefix in ("q", "qd", "qdes", "tau", "tau_est", "act"):
            cols.extend(f"{prefix}_{n}" for n in JOINT_NAMES)
        return cols

    def _hz(self) -> float:
        now = time.time()
        dt = now - self._last_wall
        self._last_wall = now
        if self._n == 0 or dt <= 1e-6:
            return 0.0
        return 1.0 / dt

    def log(self, info: dict):
        row = {}
        for key, val in info.items():
            if isinstance(val, dict):
                continue
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            row[key] = val
        self.infos.append(row)

        hz = self._hz()
        self._write_csv(row, hz)
        self._n += 1
        if self._n == 1 or self._n % self.print_every == 0:
            text = self.format_status(row, hz)
            print(text, flush=True)
            self._status_file.write(text + "\n")
            self._status_file.flush()
            self._csv_file.flush()

    def snapshot_agent(self, agent, tag: str = "snapshot"):
        """校准前后打一条当前传感器状态（不写入 CSV 时序）。"""
        rpy = _as_1d(agent.state.get_rpy())
        cmd = _as_1d(agent.commands)
        q = _as_1d(agent.dof_pos)
        contact = _as_1d(agent.contact_state)
        text = (
            f"-------- {tag} --------\n"
            f"cmd   vx={cmd[0]:7.3f}  vy={cmd[1]:7.3f}  yaw={cmd[2]:7.3f}\n"
            f"imu   r={rpy[0]:7.3f}  p={rpy[1]:7.3f}  y={rpy[2]:7.3f}\n"
            f"contact {self._fmt_contact(contact)}\n"
            f"q     {_fmt_legs(q)}\n"
            f"q0    {_fmt_legs(agent.default_dof_pos)}"
        )
        print(text, flush=True)
        self._status_file.write(text + "\n")
        self._status_file.flush()

    @staticmethod
    def _fmt_contact(c) -> str:
        c = _as_1d(c)
        bits = []
        for i, name in enumerate(FOOT_NAMES):
            v = int(c[i] > 0.5) if i < c.size else 0
            bits.append(f"{name}={v}")
        return " ".join(bits)

    def format_status(self, row: dict, hz: float) -> str:
        t = float(row.get("time", self._n * 0.02))
        step = int(row.get("timestep", self._n))
        cmd_xy = _as_1d(row.get("body_linear_vel_cmd"))
        cmd_yaw = _as_1d(row.get("body_angular_vel_cmd"))
        vx = cmd_xy[0] if cmd_xy.size else 0.0
        vy = cmd_xy[1] if cmd_xy.size > 1 else 0.0
        yaw_c = cmd_yaw[0] if cmd_yaw.size else 0.0
        rpy = _as_1d(row.get("rpy"))
        lin = _as_1d(row.get("body_linear_vel"))
        ang = _as_1d(row.get("body_angular_vel"))
        q = _as_1d(row.get("joint_pos"))
        qdes = _as_1d(row.get("joint_pos_target"))
        tau = _as_1d(row.get("torques"))
        tau_est = _as_1d(row.get("tau_est"))
        contact = _as_1d(row.get("contact_state"))
        tau_src = tau_est if tau_est.size == 12 else tau
        tau_rms = float(np.sqrt(np.mean(np.square(tau_src)))) if tau_src.size else 0.0
        tau_max = float(np.max(np.abs(tau_src))) if tau_src.size else 0.0
        lin_s = " ".join(f"{v:6.3f}" for v in (lin if lin.size >= 3 else np.zeros(3)))
        ang_s = " ".join(f"{v:6.3f}" for v in (ang if ang.size >= 3 else np.zeros(3)))
        rpy_s = rpy if rpy.size >= 3 else np.zeros(3)
        return (
            f"-------- t={t:7.2f}s  step={step:<6d}  {hz:5.1f}Hz --------\n"
            f"cmd   vx={vx:7.3f}  vy={vy:7.3f}  yaw={yaw_c:7.3f}\n"
            f"imu   r={rpy_s[0]:7.3f}  p={rpy_s[1]:7.3f}  y={rpy_s[2]:7.3f}   "
            f"v=[{lin_s}]  w=[{ang_s}]\n"
            f"contact {self._fmt_contact(contact)}   tau_rms={tau_rms:5.1f}  tau_max={tau_max:5.1f}\n"
            f"q     {_fmt_legs(q)}\n"
            f"qdes  {_fmt_legs(qdes)}"
        )

    def _write_csv(self, row: dict, hz: float):
        t = float(row.get("time", self._n * 0.02))
        step = int(row.get("timestep", self._n))
        cmd_xy = _as_1d(row.get("body_linear_vel_cmd"))
        cmd_yaw = _as_1d(row.get("body_angular_vel_cmd"))
        rpy = _as_1d(row.get("rpy"))
        lin = _as_1d(row.get("body_linear_vel"))
        ang = _as_1d(row.get("body_angular_vel"))
        contact = _as_1d(row.get("contact_state"))

        def pad(arr, n):
            a = _as_1d(arr)
            if a.size >= n:
                return a[:n].tolist()
            return a.tolist() + [0.0] * (n - a.size)

        rec = [
            f"{t:.4f}", step, f"{hz:.2f}",
            *pad(np.array([
                cmd_xy[0] if cmd_xy.size else 0.0,
                cmd_xy[1] if cmd_xy.size > 1 else 0.0,
                cmd_yaw[0] if cmd_yaw.size else 0.0,
            ]), 3),
            *pad(rpy, 3),
            *pad(lin, 3),
            *pad(ang, 3),
            *pad(contact, 4),
            *pad(row.get("joint_pos"), 12),
            *pad(row.get("joint_vel"), 12),
            *pad(row.get("joint_pos_target"), 12),
            *pad(row.get("torques"), 12),
            *pad(row.get("tau_est"), 12),
            *pad(row.get("action"), 12),
        ]
        self._csv.writerow(rec)

    def save(self):
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass
        try:
            self._status_file.close()
        except Exception:
            pass
        with open(self.log_path, "wb") as f:
            pkl.dump({"cfg": self.cfg, "infos": self.infos}, f)
        print(
            f"Saved log! timesteps={len(self.infos)}; "
            f"pkl={self.log_path}  csv={self.csv_path}  txt={self.status_path}"
        )
