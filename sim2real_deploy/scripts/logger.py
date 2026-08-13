"""部署日志：逐步 infos → pickle。"""

from __future__ import annotations

import os
import time
import pickle as pkl

import torch


class DeployLogger:
    """单机部署日志。跳过 dict（如 obs_history）以免体积过大。"""

    def __init__(self, cfg, log_root: str):
        self.cfg = cfg
        self.infos = []
        self.log_path = self._make_log_path(log_root)

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

    def log(self, info: dict):
        row = {}
        for key, val in info.items():
            if isinstance(val, dict):
                continue
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            row[key] = val
        self.infos.append(row)

    def save(self):
        with open(self.log_path, "wb") as f:
            pkl.dump({"cfg": self.cfg, "infos": self.infos}, f)
        print(f"Saved log! timesteps={len(self.infos)}; path={self.log_path}")
