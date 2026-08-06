"""Shared helpers: keyboard, math, logger loader, eval-results JSON."""

import importlib.util
import json
import os
from datetime import datetime

import numpy as np

# Repo root (sim2sim_deploy/utils/utils.py -> repo).
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def load_logger():
    """Load Logger from file without executing legged_gym.utils.__init__."""
    logger_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'utils', 'logger.py')
    spec = importlib.util.spec_from_file_location('sim2sim_logger', logger_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Logger


Logger = load_logger()


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def quaternion_to_euler_array(quat):
    """Convert quaternion [x, y, z, w] to roll/pitch/yaw."""
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    t2 = np.clip(+2.0 * (w * y - z * x), -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z])


# ---------------------------------------------------------------------------
# Eval results
# ---------------------------------------------------------------------------

def update_eval_results(run_dir, plot_path, metrics, status='pass'):
    """Merge sim2sim results into run_dir/eval_results.json."""
    path = os.path.join(run_dir, 'eval_results.json')
    data = {}
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    sim2sim = data.get('sim2sim', {})
    plots = list(sim2sim.get('plots') or [])
    rel = os.path.relpath(plot_path, run_dir) if plot_path else None
    if rel and rel not in plots:
        plots.append(rel)
    sim2sim.update({
        'status': status,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'plots': plots,
        'metrics': {**(sim2sim.get('metrics') or {}), **(metrics or {})},
    })
    data['sim2sim'] = sim2sim
    data.setdefault('summary', '')
    data.setdefault('findings', [])
    data.setdefault('sim2real', {
        'status': 'pending',
        'date': '',
        'result': '',
        'problems': '',
        'solutions': '',
        'notes': '',
        'videos': [],
        'images': [],
        'metrics': {},
    })
    for key in ('result', 'problems', 'solutions', 'notes'):
        sim2sim.setdefault(key, '')
    data['sim2sim'] = sim2sim
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f'Updated eval results: {path}')


# ---------------------------------------------------------------------------
# Plot naming helpers
# ---------------------------------------------------------------------------

def resolve_plot_meta(experiment, run, policy_path):
    """Derive (run_name, iteration) for plot filenames from policy path / settings."""
    import re

    run_name = run or experiment
    iteration = None
    if not policy_path:
        return run_name, iteration

    base = os.path.basename(policy_path)
    m = re.search(r"model_(\d+)\.pt$", base)
    if m:
        iteration = int(m.group(1))

    parts = os.path.normpath(policy_path).split(os.sep)
    if "logs" in parts:
        i = parts.index("logs")
        # logs / <experiment> / <run|0_exported> / ...
        if len(parts) > i + 2 and run is None:
            candidate = parts[i + 2]
            if candidate and candidate != "0_exported":
                run_name = candidate
    return run_name, iteration

class KeyboardCommander:
    """Velocity command interface: keys 8/2/4/6/7/9 only."""

    HELP = (
        "Controls (focus MuJoCo window):\n"
        "  8 / 2 : vx+ / vx-\n"
        "  4 / 6 : vy+ / vy-\n"
        "  7 / 9 : yaw+ / yaw-"
    )

    def __init__(
        self,
        vx=0.5,
        vy=0.0,
        yaw=0.0,
        vx_max=1.5,
        vy_max=1.0,
        yaw_max=3.0,
        lin_step=0.3,
        yaw_step=0.5,
        verbose=True,
    ):
        self.vx = float(vx)
        self.vy = float(vy)
        self.yaw = float(yaw)
        self.vx_max = float(vx_max)
        self.vy_max = float(vy_max)
        self.yaw_max = float(yaw_max)
        self.lin_step = float(lin_step)
        self.yaw_step = float(yaw_step)
        self.verbose = verbose
        self._char_map = {
            '8': self._inc_vx,
            '2': self._dec_vx,
            '4': self._inc_vy,
            '6': self._dec_vy,
            '7': self._inc_yaw,
            '9': self._dec_yaw,
        }

    @property
    def command(self):
        return self.vx, self.vy, self.yaw

    def clip(self):
        self.vx = float(np.clip(self.vx, -self.vx_max, self.vx_max))
        self.vy = float(np.clip(self.vy, -self.vy_max, self.vy_max))
        self.yaw = float(np.clip(self.yaw, -self.yaw_max, self.yaw_max))

    def _inc_vx(self):
        self.vx += self.lin_step

    def _dec_vx(self):
        self.vx -= self.lin_step

    def _inc_vy(self):
        self.vy += self.lin_step

    def _dec_vy(self):
        self.vy -= self.lin_step

    def _inc_yaw(self):
        self.yaw += self.yaw_step

    def _dec_yaw(self):
        self.yaw -= self.yaw_step

    def on_key(self, keycode):
        try:
            ch = chr(keycode)
        except ValueError:
            return
        handler = self._char_map.get(ch)
        if handler is None:
            return
        handler()
        self.clip()
        if self.verbose:
            print(f"Updated velocities: vx={self.vx:.2f}, vy={self.vy:.2f}, dyaw={self.yaw:.2f}")

    def print_help(self):
        print(self.HELP)
        print(f"Initial command: vx={self.vx:.2f}, vy={self.vy:.2f}, dyaw={self.yaw:.2f}")
