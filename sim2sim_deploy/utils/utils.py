"""Shared helpers: keyboard, math, logger loader, eval-results JSON."""

import importlib.util
import json
import os
import re
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
    run_name = run or experiment
    iteration = None
    if not policy_path:
        return run_name, iteration

    base = os.path.basename(policy_path)
    m = re.search(r"_(\d+)\.pt$", base)
    if m:
        iteration = int(m.group(1))
        if run is None:
            stem = base[: m.start()]
            if stem:
                run_name = stem
    else:
        m = re.search(r"model_(\d+)\.pt$", base)
        if m:
            iteration = int(m.group(1))

    parts = os.path.normpath(policy_path).split(os.sep)
    if "logs" in parts:
        i = parts.index("logs")
        if len(parts) > i + 2 and run is None:
            candidate = parts[i + 2]
            if candidate and candidate not in ("0_exported", "exported"):
                run_name = candidate
    return run_name, iteration


_SKIP_LOG_DIRS = {"0_exported", "exported"}
_JIT_NAME_RE = re.compile(r"^(.+)_(\d+)\.pt$")


def experiment_logs_dir(experiment):
    return os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment)


def run_dir_path(experiment, run):
    return os.path.join(experiment_logs_dir(experiment), run)


def run_artifact_dir(experiment, run, kind):
    """logs/<experiment>/<run>/{policies,sim2play,sim2sim,sim2real}."""
    path = os.path.join(run_dir_path(experiment, run), kind)
    os.makedirs(path, exist_ok=True)
    return path


def _iter_jit_files(policies_dir, run=None, iteration=None):
    if not os.path.isdir(policies_dir):
        return
    for name in os.listdir(policies_dir):
        m = _JIT_NAME_RE.match(name)
        if not m or m.group(1) == "policy":
            continue
        stem, it = m.group(1), int(m.group(2))
        if run is not None and stem != run:
            continue
        if iteration is not None and it != int(iteration):
            continue
        yield stem, it, os.path.join(policies_dir, name)


def resolve_exported_jit(experiment, run, iteration=None):
    """
    logs/<experiment>/<run>/policies/{run}_{iter}.pt

    run=None -> latest JIT across runs (skip leftover policy_1.pt).
    iteration=None -> highest iter among matches.
    """
    exp_root = experiment_logs_dir(experiment)
    cands = []

    if run:
        if iteration is not None:
            exact = os.path.join(exp_root, run, "policies", f"{run}_{int(iteration)}.pt")
            if os.path.isfile(exact):
                return exact
        cands.extend(_iter_jit_files(os.path.join(exp_root, run, "policies"), run, iteration))
    elif os.path.isdir(exp_root):
        for name in os.listdir(exp_root):
            if name in _SKIP_LOG_DIRS:
                continue
            cands.extend(_iter_jit_files(os.path.join(exp_root, name, "policies"), name, iteration))

    cands.extend(_iter_jit_files(os.path.join(exp_root, "0_exported", "policies"), run, iteration))

    if not cands:
        hint = f"{run}_<iter>.pt" if run else "{RUNS}_{iter}.pt"
        raise FileNotFoundError(
            f"No JIT named {hint} under {exp_root}/<run>/policies/. "
            "Run play.py with EXPORT_POLICY=True first."
        )
    cands.sort()
    return cands[-1][2]


def resolve_sim2sim_policy(experiment, run, iteration=None, policy=None):
    """Return (jit_path, run_dir, plot_dir) for MuJoCo sim2sim."""
    if policy:
        load_model = policy
    else:
        load_model = resolve_exported_jit(experiment, run, iteration)

    if not run:
        inferred, _ = resolve_plot_meta(experiment, None, load_model)
        if inferred and inferred != experiment:
            run = inferred

    if not run:
        raise ValueError("set run= (training RUNS) or policy= named {RUNS}_{iter}.pt")

    run_dir = run_dir_path(experiment, run)
    plot_dir = run_artifact_dir(experiment, run, "sim2sim")
    os.makedirs(run_dir, exist_ok=True)
    return load_model, run_dir, plot_dir


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
