"""Go2 MuJoCo sim2sim deployment using the official mujoco.viewer GUI."""

import os
import math
import time
import importlib.util
from datetime import datetime
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch

# Repo root (avoid importing legged_gym package, which pulls isaacgym).
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SIM2SIM_PLOT_DIR = os.path.join(
    LEGGED_GYM_ROOT_DIR, 'logs', 'go2_base', '0_exported', 'sim2sim'
)
FOOT_GEOM_NAMES = ('FL', 'FR', 'RL', 'RR')
FOOT_BODY_NAMES = ('FL_calf', 'FR_calf', 'RL_calf', 'RR_calf')


def _load_logger():
    """Load Logger from file without executing legged_gym.utils.__init__."""
    logger_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'utils', 'logger.py')
    spec = importlib.util.spec_from_file_location('sim2sim_logger', logger_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Logger


Logger = _load_logger()


class KeyboardCommander:
    """Velocity command interface driven by mujoco.viewer key events."""

    # GLFW keycodes used by mujoco.viewer key_callback
    KEY_LEFT = 263
    KEY_RIGHT = 264
    KEY_UP = 265
    KEY_DOWN = 266
    KEY_BACKSPACE = 259
    KEY_KP_2 = 322
    KEY_KP_4 = 324
    KEY_KP_5 = 325
    KEY_KP_6 = 326
    KEY_KP_7 = 327
    KEY_KP_8 = 328
    KEY_KP_9 = 329

    HELP = (
        "Controls (focus MuJoCo window):\n"
        "  arrows / 8 2 4 6 : vx+/vx-/vy+/vy-\n"
        "  7 9 or [ ]       : yaw+/-\n"
        "  Backspace / 0    : stop (zero cmd)"
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

        # Logical action name -> handler
        self._actions = {
            'vx+': self._inc_vx,
            'vx-': self._dec_vx,
            'vy+': self._inc_vy,
            'vy-': self._dec_vy,
            'yaw+': self._inc_yaw,
            'yaw-': self._dec_yaw,
            'stop': self.stop,
        }

        # Character / named keys -> action
        self._char_map = {
            '8': 'vx+', '2': 'vx-', '4': 'vy+', '6': 'vy-',
            '7': 'yaw+', '9': 'yaw-',
            '[': 'yaw+', ']': 'yaw-',
            '-': 'yaw+', '=': 'yaw-',
            '5': 'stop', '0': 'stop',
            'UP': 'vx+', 'DOWN': 'vx-', 'LEFT': 'vy+', 'RIGHT': 'vy-',
        }

        # GLFW special / keypad keys -> action
        self._keycode_map = {
            self.KEY_UP: 'vx+',
            self.KEY_DOWN: 'vx-',
            self.KEY_LEFT: 'vy+',
            self.KEY_RIGHT: 'vy-',
            self.KEY_BACKSPACE: 'stop',
            self.KEY_KP_8: 'vx+',
            self.KEY_KP_2: 'vx-',
            self.KEY_KP_4: 'vy+',
            self.KEY_KP_6: 'vy-',
            self.KEY_KP_7: 'yaw+',
            self.KEY_KP_9: 'yaw-',
            self.KEY_KP_5: 'stop',
        }

    @property
    def command(self):
        """Return (vx, vy, yaw)."""
        return self.vx, self.vy, self.yaw

    def clip(self):
        self.vx = float(np.clip(self.vx, -self.vx_max, self.vx_max))
        self.vy = float(np.clip(self.vy, -self.vy_max, self.vy_max))
        self.yaw = float(np.clip(self.yaw, -self.yaw_max, self.yaw_max))

    def stop(self):
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0

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

    def apply(self, action_name):
        handler = self._actions.get(action_name)
        if handler is None:
            return False
        handler()
        self.clip()
        if self.verbose:
            print(f"Updated velocities: vx={self.vx:.2f}, vy={self.vy:.2f}, dyaw={self.yaw:.2f}")
        return True

    def on_key(self, keycode):
        """mujoco.viewer key_callback entrypoint."""
        action = self._keycode_map.get(keycode)
        if action is not None:
            self.apply(action)
            return

        try:
            ch = chr(keycode)
        except ValueError:
            return
        action = self._char_map.get(ch)
        if action is not None:
            self.apply(action)

    def print_help(self):
        print(self.HELP)
        print(f"Initial command: vx={self.vx:.2f}, vy={self.vy:.2f}, dyaw={self.yaw:.2f}")


def quaternion_to_euler_array(quat):
    """Convert quaternion [x, y, z, w] to roll/pitch/yaw."""
    x, y, z, w = quat

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return np.array([roll_x, pitch_y, yaw_z])


def get_obs(data, model):
    """Extract observation quantities from MuJoCo state."""
    q = data.qpos[7:19].astype(np.double)
    dq = data.qvel[6:18].astype(np.double)
    # MuJoCo freejoint quat is [w, x, y, z] -> scipy [x, y, z, w]
    quat = data.qpos[3:7].astype(np.double)[[1, 2, 3, 0]]
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    # Match training: base angular velocity in body frame
    omega = r.apply(data.qvel[3:6], inverse=True).astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    base_pos = data.qpos[0:3].astype(np.double)

    foot_positions = []
    for name in FOOT_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        foot_positions.append(
            data.xpos[body_id][2].copy().astype(np.double) if body_id >= 0 else 0.0
        )

    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions)


def get_foot_contact_forces_z(model, data):
    """Vertical world-frame contact forces on FL/FR/RL/RR foot geoms.

    NOTE: data.cfrc_ext is a COM wrench and is often ~0 / not useful for
    per-foot contact plots. Use mj_contactForce on named foot geoms instead.
    """
    forces_z = np.zeros(len(FOOT_GEOM_NAMES), dtype=np.double)
    geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in FOOT_GEOM_NAMES
    ]

    for i in range(data.ncon):
        con = data.contact[i]
        for fi, gid in enumerate(geom_ids):
            if gid < 0:
                continue
            if con.geom1 != gid and con.geom2 != gid:
                continue
            c_force = np.zeros(6, dtype=np.double)
            mujoco.mj_contactForce(model, data, i, c_force)
            # Contact frame: [normal, tangent1, tangent2] stored in con.frame
            normal = np.array(con.frame[0:3], dtype=np.double)
            tangent1 = np.array(con.frame[3:6], dtype=np.double)
            tangent2 = np.array(con.frame[6:9], dtype=np.double)
            R_c2w = np.column_stack((normal, tangent1, tangent2))
            f_world = R_c2w @ c_force[:3]
            forces_z[fi] += f_world[2]
    return forces_z


def _update_eval_results(run_dir, plot_path, metrics, status='pass'):
    """Merge sim2sim results into run_dir/eval_results.json."""
    import json
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


def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    """PD torque from position targets (relative to default pose)."""
    return (target_q + cfg.robot_config.default_dof_pos - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg, commander=None, plot_dir=None, run_dir=None):
    commander = commander or KeyboardCommander()
    plot_dir = plot_dir or SIM2SIM_PLOT_DIR

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt

    data = mujoco.MjData(model)
    num_actuated_joints = cfg.env.num_actions
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos
    mujoco.mj_forward(model, data)

    target_q = np.zeros(cfg.env.num_actions, dtype=np.double)
    action = np.zeros(cfg.env.num_actions, dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 1
    logger = Logger(cfg.sim_config.dt)
    stop_state_log = 4000
    max_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    commander.print_help()

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=commander.on_key,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45
        viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.3])

        step_i = 0
        plots_saved = False
        while viewer.is_running() and step_i < max_steps:
            step_start = time.time()
            vx_cmd, vy_cmd, yaw_cmd = commander.command

            q, dq, quat, v, omega, gvec, base_pos, foot_positions = get_obs(data, model)

            if step_i > 0 and step_i % int(1.0 / cfg.sim_config.dt) == 0:
                print(
                    f"[t={step_i * cfg.sim_config.dt:5.1f}s] "
                    f"cmd=({vx_cmd:.2f}, {vy_cmd:.2f}, {yaw_cmd:.2f}) "
                    f"base_vel=({v[0]:.2f}, {v[1]:.2f}, {omega[2]:.2f})"
                )

            # Physics dt -> policy dt (e.g. 200Hz -> 50Hz with decimation=4)
            if count_lowlevel % cfg.sim_config.decimation == 0:
                obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                eu_ang = quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi

                phase = (count_lowlevel * cfg.sim_config.dt) % cfg.rewards.cycle_time / cfg.rewards.cycle_time
                obs[0, 0] = math.sin(2 * math.pi * phase)
                obs[0, 1] = math.cos(2 * math.pi * phase)
                obs[0, 2] = vx_cmd * cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = vy_cmd * cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = yaw_cmd * cfg.normalization.obs_scales.ang_vel
                obs[0, 5:8] = omega * cfg.normalization.obs_scales.ang_vel
                obs[0, 8:11] = eu_ang * cfg.normalization.obs_scales.quat
                obs[0, 11:23] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 23:35] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 35:47] = action

                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(cfg.env.frame_stack):
                    policy_input[0, i * cfg.env.num_single_obs:(i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                target_q = action * cfg.control.action_scale

            target_dq = np.zeros(cfg.env.num_actions, dtype=np.double)

            # Soft start: hold default pose for the first ~0.5s
            if step_i < 100:
                tau = pd_control(
                    np.zeros(cfg.env.num_actions), q, cfg.robot_config.kps,
                    target_dq, dq, cfg.robot_config.kds, cfg)
            else:
                tau = pd_control(
                    target_q, q, cfg.robot_config.kps,
                    target_dq, dq, cfg.robot_config.kds, cfg)
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)

            data.ctrl[:] = tau
            applied_tau = data.actuator_force

            mujoco.mj_step(model, data)
            viewer.sync()
            count_lowlevel += 1

            # Contact forces are only valid after mj_step (constraint solve).
            contact_z = get_foot_contact_forces_z(model, data)

            idx = 5
            dof_pos_target = target_q + cfg.robot_config.default_dof_pos

            if step_i < stop_state_log:
                logger.log_states({
                    'base_vel_x': v[0],
                    'command_x': vx_cmd,
                    'base_vel_y': v[1],
                    'command_y': vy_cmd,
                    'base_vel_z': v[2],
                    'base_vel_yaw': omega[2],
                    'command_yaw': yaw_cmd,
                    'dof_pos_target': dof_pos_target[idx],
                    'dof_pos': q[idx],
                    'dof_vel': dq[idx],
                    'dof_torque': applied_tau[idx],
                    'cmd_dof_torque': tau[idx],
                    'contact_forces_z': contact_z.copy(),
                    'dof_pos_target[0]': dof_pos_target[0].item(),
                    'dof_pos_target[1]': dof_pos_target[1].item(),
                    'dof_pos_target[2]': dof_pos_target[2].item(),
                    'dof_pos_target[3]': dof_pos_target[3].item(),
                    'dof_pos_target[4]': dof_pos_target[4].item(),
                    'dof_pos_target[5]': dof_pos_target[5].item(),
                    'dof_pos_target[6]': dof_pos_target[6].item(),
                    'dof_pos_target[7]': dof_pos_target[7].item(),
                    'dof_pos_target[8]': dof_pos_target[8].item(),
                    'dof_pos_target[9]': dof_pos_target[9].item(),
                    'dof_pos_target[10]': dof_pos_target[10].item(),
                    'dof_pos_target[11]': dof_pos_target[11].item(),
                    'dof_pos[0]': q[0].item(),
                    'dof_pos[1]': q[1].item(),
                    'dof_pos[2]': q[2].item(),
                    'dof_pos[3]': q[3].item(),
                    'dof_pos[4]': q[4].item(),
                    'dof_pos[5]': q[5].item(),
                    'dof_pos[6]': q[6].item(),
                    'dof_pos[7]': q[7].item(),
                    'dof_pos[8]': q[8].item(),
                    'dof_pos[9]': q[9].item(),
                    'dof_pos[10]': q[10].item(),
                    'dof_pos[11]': q[11].item(),
                    'dof_torque[0]': applied_tau[0].item(),
                    'dof_torque[1]': applied_tau[1].item(),
                    'dof_torque[2]': applied_tau[2].item(),
                    'dof_torque[3]': applied_tau[3].item(),
                    'dof_torque[4]': applied_tau[4].item(),
                    'dof_torque[5]': applied_tau[5].item(),
                    'dof_torque[6]': applied_tau[6].item(),
                    'dof_torque[7]': applied_tau[7].item(),
                    'dof_torque[8]': applied_tau[8].item(),
                    'dof_torque[9]': applied_tau[9].item(),
                    'dof_torque[10]': applied_tau[10].item(),
                    'dof_torque[11]': applied_tau[11].item(),
                    'dof_vel[0]': dq[0].item(),
                    'dof_vel[1]': dq[1].item(),
                    'dof_vel[2]': dq[2].item(),
                    'dof_vel[3]': dq[3].item(),
                    'dof_vel[4]': dq[4].item(),
                    'dof_vel[5]': dq[5].item(),
                    'dof_vel[6]': dq[6].item(),
                    'dof_vel[7]': dq[7].item(),
                    'dof_vel[8]': dq[8].item(),
                    'dof_vel[9]': dq[9].item(),
                    'dof_vel[10]': dq[10].item(),
                    'dof_vel[11]': dq[11].item(),
                })
            elif step_i == stop_state_log:
                plot_path = logger.plot_states(
                    save_dir=plot_dir, show=False, foot_labels=FOOT_GEOM_NAMES
                )
                if run_dir and plot_path:
                    _update_eval_results(run_dir, plot_path, logger.compute_tracking_metrics())
                plots_saved = True

            step_i += 1
            # Keep wall-clock roughly real-time
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

        # Save on early exit (window closed before stop_state_log)
        if (not plots_saved) and logger.num_state_steps() > 0:
            plot_path = logger.plot_states(
                save_dir=plot_dir, show=False, foot_labels=FOOT_GEOM_NAMES
            )
            if run_dir and plot_path:
                _update_eval_results(run_dir, plot_path, logger.compute_tracking_metrics())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Go2 MuJoCo sim2sim')
    parser.add_argument('--experiment', default='go2_base', help='Experiment name under logs/')
    parser.add_argument('--run', default=None, help='Training run dir name; links plots into eval_results.json')
    parser.add_argument('--policy', default=None, help='Path to TorchScript policy (.pt)')
    parser.add_argument('--terrain', action='store_true', help='Use terrain MuJoCo scene')
    args = parser.parse_args()

    use_terrain = args.terrain
    load_model = args.policy or os.path.join(
        LEGGED_GYM_ROOT_DIR, 'logs', args.experiment, '0_exported', 'policies', 'policy_1.pt'
    )

    run_dir = None
    plot_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.experiment, '0_exported', 'sim2sim')
    if args.run:
        run_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.experiment, args.run)
        plot_dir = os.path.join(run_dir, 'sim2sim')
        os.makedirs(run_dir, exist_ok=True)

    class Sim2simCfg:
        """Self-contained cfg (mirrors Go2BaseCfg fields used by sim2sim)."""

        class env:
            frame_stack = 10
            num_single_obs = 47
            num_observations = frame_stack * num_single_obs
            num_actions = 12

        class control:
            action_scale = 0.25

        class rewards:
            cycle_time = 0.5

        class normalization:
            class obs_scales:
                lin_vel = 2.0
                ang_vel = 0.25
                dof_pos = 1.0
                dof_vel = 0.05
                quat = 1.0
            clip_observations = 100.
            clip_actions = 100.

        class sim_config:
            mujoco_model_path = (
                f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/MCJF/scene_terrain_0703.xml'
                if use_terrain else
                f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/MCJF/scene.xml'
            )
            sim_duration = 120.0
            dt = 0.005
            decimation = 4

        class robot_config:
            # FL / FR / RL / RR  x  hip, thigh, calf  (matches MuJoCo actuator order)
            kps = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], dtype=np.double)
            kds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.double)
            tau_limit = 45 * np.ones(12, dtype=np.double)
            default_dof_pos = np.array(
                [0.0, 0.8, -1.5,
                 0.0, 0.8, -1.5,
                 0.0, 0.8, -1.5,
                 0.0, 0.8, -1.5],
                dtype=np.double,
            )

    print(f"Loading policy: {load_model}")
    print(f"MuJoCo model: {Sim2simCfg.sim_config.mujoco_model_path}")
    print(f"Plot dir: {plot_dir}")
    if run_dir:
        print(f"Linked run dir: {run_dir}")

    policy = torch.jit.load(load_model)
    run_mujoco(policy, Sim2simCfg(), commander=KeyboardCommander(), plot_dir=plot_dir, run_dir=run_dir)