"""Go2 stairs MuJoCo sim2sim (45-D obs, no gait phase)."""

import math
import os
import sys
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from utils import (  # noqa: E402
    KeyboardCommander,
    Logger,
    quaternion_to_euler_array,
    resolve_plot_meta,
    update_eval_results,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
FOOT_GEOM_NAMES = ('FL', 'FR', 'RL', 'RR')
FOOT_BODY_NAMES = ('FL_calf', 'FR_calf', 'RL_calf', 'RR_calf')
MCJF_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources', 'robots', 'go2', 'MCJF')


# ---------------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------------
def get_obs(data, model):
    """Extract observation quantities from MuJoCo state."""
    q = data.qpos[7:19].astype(np.double)
    dq = data.qvel[6:18].astype(np.double)
    quat = data.qpos[3:7].astype(np.double)[[1, 2, 3, 0]]
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
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
    """Vertical world-frame contact forces on FL/FR/RL/RR foot geoms."""
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
            normal = np.array(con.frame[0:3], dtype=np.double)
            tangent1 = np.array(con.frame[3:6], dtype=np.double)
            tangent2 = np.array(con.frame[6:9], dtype=np.double)
            R_c2w = np.column_stack((normal, tangent1, tangent2))
            forces_z[fi] += (R_c2w @ c_force[:3])[2]
    return forces_z


def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    """PD torque from position targets (relative to default pose)."""
    return (target_q + cfg.robot_config.default_dof_pos - q) * kp + (target_dq - dq) * kd


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def load_policy_checked(path, expected_obs_dim):
    policy = torch.jit.load(path)
    try:
        first = next(policy.parameters())
        in_features = int(first.shape[1]) if first.ndim == 2 else None
        if in_features is not None and in_features != expected_obs_dim:
            raise RuntimeError(
                f"Policy expects obs dim {in_features}, but cfg builds "
                f"{expected_obs_dim}. Check --policy / experiment."
            )
    except StopIteration:
        pass
    return policy


def build_obs_stairs(count_lowlevel, cfg, q, dq, omega, eu_ang, action, vx_cmd, vy_cmd, yaw_cmd):
    """go2_stairs: cmd + imu + dof + action  (45), no gait phase."""
    obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
    obs[0, 0] = vx_cmd * cfg.normalization.obs_scales.lin_vel
    obs[0, 1] = vy_cmd * cfg.normalization.obs_scales.lin_vel
    obs[0, 2] = yaw_cmd * cfg.normalization.obs_scales.ang_vel
    obs[0, 3:6] = omega * cfg.normalization.obs_scales.ang_vel
    obs[0, 6:9] = eu_ang * cfg.normalization.obs_scales.quat
    obs[0, 9:21] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
    obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
    obs[0, 33:45] = action
    return obs


def run_mujoco(policy, cfg, *, commander=None, plot_dir=None, run_dir=None, plot_run_name=None, plot_iteration=None):
    commander = commander or KeyboardCommander()
    if plot_dir is None:
        raise ValueError('plot_dir is required')

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    data.qpos[-cfg.env.num_actions:] = cfg.robot_config.default_dof_pos
    mujoco.mj_forward(model, data)

    target_q = np.zeros(cfg.env.num_actions, dtype=np.double)
    action = np.zeros(cfg.env.num_actions, dtype=np.double)
    hist_obs = deque(
        np.zeros([1, cfg.env.num_single_obs], dtype=np.double)
        for _ in range(cfg.env.frame_stack)
    )

    count_lowlevel = 1
    logger = Logger(cfg.sim_config.dt)
    stop_state_log = 4000
    max_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    commander.print_help()

    plot_kwargs = dict(
        save_dir=plot_dir,
        show=False,
        foot_labels=FOOT_GEOM_NAMES,
        run_name=plot_run_name,
        iteration=plot_iteration,
        prefix='sim2sim',
    )

    with mujoco.viewer.launch_passive(
        model, data, key_callback=commander.on_key, show_left_ui=True, show_right_ui=True,
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

            if count_lowlevel % cfg.sim_config.decimation == 0:
                eu_ang = quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi
                obs = build_obs_stairs(
                    count_lowlevel, cfg, q, dq, omega, eu_ang, action,
                    vx_cmd, vy_cmd, yaw_cmd,
                )
                obs = np.clip(
                    obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations
                )
                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(cfg.env.frame_stack):
                    s = i * cfg.env.num_single_obs
                    policy_input[0, s:s + cfg.env.num_single_obs] = hist_obs[i][0, :]

                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                target_q = action * cfg.control.action_scale

            target_dq = np.zeros(cfg.env.num_actions, dtype=np.double)
            if step_i < 100:
                tau = pd_control(
                    np.zeros(cfg.env.num_actions), q, cfg.robot_config.kps,
                    target_dq, dq, cfg.robot_config.kds, cfg,
                )
            else:
                tau = pd_control(
                    target_q, q, cfg.robot_config.kps,
                    target_dq, dq, cfg.robot_config.kds, cfg,
                )
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl[:] = tau
            applied_tau = data.actuator_force

            mujoco.mj_step(model, data)
            viewer.sync()
            count_lowlevel += 1

            contact_z = get_foot_contact_forces_z(model, data)
            idx = 5
            dof_pos_target = target_q + cfg.robot_config.default_dof_pos
            if step_i < stop_state_log:
                logger.log_states({
                    'base_vel_x': v[0], 'command_x': vx_cmd,
                    'base_vel_y': v[1], 'command_y': vy_cmd,
                    'base_vel_z': v[2],
                    'base_vel_yaw': omega[2], 'command_yaw': yaw_cmd,
                    'dof_pos_target': dof_pos_target[idx],
                    'dof_pos': q[idx], 'dof_vel': dq[idx],
                    'dof_torque': applied_tau[idx], 'cmd_dof_torque': tau[idx],
                    'contact_forces_z': contact_z.copy(),
                    **{f'dof_pos_target[{i}]': dof_pos_target[i].item() for i in range(12)},
                    **{f'dof_pos[{i}]': q[i].item() for i in range(12)},
                    **{f'dof_torque[{i}]': applied_tau[i].item() for i in range(12)},
                    **{f'dof_vel[{i}]': dq[i].item() for i in range(12)},
                })
            elif step_i == stop_state_log:
                plot_path = logger.plot_states(**plot_kwargs)
                if run_dir and plot_path:
                    update_eval_results(run_dir, plot_path, logger.compute_tracking_metrics())
                plots_saved = True

            step_i += 1
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

        if (not plots_saved) and logger.num_state_steps() > 0:
            plot_path = logger.plot_states(**plot_kwargs)
            if run_dir and plot_path:
                update_eval_results(run_dir, plot_path, logger.compute_tracking_metrics())


def main():
    experiment = 'go2_stairs'
    run = None                    # e.g. '2026-08-06_09-39-51_stairs' or None
    policy = None                 # e.g. '/path/to/policy_1.pt' or None -> default export
    # Scene: scene_0 | scene_stairs_L0..L9 | scene_1 | scene_2
    # See MCJF/STAIRS_DIFFICULTY.md (L1 ≈ 6.8 cm). Prefer L0–L3 for current policies.
    mujoco_model_path = os.path.join(MCJF_DIR, 'scene_stairs_L1', 'scene.xml')
    # --------------------------------------------

    load_model = policy or os.path.join(
        LEGGED_GYM_ROOT_DIR, 'logs', experiment, '0_exported', 'policies', 'policy_1.pt'
    )
    run_dir = None
    plot_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', experiment, '0_exported', 'sim2sim')
    if run:
        run_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', experiment, run)
        plot_dir = os.path.join(run_dir, 'sim2sim')
        os.makedirs(run_dir, exist_ok=True)

    class Sim2simCfg:
        class env:
            frame_stack = 1
            num_single_obs = 45
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
            mujoco_model_path = os.path.join(MCJF_DIR, 'scene_stairs_L1', 'scene.xml')
            sim_duration = 120.0
            dt = 0.005
            decimation = 4

        class robot_config:
            kps = np.array([20] * 12, dtype=np.double)
            kds = np.array([0.5] * 12, dtype=np.double)
            tau_limit = 45 * np.ones(12, dtype=np.double)
            default_dof_pos = np.array(
                [0.0, 0.75, -1.5] * 4, dtype=np.double,
            )

    print(f"Loading policy: {load_model}")
    print(f"MuJoCo model: {Sim2simCfg.sim_config.mujoco_model_path}")
    print(f"Plot dir: {plot_dir}")
    if run_dir:
        print(f"Linked run dir: {run_dir}")

    policy_jit = load_policy_checked(load_model, Sim2simCfg.env.num_observations)
    plot_run_name, plot_iteration = resolve_plot_meta(experiment, run, load_model)
    print(f"Plot tag: run={plot_run_name}  iter={plot_iteration if plot_iteration is not None else 'exported'}")
    run_mujoco(
        policy_jit, Sim2simCfg(), commander=KeyboardCommander(),
        plot_dir=plot_dir, run_dir=run_dir,
        plot_run_name=plot_run_name, plot_iteration=plot_iteration,
    )


if __name__ == '__main__':
    main()
