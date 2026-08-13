"""
真机状态缓存 + 遥控速度指令。

订阅 LCM：
  "leg_control_data"      -> 关节 q / qd / tau_est
  "state_estimator_data"  -> RPY / 足力
  "rc_command"            -> 摇杆与按键

get_command()：左摇杆 Y/X、右摇杆 X -> (vx, vy, yaw_rate)
"""

import math
import select
import threading
import time

import numpy as np

from sim2real_deploy.lcm_types.leg_control_data_lcmt import leg_control_data_lcmt
from sim2real_deploy.lcm_types.rc_command_lcmt import rc_command_lcmt
from sim2real_deploy.lcm_types.state_estimator_lcmt import state_estimator_lcmt


def get_rotation_matrix_from_rpy(rpy):
    r, p, y = rpy
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r), -math.sin(r)],
                    [0, math.sin(r), math.cos(r)]])
    R_y = np.array([[math.cos(p), 0, math.sin(p)],
                    [0, 1, 0],
                    [-math.sin(p), 0, math.cos(p)]])
    R_z = np.array([[math.cos(y), -math.sin(y), 0],
                    [math.sin(y), math.cos(y), 0],
                    [0, 0, 1]])
    return np.dot(R_z, np.dot(R_y, R_x))


class StateBuilder:
    """缓存最新传感器，并把摇杆映射成 (vx, vy, yaw)。"""

    def __init__(
        self,
        lc,
        x_scale=1.0,
        y_scale=0.6,
        yaw_scale=1.0,
        probe_vel_multiplier=1.0,
    ):
        self.joint_idxs = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.contact_idxs = [1, 0, 3, 2]
        self.lc = lc
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.yaw_scale = yaw_scale
        self.probe_vel_multiplier = probe_vel_multiplier

        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.tau_est = np.zeros(12)

        self.world_lin_vel = np.zeros(3)
        self.euler = np.zeros(3)
        self.R = np.eye(3)
        self.buf_idx = 0
        self.smoothing_length = 12
        self.deuler_history = np.zeros((self.smoothing_length, 3))
        self.dt_history = np.zeros((self.smoothing_length, 1))
        self.euler_prev = np.zeros(3)
        self.timuprev = time.time()
        self.body_lin_vel = np.zeros(3)
        self.body_ang_vel = np.zeros(3)
        self.smoothing_ratio = 0.2
        self.contact_state = np.ones(4)

        self.left_stick = [0, 0]
        self.right_stick = [0, 0]
        self.left_upper_switch = 0
        self.left_lower_left_switch = 0
        self.left_lower_right_switch = 0
        self.right_upper_switch = 0
        self.right_lower_left_switch = 0
        self.right_lower_right_switch = 0
        self.left_upper_switch_pressed = 0
        self.left_lower_left_switch_pressed = 0
        self.left_lower_right_switch_pressed = 0
        self.right_upper_switch_pressed = 0
        self.right_lower_left_switch_pressed = 0
        self.right_lower_right_switch_pressed = 0

        self.init_time = time.time()
        self.received_first_legdata = False

        self.lc.subscribe("state_estimator_data", self._imu_cb)
        self.legdata_state_subscription = self.lc.subscribe("leg_control_data", self._legdata_cb)
        self.lc.subscribe("rc_command", self._rc_command_cb)

    def get_command(self, probe=False):
        """摇杆 -> (vx, vy, yaw_rate)。"""
        cmd_x = float(self.left_stick[1]) * self.x_scale
        cmd_y = float(self.left_stick[0]) * self.y_scale
        cmd_yaw = -float(self.right_stick[0]) * self.yaw_scale
        if probe:
            cmd_x *= self.probe_vel_multiplier
            cmd_yaw *= self.probe_vel_multiplier
        return np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float64)

    def get_body_linear_vel(self):
        self.body_lin_vel = np.dot(self.R.T, self.world_lin_vel)
        return self.body_lin_vel

    def get_body_angular_vel(self):
        self.body_ang_vel = self.smoothing_ratio * np.mean(
            self.deuler_history / self.dt_history, axis=0
        ) + (1 - self.smoothing_ratio) * self.body_ang_vel
        return self.body_ang_vel

    def get_contact_state(self):
        return self.contact_state[self.contact_idxs]

    def get_rpy(self):
        return self.euler

    def get_buttons(self):
        return np.array([
            self.left_lower_left_switch,
            self.left_upper_switch,
            self.right_lower_right_switch,
            self.right_upper_switch,
        ])

    def get_dof_pos(self):
        return self.joint_pos[self.joint_idxs]

    def get_dof_vel(self):
        return self.joint_vel[self.joint_idxs]

    def _legdata_cb(self, channel, data):
        if not self.received_first_legdata:
            self.received_first_legdata = True
            print(f"First legdata: {time.time() - self.init_time}")
        msg = leg_control_data_lcmt.decode(data)
        self.joint_pos = np.array(msg.q)
        self.joint_vel = np.array(msg.qd)
        self.tau_est = np.array(msg.tau_est)

    def _imu_cb(self, channel, data):
        msg = state_estimator_lcmt.decode(data)
        self.euler = np.array(msg.rpy)
        self.R = get_rotation_matrix_from_rpy(self.euler)
        self.contact_state = 1.0 * (np.array(msg.contact_estimate) > 200)
        self.deuler_history[self.buf_idx % self.smoothing_length, :] = msg.rpy - self.euler_prev
        self.dt_history[self.buf_idx % self.smoothing_length] = time.time() - self.timuprev
        self.timuprev = time.time()
        self.buf_idx += 1
        self.euler_prev = np.array(msg.rpy)

    def _rc_command_cb(self, channel, data):
        msg = rc_command_lcmt.decode(data)
        self.left_upper_switch_pressed = (
            (msg.left_upper_switch and not self.left_upper_switch)
            or self.left_upper_switch_pressed
        )
        self.left_lower_left_switch_pressed = (
            (msg.left_lower_left_switch and not self.left_lower_left_switch)
            or self.left_lower_left_switch_pressed
        )
        self.left_lower_right_switch_pressed = (
            (msg.left_lower_right_switch and not self.left_lower_right_switch)
            or self.left_lower_right_switch_pressed
        )
        self.right_upper_switch_pressed = (
            (msg.right_upper_switch and not self.right_upper_switch)
            or self.right_upper_switch_pressed
        )
        self.right_lower_left_switch_pressed = (
            (msg.right_lower_left_switch and not self.right_lower_left_switch)
            or self.right_lower_left_switch_pressed
        )
        self.right_lower_right_switch_pressed = (
            (msg.right_lower_right_switch and not self.right_lower_right_switch)
            or self.right_lower_right_switch_pressed
        )
        self.right_stick = msg.right_stick
        self.left_stick = msg.left_stick
        self.left_upper_switch = msg.left_upper_switch
        self.left_lower_left_switch = msg.left_lower_left_switch
        self.left_lower_right_switch = msg.left_lower_right_switch
        self.right_upper_switch = msg.right_upper_switch
        self.right_lower_left_switch = msg.right_lower_left_switch
        self.right_lower_right_switch = msg.right_lower_right_switch

    def poll(self):
        try:
            while True:
                rfds, _, _ = select.select([self.lc.fileno()], [], [], 0.01)
                if rfds:
                    self.lc.handle()
        except KeyboardInterrupt:
            pass

    def spin(self):
        self.run_thread = threading.Thread(target=self.poll, daemon=False)
        self.run_thread.start()

    def close(self):
        self.lc.unsubscribe(self.legdata_state_subscription)


if __name__ == "__main__":
    import lcm

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
    StateBuilder(lc).poll()
