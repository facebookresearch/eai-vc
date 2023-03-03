#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


class ImpedanceController:
    """
    Impedance controller for TriFinger robot

    args:
        kinematics: CustomPinocchioUtils() class for kinematics
        max_x_err: maximum magnitude of position error (for velocity clipping)
        kp, kv: joint space gains
        Nf: number of fingers in hand
    """

    def __init__(
        self,
        kinematics,
        max_x_err=0.01,
        kp=[300] * 9,
        kv=[5] * 9,
        Nf=3,
    ):
        self.kp = kp
        self.kv = kv
        self.max_x_err = max_x_err
        self.kinematics = kinematics

        self.Nf = Nf  # Number of finger in hand
        self.Nq = self.Nf * 3  # Number of joints in hand (3 per finger)

        self._dx_mag_avg = np.ones(
            self.Nf
        )  # Init linear velocity mag average for moving average per finger

        # For observation
        self.x_des = np.ones(9) * np.nan
        self.dx_des = np.ones(9) * np.nan
        self.x_cur = np.ones(9) * np.nan
        self.dx_cur = np.ones(9) * np.nan
        self.torque = np.ones(9) * np.nan

    def get_command_torque(self, x_des, dx_des, q_cur, dq_cur, f_des=None):
        """
        Compute joint torques to move finger all fingers to x_des, dx_des

        args:
            x_des: Desired fingertip position, world frame (NOT CONTROLLING ORIENTATION RIGHT NOW) [9,]
            dx_des: Desired fingertip lin vel, world frame (NOT CONTROLLING ORIENTATION RIGHT NOW) [9,]
            q_cur: Current joint angles of all fingers [Nq,]
            dq_cur: Current joint velocities [Nq,]
            f_des: fingertip forces in world frame [9,]

        return:
            torque: joint torques [9,]
        """

        Kp = np.diag(self.kp)
        Kv = np.diag(self.kv)

        # Linear Jacobian for hand [9, Nq] (stacked finger linear Jacobians [3, Nq])
        J_lin = self.kinematics.get_hand_lin_jacobian(q_cur)
        g = self.kinematics.get_hand_g(q_cur, J_lin)  # Joint space gravity vector

        x_des = np.expand_dims(x_des, 1)
        # Compute current fingertip position
        x_cur = np.array(self.kinematics.forward_kinematics(q_cur)).reshape(
            (self.Nq, 1)
        )
        delta_x = np.array(x_des) - np.array(x_cur)
        # print("Current x: {}".format(x_cur))
        # print("Desired x: {}".format(x_des))
        delta_x_mags = np.linalg.norm(delta_x.reshape((self.Nf, 3)), axis=1)
        # print("Delta: {}".format(delta_x_mags))

        # Cap delta_x magnitude to self.max_x_err
        if self.max_x_err is not None:
            for f_id in range(3):
                delta_x_i = delta_x[f_id * 3 : f_id * 3 + 3, :]
                if np.linalg.norm(delta_x_i) > self.max_x_err:
                    unit = delta_x_i / np.linalg.norm(delta_x_i)
                    delta_x[f_id * 3 : f_id * 3 + 3, :] = unit * self.max_x_err

        # Get current fingertip velocity
        dx_cur = J_lin @ np.expand_dims(np.array(dq_cur), 1)

        delta_dx = np.expand_dims(np.array(dx_des), 1) - np.array(dx_cur)

        F_star = Kv @ delta_dx + Kp @ delta_x

        dx_mags = np.linalg.norm(dx_cur.reshape((self.Nf, 3)), axis=1)
        # print("dx mag: ", dx_mags)

        torque = np.squeeze(J_lin.T @ F_star) + g

        # TODO Feed-forward term for desired forces
        if f_des is not None:
            torque += J_lin.T @ f_des

        # For observation
        self.x_des = np.squeeze(x_des)
        self.dx_des = np.squeeze(dx_des)
        self.x_cur = np.squeeze(x_cur)
        self.dx_cur = np.squeeze(dx_cur)
        self.torque = np.squeeze(torque)

        return torque

    def is_avg_dx_converged(self, q_cur, dq_cur, epsilon=0.001):
        """
        Return: True if linear velocity magnitude moving average of each finger  has converged to < epsilon
        """

        J_lin = self.kinematics.get_hand_lin_jacobian(
            q_cur
        )  # Linear Jacobian for hand [9, Nq]

        # Get current fingertip velocity
        dx_cur = J_lin @ np.expand_dims(np.array(dq_cur), 1)

        all_fingers_converged = True

        for f_id in range(self.Nf):
            dx_cur_i = dx_cur[f_id * 3 : f_id * 3 + 3]
            self._dx_mag_avg[f_id] = (
                0.5 * np.linalg.norm(dx_cur_i) + 0.5 * self._dx_mag_avg[f_id]
            )
            all_fingers_converged = (
                all_fingers_converged and self._dx_mag_avg[f_id] < epsilon
            )

        return all_fingers_converged

    def get_observation(self):
        """Create and return observation"""

        obs = {
            "ft_pos_des": self.x_des,
            "ft_vel_des": self.dx_des,
            "ft_pos_cur": self.x_cur,
            "ft_vel_cur": self.dx_cur,
            "kp": self.kp,
            "kv": self.kv,
            "max_x_err": self.max_x_err,
            "torque": self.torque,
        }

        return obs
