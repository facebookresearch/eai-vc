#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import pinocchio

from trifinger_simulation.pinocchio_utils import Kinematics


class CustomPinocchioUtils(Kinematics):
    """
    Consists of kinematic methods for the finger platform.
    """

    m1 = 0.2
    m2 = 0.2
    m3 = 0.01
    ms = [m1, m2, m3]
    tip_m = 0.02
    I1 = np.zeros((3, 3))
    # np.fill_diagonal(I1,[3.533e-4,5.333e-5,3.533e-4])
    np.fill_diagonal(I1, [4.59 - 4, 6.93e-5, 4.59e-4])
    I2 = np.zeros((3, 3))
    # np.fill_diagonal(I2,[3.533e-4,3.533e-4,5.333e-5])
    np.fill_diagonal(I2, [4.41e-4, 4.41e-4, 6.67e-5])
    I3 = np.zeros((3, 3))
    # np.fill_diagonal(I3,[1.667e-5,1.667e-5,6.667e-7])
    np.fill_diagonal(I3, [3.5e-5, 3.5e-5, 1.4e-6])
    Is = [I1, I2, I3]

    def __init__(self, finger_urdf_path, tip_link_names, link_names):
        """
        Initializes the finger model on which control's to be performed.

        Args:
            finger (SimFinger): An instance of the SimFinger class
            link_names: finger link names
        """
        super().__init__(finger_urdf_path, tip_link_names)

        self.link_ids = [
            self.robot_model.getFrameId(link_name) for link_name in link_names
        ]

    def get_hand_lin_jacobian(self, q):
        J = np.zeros((9, 9))

        for f_id in range(3):
            J_i = self.get_tip_link_jacobian(f_id, q)  # [6, 9]
            J[f_id * 3 : f_id * 3 + 3, :] = J_i[:3, :]
        return J

    def get_tip_link_jacobian(self, finger_id, q):
        """
        Get Jacobian for tip link of specified finger
        All other columns are 0
        """
        pinocchio.computeJointJacobians(
            self.robot_model,
            self.data,
            q,
        )
        # pinocchio.framesKinematics(
        #    self.robot_model, self.data, q,
        # )
        pinocchio.framesForwardKinematics(
            self.robot_model,
            self.data,
            q,
        )
        frame_id = self.tip_link_ids[finger_id]
        Ji = pinocchio.getFrameJacobian(
            self.robot_model,
            self.data,
            frame_id,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        # print(self.robot_model.frames[frame_id].placement)
        # print(self.data.oMf[frame_id].rotation)
        return Ji

    def get_any_link_jacobian(self, frame_id, q):
        """
        Get Jacobian for link with frame_id
        """
        pinocchio.computeJointJacobians(
            self.robot_model,
            self.data,
            q,
        )
        pinocchio.framesForwardKinematics(
            self.robot_model,
            self.data,
            q,
        )
        Ji = pinocchio.getFrameJacobian(
            self.robot_model,
            self.data,
            frame_id,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return Ji  # 6x9

    def get_finger_g(self, f_id, q, Jvi):
        """Get joint space gravity vector for single finger finger_id"""

        g = np.zeros(9)
        grav = np.array([0, 0, -9.81])

        # Get g for each link in finger
        for j, l_id in enumerate(self.link_ids[f_id * 3 : f_id * 3 + 3]):
            Jj = self.get_any_link_jacobian(l_id, q)
            Jjv = Jj[:3, :]
            g -= self.ms[j] * Jjv.T @ grav * 0.33

        Jj = self.get_any_link_jacobian(self.tip_link_ids[f_id], q)
        Jjv = Jj[:3, :]
        g -= self.tip_m * Jjv.T @ grav * 0.33

        return g

    def get_hand_g(self, q_cur, J):
        """Get joint space gravity vector for 3-fingered hand"""

        return self.inverse_dyn(q_cur)

        # This doesn't work well, fingers droop slightly when trying to hold position. Depends on accurate model.
        # g = np.zeros(9)

        # for f_id in range(3):
        #    g_i = self.get_finger_g(f_id, q_cur, J)
        #    g += g_i

        # return g

    def inverse_dyn(self, q):
        # q = pinocchio.neutral(self.robot_model)
        v = pinocchio.utils.zero(self.robot_model.nv)
        a = pinocchio.utils.zero(self.robot_model.nv)

        tau = pinocchio.rnea(self.robot_model, self.data, q, v, a)
        return tau
