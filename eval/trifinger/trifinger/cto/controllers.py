import pinocchio as pin
import numpy as np


class ImpedanceController:
    def __init__(self, P, D, pin_robot, frame_id):
        # setup control parameters
        self.P = P
        self.D = D
        self.pin_robot = pin_robot
        self.frame_id = frame_id

    def compute_torque(self, q, dq, x_ref, dx_ref, f_ref):
        oriented_jacobian = self.compute_oriented_jacobian(q)
        Jov = oriented_jacobian[:3]

        x = self.pin_robot.data.oMf[self.frame_id].translation
        dx = Jov.dot(dq)

        force_des = self.P @ (x_ref - x) + self.D @ (dx_ref - dx) + f_ref
        tau = Jov.T.dot(force_des)

        return tau

    def compute_oriented_jacobian(self, q):
        """
        calculate oriented jacobian of the frame,
        hence the jacobian of an accessory frame that has the same position as the given frame but oriented as the world frame
        """
        # get frame pose
        pose = self.pin_robot.data.oMf[self.frame_id]

        # get oriented jacobian
        body_jocobian = pin.computeFrameJacobian(
            self.pin_robot.model, self.pin_robot.data, q, self.frame_id
        )
        adjoint = np.zeros((6, 6))
        adjoint[:3, :3] = pose.rotation
        adjoint[3:, 3:] = pose.rotation

        return adjoint @ body_jocobian
