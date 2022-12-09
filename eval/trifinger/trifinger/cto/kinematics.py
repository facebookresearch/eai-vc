import numpy as np
import pinocchio as pin


def forward_kinematics(pin_robot, q, frame_id):
    """
    forward kinematics
    """
    pin.framesForwardKinematics(pin_robot.model, pin_robot.data, q)
    # get pose
    return pin.updateFramePlacement(pin_robot.model, pin_robot.data, frame_id)


def oriented_jacobian(pin_robot, q, frame_id):
    """
    Jacobian of the frame located at the end-effector but oriented as the base
    """
    # get pose
    pose = pin.updateFramePlacement(pin_robot.model, pin_robot.data, frame_id)

    # get oriented jacobian
    body_jacobian = pin.computeFrameJacobian(
        pin_robot.model, pin_robot.data, q, frame_id
    )
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = pose.rotation
    Ad[3:, 3:] = pose.rotation

    return Ad @ body_jacobian


def pinv(J, eps=1e-3):
    JJT = J @ J.T
    return J.T @ np.linalg.inv((JJT + eps * np.eye(*JJT.shape)))


def inverse_kinematics_3d(
    pin_robot,
    p_des,
    frame_id,
    q_init=None,
    max_it=100,
    alpha=0.1,
    threshold=0.02,
    q_null=None,
):

    if q_init is None:
        q = np.zeros(pin_robot.nv)
    else:
        q = q_init

    I = np.eye(len(q))

    for i in range(max_it):
        p = forward_kinematics(pin_robot, q, frame_id).translation
        J = oriented_jacobian(pin_robot, q, frame_id)
        Jlin = J[:3, :]
        Jlin_inv = pinv(Jlin)
        if q_null is None:
            q += alpha * Jlin_inv @ (p_des - p)
        else:
            null_proj = (I - Jlin_inv @ Jlin) @ (q_null - q)
            q += alpha * (Jlin_inv @ (p_des - p) + null_proj)
        err = np.linalg.norm(p_des - p)
        if err <= threshold:
            return q, True

    return q, False
