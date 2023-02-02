import numpy as np

from control.impedance_controller import ImpedanceController
from control.custom_pinocchio_utils import CustomPinocchioUtils

from cto.trajectory import generate_ee_motion_trifinger


class OpenLoopPolicy:
    def __init__(
        self,
        action_space,
        finger,
        time_step=0.001,
        x_des=None,
        f_des=None,
    ):
        self.action_space = action_space
        self.time_step = time_step
        self.finger = finger
        self.nq = finger.kinematics.robot_model.nq
        self.nv = finger.kinematics.robot_model.nv

        if x_des is not None:
            self.x_des = x_des
        if f_des is not None:
            self.f_des = f_des

        # class with kinematics functions
        self.kinematics = CustomPinocchioUtils(
            self.finger.finger_urdf_path,
            self.finger.tip_link_names,
            self.finger.link_names,
        )
        self.reset()

    def reset(self):
        self.controller = ImpedanceController(self.kinematics, kp=[200] * 9, kv=[5] * 9)
        self.done = False
        self.t = 0

    def predict(self, observation):
        t = self.t
        self.t += 1
        x_des, dx_des = self.x_des[t], np.zeros(9)
        f_des = self.f_des[t]
        q_cur = observation["robot_position"]
        dq_cur = observation["robot_velocity"]
        torque = self.controller.get_command_torque(x_des, dx_des, q_cur, dq_cur, f_des)

        return self.clip_to_space(torque)

    def clip_to_space(self, action):
        """Clip action to action space"""

        return np.clip(action, self.action_space.low, self.action_space.high)

    def get_ft_pos(self, q):
        """Get fingertip positions given current joint configuration q"""

        ft_pos = np.array(self.kinematics.forward_kinematics(q)).reshape(self.nq)
        return ft_pos

    def get_observation(self):
        obs = {
            "policy": {
                "controller": self.controller.get_observation(),
                "t": self.t,
            }
        }

        return obs

    def set_trajs(
        self, observation, state, sol, params, dt_sim=1e-3, dt_plan=0.1, warmup_time=1.0
    ):
        rest_locations, trajs, forces = generate_ee_motion_trifinger(
            state, sol, dt_sim, dt_plan, params
        )
        x_des_all = []
        f_des_all = []

        # interpolate a trajectories (duration=warmup_time) to move the fingers
        # from the current position to the first planned position
        q_cur = observation["robot_position"]
        start = np.reshape(self.finger.kinematics.forward_kinematics(q_cur), self.nq)
        end = np.hstack([traj[0] for traj in trajs[0]])
        x_des, _, _ = spline_interpolation(
            start, end, int(warmup_time / dt_sim), dt_sim
        )
        x_des_all.append(x_des)
        f_des_all.append(np.zeros_like(x_des))

        # combine all finger trajectories to one trajectory
        for i in range(len(trajs)):
            traj0, traj1, traj2 = trajs[i]
            force0, force1, force2 = forces[i]
            N0 = len(traj0)
            N1 = len(traj1)
            N2 = len(traj2)
            N = np.max((N0, N1, N2))
            x_des = np.zeros((N, 9))
            f_des = np.zeros((N, 9))
            for n in range(N):
                n0 = n if n < N0 else -1
                n1 = n if n < N1 else -1
                n2 = n if n < N2 else -1

                x_des[n, 0:3] = traj0[n0]
                x_des[n, 3:6] = traj1[n1]
                x_des[n, 6:9] = traj2[n2]

                f_des[n, 0:3] = force0[n0]
                f_des[n, 3:6] = force1[n1]
                f_des[n, 6:9] = force2[n2]

            x_des_all.append(x_des)
            f_des_all.append(f_des)

        x_des_all = np.vstack(x_des_all)
        f_des_all = np.vstack(f_des_all)

        self.x_des = x_des_all
        self.f_des = f_des_all


def spline_interpolation(start, end, horizon, dt):
    duration = (horizon - 1) * dt
    diff = end - start

    a5 = 6 / (duration**5)
    a4 = -15 / (duration**4)
    a3 = 10 / (duration**3)

    q = start * np.ones((horizon, len(diff)))
    dq = np.zeros((horizon, len(diff)))
    ddq = np.zeros((horizon, len(diff)))

    for n in range(horizon):
        t = n * dt

        s = a3 * t**3 + a4 * t**4 + a5 * t**5
        ds = 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
        dds = 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3

        q[n] += s * diff
        dq[n] = diff * ds
        ddq[n] = diff * dds

    return q, dq, ddq
