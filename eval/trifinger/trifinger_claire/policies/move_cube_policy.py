import os
import sys
import numpy as np
import enum

import trifinger_simulation.finger_types_data
import trifinger_simulation.pinocchio_utils

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, ".."))

from control.impedance_controller import ImpedanceController
from control.custom_pinocchio_utils import CustomPinocchioUtils
import control.cube_utils as c_utils


class Mode(enum.Enum):
    INIT = enum.auto()
    GRASP = enum.auto()
    MOVE_CUBE = enum.auto()
    DONE = enum.auto()


class MoveCubePolicy:
    """

    Move cube

    - Compute contact points on cube given cube init pose
    - Get ft_pos in world frame
    - Move fingers to cube
    - Compute ft goal positions give cube goal pose, contact poinst
    """

    def __init__(self, action_space, platform, time_step=0.001):
        self.action_space = action_space
        self.time_step = time_step

        # TODO hardcoded
        robot_properties_path = (
            "../trifinger_simulation/trifinger_simulation/robot_properties_fingers"
        )

        urdf_file = trifinger_simulation.finger_types_data.get_finger_urdf(
            "trifingerpro"
        )

        finger_urdf_path = os.path.join(robot_properties_path, "urdf", urdf_file)

        # initial joint positions (lifting the fingers up)
        self.joint_positions = np.array(
            [-0.08, 1.15, -1.5] * 3
        )  # "down and out" position

        # set platform (robot)
        self.platform = platform

        self.Nf = 3  # Number of fingers
        self.Nq = self.Nf * 3  # Number of joints in hand

        # class with kinematics functions
        self.kinematics = CustomPinocchioUtils(
            self.platform.simfinger.finger_urdf_path,
            self.platform.simfinger.tip_link_names,
            self.platform.simfinger.link_names,
        )

        self.controller = ImpedanceController(self.kinematics)

        # mode and trajectory initializations
        self.mode = Mode.INIT
        self.prev_mode = None
        self.traj_counter = 0

        # Initial ft pos and vel trajectories
        self.init_x = self.get_ft_pos(self.joint_positions)  # initial fingertip pos
        self.ft_pos_traj = np.expand_dims(self.init_x, 0)
        self.ft_vel_traj = np.zeros((1, 9))

        self.done = False

        self.t = 0

        self.ft_pos_targets_per_mode = []  # List of fingertip position targets per mode

    def reset(self):
        # initial joint positions (lifting the fingers up)
        self.joint_positions = self.joint_positions

        # mode and trajectory initializations
        self.mode = Mode.INIT
        self.prev_mode = None
        self.traj_counter = 0

        # Initial ft pos and vel trajectories
        self.init_x = self.get_ft_pos(self.joint_positions)  # initial fingertip pos
        self.ft_pos_traj = np.expand_dims(self.init_x, 0)
        self.ft_vel_traj = np.zeros((1, 9))

        self.t = 0

        self.done = False

        self.ft_pos_targets_per_mode = []  # List of fingertip position targets per mode

    def state_machine(self, observation, t):
        """Define mode transition logic"""

        self.prev_mode = self.mode

        q_cur = observation["robot_position"]
        dq_cur = observation["robot_velocity"]

        if self.mode == Mode.INIT:
            self.mode = Mode.GRASP

        elif self.mode == Mode.GRASP:
            # if self.controller.is_avg_dx_converged(q_cur, dq_cur):
            if self.traj_counter == len(self.ft_pos_traj) - 1:
                self.mode = Mode.MOVE_CUBE

        elif self.mode == Mode.MOVE_CUBE:
            if self.traj_counter == len(self.ft_pos_traj) - 1:
                self.mode = Mode.DONE

        elif self.mode == Mode.DONE:
            self.done = True

        else:
            raise ValueError(f"{self.mode} is an invalide Mode")

    def get_ft_des(self, observation):
        """Get fingertip desired pos based on current self.mode"""

        ft_pos_des = self.ft_pos_traj[self.traj_counter, :]
        ft_vel_des = self.ft_vel_traj[self.traj_counter, :]

        if self.traj_counter < len(self.ft_pos_traj) - 1:
            self.traj_counter += 1

        return ft_pos_des, ft_vel_des

    def _to_obj_dict(self, ort, pos):
        return {"orientation": ort, "position": pos}

    def set_ft_traj(self, observation):
        """Given self.mode, set self.ft_pos_traj and self.ft_vel_traj; reset self.traj_counter"""

        obj_orientation = observation["object_orientation"]
        obj_position = observation["object_position"]

        goal_pose = {
            "position": observation["desired_goal"][:3],
            "orientation": obj_orientation,
        }

        q_cur = observation["robot_position"]
        ft_pos_cur = self.get_ft_pos(q_cur)

        if self.mode == Mode.INIT:
            self.ft_pos_traj = np.expand_dims(self.init_x, 0)
            self.ft_vel_traj = np.zeros((1, 9))

        elif self.mode == Mode.GRASP:
            obj_pose = self._to_obj_dict(obj_orientation, obj_position)
            self.cp_params = c_utils.get_cp_params(obj_pose)
            ft_pos = c_utils.get_cp_pos_wf_from_cp_params(self.cp_params, obj_pose)
            ft_pos = np.concatenate(ft_pos)

            self.ft_pos_traj, self.ft_vel_traj = c_utils.lin_interp_pos_two_points(
                ft_pos_cur, ft_pos, 1.5, time_step=self.time_step
            )

            self.ft_pos_targets_per_mode.append(ft_pos)

        elif self.mode == Mode.MOVE_CUBE:
            # Get object trajectory
            o_traj, do_traj = c_utils.lin_interp_pos_two_points(
                obj_position, goal_pose["position"], 2.5, time_step=self.time_step
            )

            # Get ft pos trajectory from object trajectory
            ft_pos_traj = np.zeros((o_traj.shape[0], 9))
            for i in range(o_traj.shape[0]):
                o_des = o_traj[i, :]
                o_pose = {"position": o_des[:3], "orientation": obj_orientation}
                ft_pos = np.concatenate(
                    c_utils.get_cp_pos_wf_from_cp_params(self.cp_params, o_pose)
                )
                ft_pos_traj[i, :] = ft_pos

            self.ft_vel_traj = np.tile(do_traj, (1, 3))
            self.ft_pos_traj = ft_pos_traj

            self.ft_pos_targets_per_mode.append(ft_pos_traj[-1, :])
        else:
            raise ValueError(f"{self.mode} is an invalide Mode")

        # Reset traj counter
        self.traj_counter = 0

    def predict(self, observation):
        """
        Returns torques to command to robot
        1. Call state_machine() to determine mode
        2. If entering new mode, set new finger tip traj
        3. Get current waypoints for finger tips
        4. Get torques from controller
        """
        t = self.t

        # print(observation["desired_goal"]["position"])

        # 1. Call state_machine() to determine mode
        self.state_machine(observation, t)

        # 2. If entering new mode, set new finger tip traj
        if self.prev_mode != self.mode and self.mode != Mode.DONE:
            self.set_ft_traj(observation)

        # 3. Get current waypoints for finger tips
        x_des, dx_des = self.get_ft_des(observation)

        # 4. Get torques from controller
        q_cur = observation["robot_position"]
        dq_cur = observation["robot_velocity"]
        torque = self.controller.get_command_torque(x_des, dx_des, q_cur, dq_cur)

        self.t += 1

        return self.clip_to_space(torque)

    def clip_to_space(self, action):
        """Clip action to action space"""

        return np.clip(action, self.action_space.low, self.action_space.high)

    def get_ft_pos(self, q):
        """Get fingertip positions given current joint configuration q"""

        ft_pos = np.array(self.kinematics.forward_kinematics(q)).reshape(self.Nq)
        return ft_pos

    def get_observation(self):

        obs = {
            "policy": {
                "controller": self.controller.get_observation(),
                "ft_pos_targets_per_mode": np.array(self.ft_pos_targets_per_mode),
                "t": self.t,
            }
        }

        return obs
