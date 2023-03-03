#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import enum
import typing
import gym
import math
import numpy as np
import pybullet
import torch

import trifinger_simulation
from trifinger_simulation import camera
from trifinger_simulation import visual_objects
from trifinger_simulation import trifingerpro_limits
import trifinger_simulation.tasks.move_cube as task
from trifinger_simulation import sample

from trifinger_vc.trifinger_envs.action import ActionType
from trifinger_vc.control.impedance_controller import ImpedanceController
from trifinger_vc.control.custom_pinocchio_utils import CustomPinocchioUtils
from trifinger_simulation.trifinger_platform import ObjectType

import trifinger_vc.utils.data_utils as d_utils

try:
    import robot_fingers
except ImportError:
    robot_fingers = None

import trifinger_vc.control.cube_utils as c_utils


REACH_EPISODE_LENGTH = 500


class CubeReachEnv(gym.Env):
    """Gym environment for moving cubes with TriFingerPro."""

    def __init__(
        self,
        render_mode: str = "",
        fixed_goal: bool = True,
        visual_observation: bool = False,
        action_type: ActionType = ActionType.TORQUE,
        step_size: int = 100,
        visualization: bool = False,
        enable_cameras: bool = True,
        camera_id: int = 0,
        finger_type: str = "trifingerpro",
        camera_delay_steps: int = 0,
        time_step: float = 0.004,
        randomize_starts: bool = True,
        randomize_all: bool = False,
        sample_radius: float = 0.00,
        max_goal_dist: float = 100,
        object_type: ObjectType = ObjectType.COLORED_CUBE,
        enable_shadows: bool = False,
        camera_view: str = "default",
        arena_color: str = "default",
        random_q_init: bool = False,
        run_rl_policy: bool = True,
        seq_eval: bool = True,
    ):
        """Initialize.

        Args:
            fixed_goal: Default true, if false will sample random goal.
            visual_observation: Default false, if true will output images as observations instead of state of robot fingertips.
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
            no_collisions (bool): If true, turn of collisions between platform and object.
            enable_cameras (bool): If true, enable cameras that capture RGB image
                observations.
            finger_type (str): Finger type ("trifingerpro", "trifingeredu")
            camera_delay_steps (int):  Number of time steps by which camera
                observations are held back after they are generated.  This is
                used to simulate the delay of the camera observation that is
                happening on the real system due to processing (mostly the
                object detection).
            time_step (float): Simulation timestep
            run_rl_policy (bool): If true, don't add extra observation fields used for bc policy
        """
        super().__init__()
        if render_mode == "human":
            visualization = True
        self.visualization = visualization
        self.enable_cameras = enable_cameras
        self.finger_type = finger_type
        self.run_rl_policy = run_rl_policy

        self.time_step = time_step
        self.randomize_starts = randomize_starts
        self.sample_radius = sample_radius
        self.randomize_all = randomize_all
        self.max_goal_dist = max_goal_dist
        self.camera_id = camera_id
        if self.camera_id > 2:
            raise ValueError("Not a valid camera_id, choose value [0,1,2].")

        if self.randomize_all:
            self.randomize_starts = True

        self.max_episode_len = REACH_EPISODE_LENGTH
        if self.randomize_all:
            self.max_episode_len = 1000

        # initialize simulation
        self.q_nominal = np.array([-0.08, 1.15, -1.5] * 3)
        self.random_q_init = random_q_init
        self.initial_robot_position = self.q_nominal

        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=self.visualization,
            enable_cameras=self.enable_cameras,
            finger_type=self.finger_type,
            time_step_s=self.time_step,
            initial_robot_position=self.q_nominal,
            camera_delay_steps=camera_delay_steps,
            object_type=object_type,
            enable_shadows=enable_shadows,
            camera_view=camera_view,
            arena_color=arena_color,
            fix_cube_base=True,
        )

        self.hand_kinematics = HandKinematics(self.platform.simfinger)

        # Make camera for RL training
        if self.run_rl_policy:
            target_positions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            camera_up_vectors = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
            field_of_view = 33
            self.tricamera = camera.TriFingerCameras(
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
                target_positions=target_positions,
                camera_up_vectors=camera_up_vectors,
                field_of_view=field_of_view,
            )
        else:
            self.tricamera = None

        # Basic initialization
        # ====================
        self.visual_observation = visual_observation
        self.action_type = action_type
        self.dense_reward_weights = np.zeros(2)
        self.dense_reward_weights[0] = 100000

        if step_size < 1:
            raise ValueError("step_size cannot be less than 1.")
        self.step_size = step_size

        # will be initialized in reset()
        # self.platform = None

        # Create the action and observation spaces
        # ========================================

        self.robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )

        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        goal_state_space = gym.spaces.Box(
            low=np.ones(9) * -0.7,
            high=np.ones(9) * 0.7,
        )

        self.observation_state_space = gym.spaces.Box(
            low=np.ones(18) * -0.7,
            high=np.ones(18) * 0.7,
        )

        self.action_space = gym.spaces.Box(
            low=np.ones(3) * -2,
            high=np.ones(3) * 2,
        )

        # used for initializing random start positions
        self.action_bounds = {
            "low": trifingerpro_limits.robot_position.low,
            "high": trifingerpro_limits.robot_position.high,
        }

        # actions are dealt with as displacement of fingertips regardless of type
        if self.action_type == ActionType.TORQUE:
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")

        self.ftip_dist_space = gym.spaces.Box(
            low=np.ones(1) * -100,
            high=np.ones(1) * 100,
        )
        self.total_success_space = gym.spaces.Box(
            low=np.ones(1) * -100,
            high=np.ones(1) * 100,
        )

        self.img_size = (270, 270, 3)
        self.image_observation_space = gym.spaces.Box(
            low=np.zeros(self.img_size), high=np.ones(self.img_size) * 255
        )
        self.observation_space = gym.spaces.Dict(
            {
                "observation": self.observation_state_space,
            }
        )

        if self.visual_observation:
            if self.camera_id == -1:
                self.observation_space = gym.spaces.Dict(
                    {
                        "pixels": gym.spaces.Box(
                            low=np.zeros((256, 256, 9)),
                            high=np.ones((256, 256, 9)) * 255,
                        ),
                        "ftip_dist": self.observation_state_space,
                    }
                )
            else:
                self.observation_space = gym.spaces.Dict(
                    {
                        "pixels": self.image_observation_space,
                        "ftip_dist": self.observation_state_space,
                        "scaled_success": self.total_success_space,
                    }
                )

        self.start_pos = None
        # self.goal_marker = trifinger_simulation.visual_objects.CubeMarker(
        #     width=task._CUBE_WIDTH,
        #     position=[0.05, 0.07, 0.0319],
        #     orientation=0,
        #     pybullet_client_id=self.platform.simfinger._pybullet_client_id,
        # )

        self.eval_goal_list = self.get_eval_goal_list()
        self.eval_count = 0
        # goes through hardcoded eval goal values in order rather than randomly choosing
        self.sequential_eval = seq_eval
        self.train_goal_list = self.get_train_goal_list()

        # Finger ids to move
        self.finger_to_move_list = [0]  # TODO hardcoded

    """
        mode=rgb_array returns numpy.ndarray with shape (x, y, 3) of current observation
    """

    def get_train_goal_list(self):
        return [
            np.array([3.05603813, -7.55214019, 3.25]),
            np.array([-3.15326713, 0.35681094, 3.25]),
            np.array([-0.20568451, 7.48419172, 3.25]),
            np.array([-1.80023987, -3.33667845, 3.25]),
            np.array([0.63224735, -0.20621713, 3.25]),
            np.array([2.49144056, -1.52591661, 3.25]),
            np.array([-8.10157516, 3.60477928, 3.25]),
            np.array([-4.75578621, -5.62289382, 3.25]),
            np.array([0.60647659, -2.64716854, 3.25]),
            np.array([-1.11332975, 5.00887828, 3.25]),
            np.array([5.98420496, -4.31522391, 3.25]),
            np.array([-4.18048378, 5.86477577, 3.25]),
            np.array([2.63104316, -0.24772835, 3.25]),
            np.array([-4.98861264, 5.96657986, 3.25]),
            np.array([-2.10679078, -3.15221106, 3.25]),
            np.array([-7.90809522, -4.2657171, 3.25]),
            np.array([-1.3794515, 5.83348671, 3.25]),
            np.array([4.48787389, -2.4191718, 3.25]),
            np.array([-1.36567956, -5.11484226, 3.25]),
            np.array([-2.9759321, 7.29904344, 3.25]),
            np.array([-1.68308814, 0.35553572, 3.25]),
            np.array([8.93032708, 0.30403264, 3.25]),
            np.array([4.41736031, -6.83057901, 3.25]),
            np.array([-3.28454635, 2.72672544, 3.25]),
            np.array([4.51527941, 3.46186233, 3.25]),
            np.array([0.02471094, 6.74989932, 3.25]),
            np.array([-7.25012877, -4.12715448, 3.25]),
            np.array([0.08717153, 6.12825175, 3.25]),
            np.array([0.47511044, -4.20393201, 3.25]),
            np.array([8.20551313, 0.42598918, 3.25]),
            np.array([7.53531281, -3.53960009, 3.25]),
            np.array([1.63535131, -4.59013092, 3.25]),
            np.array([0.65539638, 6.58593092, 3.25]),
            np.array([2.83107544, -2.68763681, 3.25]),
            np.array([2.82826438, -8.44225464, 3.25]),
            np.array([-1.55811306, -3.29802461, 3.25]),
            np.array([8.48321033, 0.93042389, 3.25]),
            np.array([-3.14584343, -4.08948458, 3.25]),
            np.array([-2.80634012, -8.02044702, 3.25]),
            np.array([3.14693547, 8.00778896, 3.25]),
            np.array([-6.57006396, -4.22565421, 3.25]),
            np.array([-2.99551142, -3.63649108, 3.25]),
            np.array([-1.08590006, 6.13535156, 3.25]),
            np.array([-6.13850402, -5.16321051, 3.25]),
            np.array([2.82973147, 4.65223176, 3.25]),
            np.array([2.87652314, -4.5091759, 3.25]),
            np.array([2.89854216, -6.15023629, 3.25]),
            np.array([-0.24121648, 5.12888577, 3.25]),
            np.array([-5.52839414, 2.1008083, 3.25]),
            np.array([6.99050079, 2.24616699, 3.25]),
            np.array([-0.96494484, -3.1828791, 3.25]),
            np.array([-3.10124255, 3.8221943, 3.25]),
            np.array([-2.56092877, -3.03297289, 3.25]),
            np.array([4.50346113, -7.31932264, 3.25]),
            np.array([5.91994241, 4.94647579, 3.25]),
            np.array([-0.48606156, -5.32731048, 3.25]),
            np.array([-0.32667426, -8.66828972, 3.25]),
            np.array([1.07453595, 7.36318464, 3.25]),
            np.array([-3.25205737, 6.89068226, 3.25]),
            np.array([3.26506201, 3.42383366, 3.25]),
            np.array([2.07172391, 2.67414843, 3.25]),
            np.array([0.48822116, -8.55367921, 3.25]),
            np.array([4.83845338, -0.06968285, 3.25]),
            np.array([2.81093887, 7.46827855, 3.25]),
            np.array([0.16453263, 2.7395888, 3.25]),
            np.array([0.72086808, 3.73863384, 3.25]),
            np.array([-2.60081194, -4.16909876, 3.25]),
            np.array([3.839713, -0.29123967, 3.25]),
            np.array([-1.61879305, -4.78198183, 3.25]),
            np.array([-7.55117813, 1.13727678, 3.25]),
            np.array([3.66259269, 6.03049238, 3.25]),
            np.array([-4.33543528, -4.87801221, 3.25]),
            np.array([-1.29923973, -0.15892838, 3.25]),
            np.array([3.68191348, -4.96217322, 3.25]),
            np.array([-3.81746439, 6.50004219, 3.25]),
            np.array([-3.421152, -5.53083725, 3.25]),
            np.array([5.49898056, -2.90976879, 3.25]),
            np.array([-0.38942852, -6.84294041, 3.25]),
            np.array([3.27499388, 3.09205193, 3.25]),
            np.array([1.468062, 8.53217955, 3.25]),
            np.array([-4.66475019, -3.24606976, 3.25]),
            np.array([-4.65764194, 3.18195181, 3.25]),
            np.array([-1.57019021, -6.97081706, 3.25]),
            np.array([7.57547351, 0.02846027, 3.25]),
            np.array([-4.86324653, -1.69117867, 3.25]),
            np.array([0.96394429, 0.18087209, 3.25]),
            np.array([-3.34152739, -5.18181183, 3.25]),
            np.array([-4.18771876, 3.58084266, 3.25]),
            np.array([5.86468526, -5.3484374, 3.25]),
            np.array([1.59870173, 8.36118042, 3.25]),
            np.array([5.89203303, 2.6759065, 3.25]),
            np.array([-0.79057999, 6.58881004, 3.25]),
            np.array([-4.04837897, 2.31781327, 3.25]),
            np.array([3.66880724, -6.76704128, 3.25]),
            np.array([-6.97825733, 3.36637523, 3.25]),
            np.array([5.63888276, 4.1776771, 3.25]),
            np.array([-2.15349959, 5.91943316, 3.25]),
            np.array([-4.85276579, 4.91514082, 3.25]),
            np.array([-7.31107254, -3.19688512, 3.25]),
            np.array([-7.56355014, -2.69394404, 3.25]),
        ]

    def get_eval_goal_list(self):
        return [
            np.array([0.00927656, 3.03888736, 3.25]),
            np.array([1.0535054, 0.54244131, 3.25]),
            np.array([2.97988333, 2.19828506, 3.25]),
            np.array([-0.08625725, -2.66008382, 3.25]),
            np.array([-5.53817563, 1.30016464, 3.25]),
            np.array([-7.34284403, -3.30897914, 3.25]),
            np.array([5.34721599, -7.04574016, 3.25]),
            np.array([1.5701743, 2.77699441, 3.25]),
            np.array([5.51455727, 6.71779349, 3.25]),
            np.array([-0.62604526, 1.95728886, 3.25]),
            np.array([2.18948636, -7.21505172, 3.25]),
            np.array([0.99774909, -8.47347619, 3.25]),
            np.array([8.5452943, 0.08286776, 3.25]),
            np.array([-7.71756237, 3.42348443, 3.25]),
            np.array([3.66341366, 1.91997392, 3.25]),
            np.array([4.89323018, 6.2648753, 3.25]),
            np.array([4.04716893, 3.53093616, 3.25]),
            np.array([8.5513687, 0.39826775, 3.25]),
            np.array([-3.07441005, -3.34725609, 3.25]),
            np.array([-3.42368536, -4.14163919, 3.25]),
            np.array([2.61979674, 5.75253347, 3.25]),
            np.array([0.54666075, -1.66785584, 3.25]),
            np.array([4.90558802, 2.54940494, 3.25]),
            np.array([5.24091262, 6.37654168, 3.25]),
            np.array([3.30044642, 6.45136387, 3.25]),
        ]

    def render(self, mode="human"):
        # TODO implement "human" and "ansi" modes
        if mode == "rgb_array":
            # 0:camera 60, 1:camera180, 2:camera300
            # camera_observation = self.platform.get_camera_observation(self.step_count)
            if self.camera_id == -1:  # return ALL images
                cam_imgs = self.tricamera.get_images()
                camera_observation = np.concatenate(
                    (cam_imgs[0], cam_imgs[1], cam_imgs[2]), axis=2
                )
            else:
                camera_observation = self.tricamera.get_images()[self.camera_id]
                camera_observation = torch.tensor(camera_observation)
            return camera_observation
        elif mode == "eval":
            camera_observation = self.tricamera.get_images()[0]
            # self.hide_marker_from_camera()
            return camera_observation

        else:
            raise NotImplementedError

    def compute_ftip_dist(self, achieved_goal, desired_goal) -> dict:
        d = {}
        for i in range(3):
            k = f"f%s_dist" % i
            x = torch.tensor(
                desired_goal[(3 * i) : (3 * i) + 3]
                - achieved_goal[(3 * i) : (3 * i) + 3]
            )
            d[k] = torch.sum(torch.sqrt(x**2))
        return d

    def scaled_success(self, cur_ft_pos):
        """
        args:
            ftpos (np.array): current fingertip positions [9,]
        """

        scaled_err = d_utils.get_reach_scaled_err(
            self.finger_to_move_list,
            self.start_pos,
            cur_ft_pos,
            self.goal.clone().detach().cpu().numpy(),
            task._CUBE_WIDTH / 2,
        )

        success = 1 - scaled_err
        if success < 0:
            success = 0
        return success

    def compute_reward(
        self,
        achieved_goal,
        desired_goal,
        info: dict,
    ) -> float:
        """Compute the reward for the given achieved and desired goal.

        Args:
            achieved_goal: Current position of the object.
            desired_goal: Goal pose of the object.
            info: An info dictionary containing a field "time_index" which
                contains the time index of the achieved_goal.

        Returns:
            The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal. Note that the following should always hold true::

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'],
                    ob['desired_goal'],
                    info,
                )
        """
        current_dist_to_goal = (achieved_goal[0:3] - self.goal).norm()
        reward = -750 * (current_dist_to_goal)
        return reward

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """

        if self.run_rl_policy:
            action = torch.tensor(action) / 50.0
            action = torch.clip(action, -0.02, 0.02)

        if not self.action_space.contains(np.array(action, dtype=np.float32)):
            print(action)
            raise ValueError("Given action is not contained in the action space.")

        # TODO add option for more than one finger?
        # TODO check if tensor
        if self.run_rl_policy:
            three_finger_action = torch.zeros(9, dtype=torch.float32)
            three_finger_action[0:3] = action.clone().detach()
        else:
            three_finger_action = torch.zeros(9)
            three_finger_action[0:3] = torch.FloatTensor(action).detach()

        num_steps = self.step_size
        # ensure episode length is not exceeded due to step_size
        step_count_after = self.step_count + num_steps
        if step_count_after > self.max_episode_len:
            excess = step_count_after - self.max_episode_len
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for i in range(num_steps):
            # Get current robot state
            robot_obs = self.platform.get_robot_observation(self.step_count)
            joint_position = robot_obs.position
            joint_velocity = robot_obs.velocity

            self.step_count += 1
            if self.step_count > self.max_episode_len:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # Update desired position and velocity
            x_des_i = self.x_des_plan + (i + 1) * (three_finger_action / num_steps)
            dx_des_i = three_finger_action / (self.step_size * self.time_step)

            # Compute torque with impedance controller
            torque = self.hand_kinematics.get_torque(
                x_des_i,
                dx_des_i,
                joint_position,
                joint_velocity,
            )
            torque = np.clip(
                torque, self.robot_torque_space.low, self.robot_torque_space.high
            )

            # Send action to robot
            robot_action = self._gym_action_to_robot_action(torque)
            t = self.platform.append_desired_action(robot_action)

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            self.info["time_index"] = t + 1

            # Alternatively use the observation of step t.  This is the
            # observation from the moment before action_t is applied, i.e. the
            # result of that action is not yet visible in this observation.
            #
            # When using this observation, the resulting cumulative reward
            # should match exactly the one computed during replay (with the
            # above it will differ slightly).
            # self.info["time_index"] = t

        observation = self._create_observation(self.info["time_index"])
        # Update plan with action
        self.x_des_plan += three_finger_action

        # Compute reward
        reward = 0
        if self.run_rl_policy:
            reward += self.compute_reward(
                observation["ftip_dist"][:9],
                self.goal,
                self.info,
            )

        is_done = self.step_count >= self.max_episode_len

        return observation, reward, is_done, self.info

    def rand_step(self, tensordict):
        action = (np.random.rand(3) * 2) - np.ones(3)
        print("rand_step")
        print(action)
        return self.step(action)

    def state_dict(self):
        return {}

    def fake_tensordict(self):
        # TODO is this still used?
        observation = self._create_observation(self.info["time_index"])
        if self.visual_observation:
            observation = {
                "pixels": self.render("rgb_array"),
                "ftip_dist": observation,
                "scaled_success": self.scaled_success(
                    observation,
                ),
            }
        return observation

    def _is_above_table(self, coord):
        return (
            True
            if (coord[0][-1] > 0 and coord[1][-1] > 0 and coord[2][-1] > 0)
            else False
        )

    def choose_start_pos(self):
        while True:
            initial_robot_position = (
                trifingerpro_limits.robot_position.default
                + ((np.random.rand(9) * 2) - 1) * self.sample_radius
            )
            eepos = self.platform.simfinger.kinematics.forward_kinematics(
                initial_robot_position
            )
            if self._is_above_table(eepos):
                return initial_robot_position

    def choose_goal(self):
        return self.goal

    def choose_goal_from_demos(self, eval=False):
        if eval:
            self.eval_count += 1
            if self.eval_count == len(self.eval_goal_list):
                self.eval_count = 0

            goal_pos_list = self.eval_goal_list
        else:
            goal_pos_list = self.train_goal_list

        if self.sequential_eval and eval:
            idx = self.eval_count
        else:
            idx = np.random.randint(0, len(goal_pos_list))
        return torch.FloatTensor(goal_pos_list[idx] / 100.0)

    def sample_init_robot_position(self):
        q0_range = [-0.15, 0.15]
        q1_range = [0.8, 1.15]
        q2_range = [-1.35, -1.65]

        i = 0
        q_new = np.array([q0_range[i], q1_range[i], q2_range[i]] * 3)

        q_new = np.zeros(9)
        for i in range(3):
            q0 = np.random.uniform(q0_range[0], q0_range[1])
            q1 = np.random.uniform(q1_range[0], q1_range[1])
            q2 = np.random.uniform(q2_range[0], q2_range[1])

            q_new[3 * i] = q0
            q_new[3 * i + 1] = q1
            q_new[3 * i + 2] = q2

        return q_new

    def reset(self, init_pose_dict=None, init_robot_position=None, eval_mode=False):
        """Reset the environment."""

        # initialize cube at the centre
        if init_pose_dict is None:
            initial_object_pose = task.sample_goal(difficulty=-1)
        else:
            initial_object_pose = task.Pose.from_dict(init_pose_dict)

        if self.run_rl_policy:
            # train/test split use same pos. as those used in demonstrations
            initial_object_pose.position = self.choose_goal_from_demos(eval_mode)

        if init_robot_position is None:
            if self.random_q_init:
                init_robot_position = self.sample_init_robot_position()
            else:
                init_robot_position = self.initial_robot_position

        self.platform.reset(
            initial_object_pose=initial_object_pose,
            initial_robot_position=init_robot_position,
        )

        # Set pybullet GUI params
        self._set_sim_params()

        self.start_pos = self.hand_kinematics.get_ft_pos(init_robot_position)
        self.goal = torch.tensor(initial_object_pose.position)  # Cube is fixed
        self.info = {"time_index": -1, "goal": self.goal}
        self.step_count = 0

        new_obs = self._create_observation(0)

        # Reset state for policy execution
        self.x_des_plan = torch.FloatTensor(self.start_pos.copy())

        return new_obs

    def seed(self, seed=None):
        """Sets the seed for this env’s random number generator.

        .. note::

           Spaces need to be seeded separately.  E.g. if you want to sample
           actions directly from the action space using
           ``env.action_space.sample()`` you can set a seed there using
           ``env.action_space.seed()``.

        Returns:
            List of seeds used by this environment.  This environment only uses
            a single seed, so the list contains only one element.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        task.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def _create_observation(self, t):
        robot_observation = self.platform.get_robot_observation(t)
        ftip_pos = self.hand_kinematics.get_ft_pos(robot_observation.position)
        scaled_success = self.scaled_success(ftip_pos)
        if self.run_rl_policy:
            goal_pos = torch.clone(torch.FloatTensor(ftip_pos))
            goal_pos[0:3] = self.goal
            observation_vec = torch.cat(
                (torch.FloatTensor(ftip_pos), torch.FloatTensor(goal_pos))
            )
            if self.visual_observation:
                observation = {
                    "pixels": self.render("rgb_array"),
                    "ftip_dist": observation_vec,
                    "scaled_success": scaled_success,
                }
            else:
                # robot_observation = self.platform.simfinger.get_observation(t)
                # camera_observation = self.platform.get_camera_observation(t)
                goal_pos = torch.clone(torch.FloatTensor(ftip_pos))
                goal_pos[0:3] = self.goal
                observation = torch.cat(
                    (torch.FloatTensor(ftip_pos), torch.FloatTensor(goal_pos))
                )
        else:
            camera_observation = self.platform.get_camera_observation(t)
            object_observation = camera_observation.filtered_object_pose

            # Get cube vertices
            obj_pose = {
                "position": object_observation.position,
                "orientation": object_observation.orientation,
            }
            observation = {
                "t": t,
            }

            # Compute distances of each finger to object
            ftpos_dist_to_obj = d_utils.get_per_finger_ftpos_err(
                np.expand_dims(ftip_pos, 0),
                np.tile(object_observation.position, (1, 3)),
            )

            # Add new observation fields
            v_wf_dict = c_utils.get_vertices_wf(obj_pose)
            observation["robot_position"] = robot_observation.position
            observation["object_position"] = object_observation.position
            observation["object_orientation"] = object_observation.orientation
            observation["object_vertices"] = v_wf_dict
            observation["desired_goal"] = self.goal.clone().detach().cpu().numpy()
            observation["scaled_success"] = scaled_success
            observation["achieved_goal_position_error"] = ftpos_dist_to_obj
            observation["ft_pos_cur"] = ftip_pos
            # Save camera observation images
            if self.visual_observation:
                camera_observation_dict = {
                    "camera60": {
                        "image": camera_observation.cameras[0].image,
                        "timestamp": camera_observation.cameras[0].timestamp,
                    },
                    "camera180": {
                        "image": camera_observation.cameras[1].image,
                        "timestamp": camera_observation.cameras[1].timestamp,
                    },
                    "camera300": {
                        "image": camera_observation.cameras[2].image,
                        "timestamp": camera_observation.cameras[2].timestamp,
                    },
                }

                observation["camera_observation"] = camera_observation_dict
            observation["policy"] = {
                "controller": self.hand_kinematics.get_observation()
            }

        return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def _set_sim_params(self):
        """Set pybullet GUI params"""

        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_GUI, 0
        )  # Turn off debug camera visuals
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_SHADOWS, 0
        )  # Turn off debug camera visuals


# kinematics wrapper
class HandKinematics:
    def __init__(self, simfinger):
        self.Nf = 3  # Number of fingers
        self.Nq = self.Nf * 3  # Number of joints in hand
        # class with kinematics functions
        self.kinematics = CustomPinocchioUtils(
            simfinger.finger_urdf_path,
            simfinger.tip_link_names,
            simfinger.link_names,
        )

        self.controller = ImpedanceController(self.kinematics)

    def get_ft_pos(self, q):
        """Get fingertip positions given current joint configuration q"""

        ft_pos = np.array(self.kinematics.forward_kinematics(q)).reshape(self.Nq)
        return ft_pos

    def get_torque(self, x_des, dx_des, q_cur, dq_cur):
        return self.controller.get_command_torque(x_des, dx_des, q_cur, dq_cur)

    def get_observation(self):
        return self.controller.get_observation()
