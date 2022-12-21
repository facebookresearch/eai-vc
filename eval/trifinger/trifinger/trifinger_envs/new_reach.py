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

from trifinger_envs.cube_env import ActionType
from control.impedance_controller import ImpedanceController
from control.custom_pinocchio_utils import CustomPinocchioUtils

try:
    import robot_fingers
except ImportError:
    robot_fingers = None

import control.cube_utils as c_utils


# REACH_EPISODE_LENGTH = 500
REACH_EPISODE_LENGTH = 1000


class NewReachEnv(gym.Env):
    """Gym environment for moving cubes with TriFingerPro."""

    def __init__(
        self,
        render_mode: str = "",
        fixed_goal: bool = True,
        visual_observation: bool = False,
        action_type: ActionType = ActionType.TORQUE,
        step_size: int = 50,
        visualization: bool = False,
        enable_cameras: bool = True,
        camera_id: int = 0,
        finger_type: str = "trifingerpro",
        camera_delay_steps: int = 90,
        time_step: float = 0.004,
        randomize_starts: bool = True,
        randomize_all: bool = False,
        sample_radius: float = 0.00,
        max_goal_dist: float = 100,
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
        """
        super().__init__()
        if render_mode == "human":
            visualization = True
        self.visualization = visualization
        self.enable_cameras = enable_cameras
        self.finger_type = finger_type
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
        # initial_robot_position = trifingerpro_limits.robot_position.default
        initial_robot_position = [-0.08, 1.15, -1.5] * 3

        # self.platform = trifinger_simulation.TriFingerPlatform(
        #     visualization=self.visualization,
        #     enable_cameras=self.enable_cameras,
        #     finger_type=self.finger_type,
        #     time_step_s=self.time_step,
        #     initial_robot_position=initial_robot_position,
        #     camera_delay_steps=camera_delay_steps,
        # )

        self.simfinger = trifinger_simulation.SimFinger(
            finger_type=self.finger_type,
            time_step=self.time_step,
            # obj_containment=True,
            # obj_display=True,
            enable_visualization=self.visualization,
        )

        self.hand_kinematics = HandKinematics(self.simfinger)
        target_positions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        camera_up_vectors = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
        field_of_view = 33
        self.tricamera = camera.TriFingerCameras(
            pybullet_client_id=self.simfinger._pybullet_client_id,
            target_positions=target_positions,
            camera_up_vectors=camera_up_vectors,
            field_of_view=field_of_view,
        )
        self.vert_markers = None
        # Basic initialization
        # ====================

        self.visual_observation = visual_observation
        self.action_type = action_type
        self.previous_dist_to_goal = 1
        self.previous_joint_velocities = 0
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
            low=np.ones(9) * -2,
            high=np.ones(9) * 2,
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
                "ftip_dist": self.observation_state_space,
                #  "scaled_success": self.total_success_space,
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
        self.goal_marker = visual_objects.Marker(
            3,
            initial_position=np.array(
                [
                    [0.1029, 0.1416, 0.0319],
                    [0.0712, -0.160, 0.0319],  # 0.0819,
                    [-0.174, 0.0183, 0.0319],  # 0.0819,
                ]
            ),
        )

    """
        mode=rgb_array returns numpy.ndarray with shape (x, y, 3) of current observation
    """

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
            self.show_marker()
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

    def start_to_goal(self):
        d = []
        x0 = np.array(self.start_pos).flatten()
        for i in range(3):
            x = np.linalg.norm(
                x0[(3 * i) : (3 * i) + 3] - self.goal[(3 * i) : (3 * i) + 3]
            )
            d.append(torch.abs(torch.tensor(x)))
        return d

    def scaled_success(self, obs, total_dist):
        # d = {}
        total_success = 0
        for i in range(3):
            k = f"f%s_success" % i
            dist_to_goal = torch.norm(
                torch.tensor(
                    obs[(3 * i) : (3 * i) + 3] - self.goal[(3 * i) : (3 * i) + 3]
                )
            )
            success = 1 - (dist_to_goal / total_dist[i])
            total_success += success
            # d[k] = success
        if total_success < 0:
            total_success = 0
        return total_success / 3

    def _get_fingertip_pos(self, t):
        # r_obs = self.platform.get_robot_observation(t)
        r_obs = self.simfinger.get_observation(t)
        # pass joint pos for xyz coordinates
        return self.hand_kinematics.get_ft_pos(r_obs.position)

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
        # current_dist_to_goal = np.linalg.norm(desired_goal - achieved_goal)
        # rewards = []
        # rewards.append(self.previous_dist_to_goal - current_dist_to_goal)
        # rewards.append(-current_dist_to_goal)

        # goal_reward = -1 * torch.mean(torch.abs(desired_goal - achieved_goal))
        # dense_rewards = torch.sum(torch.tensor(rewards) * self.dense_reward_weights)
        # self.previous_dist_to_goal = current_dist_to_goal
        # reward = dense_rewards + goal_reward
        reward = -torch.norm(desired_goal - achieved_goal)
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
        initial_action = np.copy(action)
        action = action / 50.0
        action = np.clip(action, -0.02, 0.02)

        if not self.action_space.contains(np.array(action, dtype=np.float32)):
            print(action)
            raise ValueError("Given action is not contained in the action space.")

        num_steps = self.step_size

        # ensure episode length is not exceeded due to step_size
        step_count_after = self.step_count + num_steps
        if step_count_after > self.max_episode_len:
            excess = step_count_after - self.max_episode_len
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        x_curr = self._get_fingertip_pos(self.step_count)
        initial_position = np.copy(x_curr)
        x_des = x_curr + action
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > self.max_episode_len:
                raise RuntimeError("Exceeded number of steps for one episode.")

            robot_obs = self.simfinger.get_observation(self.step_count - 1)
            joint_position = robot_obs.position
            joint_velocity = robot_obs.velocity

            x_curr = self.hand_kinematics.get_ft_pos(joint_position)
            x_i = x_curr + (action / num_steps)
            dx_des = action / (self.step_size * self.time_step)

            torque = self.hand_kinematics.get_torque(
                x_i,
                dx_des,
                joint_position,
                joint_velocity,
            )
            torque = np.clip(
                torque, self.robot_torque_space.low, self.robot_torque_space.high
            )

            # send action to robot
            robot_action = self._gym_action_to_robot_action(torque)
            t = self.simfinger.append_desired_action(robot_action)

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            # self.info["time_index"] = t + 1

            # Alternatively use the observation of step t.  This is the
            # observation from the moment before action_t is applied, i.e. the
            # result of that action is not yet visible in this observation.
            #
            # When using this observation, the resulting cumulative reward
            # should match exactly the one computed during replay (with the
            # above it will differ slightly).
            self.info["time_index"] = t

            observation = self._create_observation(self.info["time_index"])

        reward = 0
        achieved_position = observation[:9]

        reward += self.compute_reward(
            observation[:9],
            observation[9:],
            self.info,
        )

        completed = np.linalg.norm(observation[9:] - observation[:9]) < 0.01

        is_done = self.step_count >= self.max_episode_len

        if self.visual_observation:
            observation = {
                "pixels": self.render("rgb_array"),
                "ftip_dist": observation,
                "scaled_success": self.scaled_success(
                    observation, self.start_to_goal()
                ),
            }
        return observation, reward, is_done, self.info

    def rand_step(self, tensordict):
        action = (np.random.rand(9) * 2) - np.ones(9)
        return self.step(action)

    def state_dict(self):
        return {}

    def fake_tensordict(self):
        observation = self._create_observation(self.info["time_index"])
        if self.visual_observation:
            observation = {
                "pixels": self.render("rgb_array"),
                "ftip_dist": observation,
                "scaled_success": self.scaled_success(
                    observation, self.start_to_goal()
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
            eepos = self.simfinger.kinematics.forward_kinematics(initial_robot_position)
            if self._is_above_table(eepos):
                return initial_robot_position

    def choose_goal(self):
        # makes sure that goal is above bowl
        while True:
            target_joint_config = np.asarray(
                sample.feasible_random_joint_positions_for_reaching(
                    self.simfinger, self.action_bounds
                )
            )
            goal = self.simfinger.kinematics.forward_kinematics(target_joint_config)
            dist_to_start = np.linalg.norm(
                np.array(self.start_pos).flatten() - np.array(goal).flatten()
            )
            if self._is_above_table(goal):
                return np.array(goal).flatten()

    def hide_marker_from_camera(self):
        self.goal_marker.set_state(
            [
                self.action_bounds["high"][0:3],
                self.action_bounds["high"][3:6],
                self.action_bounds["high"][6:9],
            ]
        )

    def show_marker(self):
        self.goal_marker.set_state([self.goal[0:3], self.goal[3:6], self.goal[6:9]])

    def get_start_pos(self, return_goal=False):
        return np.concatenate(
            (self.start_pos[0], self.start_pos[1], self.start_pos[2], self.goal)
        )

    def reset(self, init_pose_dict=None, init_robot_position=None):
        """Reset the environment."""
        self.simfinger.reset()
        start_to_goal = 1000.0
        while start_to_goal > self.max_goal_dist:
            if self.randomize_starts:
                initial_robot_position = self.choose_start_pos()
                self.start_pos = self.simfinger.kinematics.forward_kinematics(
                    initial_robot_position
                )
                self.simfinger.reset_finger_positions_and_velocities(
                    initial_robot_position
                )

            # Set pybullet GUI params
            self._set_sim_params()

            self.goal = np.array(
                [
                    0.05,
                    0.07,
                    0.0319,  # RED, front right (camera back right)
                    0.045,
                    -0.08,
                    0.0319,  # GREEN, (camera top left)
                    -0.08,
                    0.009,
                    0.0319,  # BLUE, camera front
                ]
            )
            if self.randomize_all:
                self.goal = self.choose_goal()
            start_to_goal = 0
            for i in self.start_to_goal():
                start_to_goal += i

        # visualize the goal
        # self.hide_marker_from_camera()
        self.goal_marker.set_state([self.goal[0:3], self.goal[3:6], self.goal[6:9]])
        self.show_marker()

        self.info = {"time_index": -1, "goal": self.goal}
        self.step_count = 0
        if self.visual_observation:
            obs = self._create_observation(0)
            new_obs = {
                "pixels": self.render("rgb_array"),
                "ftip_dist": obs,
                "scaled_success": self.scaled_success(obs, self.start_to_goal()),
            }
        else:
            new_obs = self._create_observation(0)

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
        # robot_observation = self.platform.get_robot_observation(t)
        robot_observation = self.simfinger.get_observation(t)
        # camera_observation = self.platform.get_camera_observation(t)
        ftip_pos = self.hand_kinematics.get_ft_pos(robot_observation.position)
        observation = torch.cat(
            (torch.FloatTensor(ftip_pos), torch.FloatTensor(self.goal))
        )
        return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.simfinger.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.simfinger.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.simfinger.Action(
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
