import enum
import typing
import gym
import math
import numpy as np
import pybullet
import torch

import trifinger_simulation
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


class ReachEnv(gym.Env):
    """Gym environment for moving cubes with TriFingerPro."""

    def __init__(
        self,
        render_mode: str = "",
        fixed_goal: bool = True,
        visual_observation: bool = False,
        action_type: ActionType = ActionType.TORQUE,
        step_size: int = 50,
        visualization: bool = False,
        no_collisions: bool = False,
        enable_cameras: bool = True,
        finger_type: str = "trifingerpro",
        camera_delay_steps: int = 90,
        time_step: float = 0.004,
        randomize_starts: bool = False,
        randomize_all: bool = False,
        sample_radius: float = 0.01,
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
        self.no_collisions = no_collisions
        self.enable_cameras = enable_cameras
        self.finger_type = finger_type
        self.time_step = time_step
        self.randomize_starts = randomize_starts
        self.sample_radius = sample_radius
        self.randomize_all = randomize_all

        if self.randomize_all:
            self.randomize_starts = True

        self.max_episode_len = REACH_EPISODE_LENGTH
        if self.randomize_all:
            self.max_episode_len = 1000
        # initialize simulation
        # initial_robot_position = trifingerpro_limits.robot_position.default
        initial_robot_position = [-0.08, 1.15, -1.5] * 3

        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=self.visualization,
            enable_cameras=self.enable_cameras,
            finger_type=self.finger_type,
            time_step_s=self.time_step,
            initial_robot_position=initial_robot_position,
            camera_delay_steps=camera_delay_steps,
        )

        self.hand_kinematics = HandKinematics(self.platform)
        # visualize the cube vertices
        if self.visualization and not self.enable_cameras:
            self.draw_verts = False
        else:
            self.draw_verts = False
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

        self.img_size = (270, 270, 3)
        self.image_observation_space = gym.spaces.Box(
            low=np.zeros(self.img_size), high=np.ones(self.img_size) * 255
        )
        self.observation_space = self.observation_state_space
        if self.visual_observation:
            # self.observation_space = gym.spaces.Dict(
            #     {
            #         "observation": self.image_observation_space,
            #         "desired_goal": goal_state_space,
            #         "achieved_goal": goal_state_space,
            #         "f0_dist": self.ftip_dist_space,
            #         "f1_dist": self.ftip_dist_space,
            #         "f2_dist": self.ftip_dist_space,
            #     }
            # )
            self.observation_space = gym.spaces.Dict(
                {
                    "pixels": self.image_observation_space,
                    "ftip_dist": self.observation_state_space,
                }
            )

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
            camera_observation = self.platform.get_camera_observation(self.step_count)

            return camera_observation.cameras[0].image
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

    def _get_fingertip_pos(self, t):
        r_obs = self.platform.get_robot_observation(t)
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
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")
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

            robot_obs = self.platform.get_robot_observation(self.step_count - 1)
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
            t = self.platform.append_desired_action(robot_action)

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
        DEBUG_PRINT = False
        if DEBUG_PRINT:
            np.set_printoptions(precision=3)
            print("action passed in:")
            print(initial_action)
            print("action after:")
            print(action)
            print("achieved position")
            print(achieved_position)
            print("desired position")
            print(initial_position)
            print(action)
            print(initial_position + action)
            print("error")
            err = (initial_position + action) - achieved_position
            print(err)
            print("\n \n error val:")
            step_err = np.linalg.norm(err)
            print(step_err)

        reward += self.compute_reward(
            observation[:9],
            observation[9:],
            self.info,
        )
        if DEBUG_PRINT:
            print("diff:" + str(np.linalg.norm(x_des - achieved_position)))

        completed = np.linalg.norm(observation[9:] - observation[:9]) < 0.01

        is_done = self.step_count >= self.max_episode_len

        if self.visual_observation:
            observation = {"pixels": self.render("rgb_array"), "ftip_dist": observation}
        return observation, reward, is_done, self.info

    def reset(self, init_pose_dict=None, init_robot_position=None):
        """Reset the environment."""

        ##hard-reset simulation
        # del self.platform

        # initialize cube at the centre
        if init_pose_dict is None:
            initial_object_pose = task.sample_goal(difficulty=-1)
            initial_object_pose.position = [
                0,
                0,
                task._CUBE_WIDTH / 2,
            ]  # TODO hardcoded init pose to arena center
        else:
            initial_object_pose = task.Pose.from_dict(init_pose_dict)

        if self.randomize_starts:
            initial_robot_position = (
                self.platform.spaces.robot_position.default
                + np.random.rand(9) * self.sample_radius
            )
            self.platform.reset(
                initial_object_pose=initial_object_pose,
                # initial_robot_position=initial_robot_position,
            )
        else:
            self.platform.reset(
                initial_object_pose=initial_object_pose,
                # initial_robot_position=initial_robot_position,
            )

        # Set pybullet GUI params
        self._set_sim_params()

        if self.no_collisions:
            self.disable_collisions()

        self.goal = np.array(
            [
                0.1029,
                0.1416,
                0.0319,
                0.0712,
                -0.160,
                0.0319,  # 0.0819,
                -0.174,
                0.0183,
                0.0319,  # 0.0819,
            ]
        )
        if self.randomize_all:
            target_joint_config = np.asarray(
                sample.feasible_random_joint_positions_for_reaching(
                    self.platform.simfinger, self.action_bounds
                )
            )
            goal = self.platform.simfinger.kinematics.forward_kinematics(
                target_joint_config
            )
            self.goal = np.array(goal).flatten()

        # visualize the goal
        self.goal_marker.set_state([self.goal[0:3], self.goal[3:6], self.goal[6:9]])

        if self.draw_verts:
            v_wf_dict = c_utils.get_vertices_wf(initial_object_pose.to_dict())
            if self.vert_markers is None:
                self.vert_markers = visual_objects.Marker(
                    8,
                    goal_size=0.005,
                    initial_position=[v_wf for k, v_wf in v_wf_dict.items()],
                )
            else:
                positions = [v_wf for k, v_wf in v_wf_dict.items()]
                self.vert_markers.set_state(positions)

        self.info = {"time_index": -1, "goal": self.goal}
        self.step_count = 0

        if self.visual_observation:
            new_obs = {
                "pixels": self.render("rgb_array"),
                "ftip_dist": self._create_observation(0),
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
        robot_observation = self.platform.get_robot_observation(t)
        # camera_observation = self.platform.get_camera_observation(t)
        ftip_pos = self.hand_kinematics.get_ft_pos(robot_observation.position)
        observation = torch.cat(
            (torch.FloatTensor(ftip_pos), torch.FloatTensor(self.goal))
        )
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

    def disable_collisions(self):
        """Disable collisions between finger and object, for debugging finger controllers"""

        obj_id = self.platform.cube._object_id
        robot_id = self.platform.simfinger.finger_id
        obj_link_id = -1
        finger_link_ids = (
            self.platform.simfinger.pybullet_link_indices
            + self.platform.simfinger.pybullet_tip_link_indices
        )

        for link_id in finger_link_ids:
            pybullet.setCollisionFilterPair(
                robot_id, obj_id, link_id, obj_link_id, enableCollision=0
            )

        # Make object invisible
        # pybullet.changeVisualShape(obj_id, obj_link_id, rgbaColor=[0,0,0,0])


# kinematics wrapper
class HandKinematics:
    def __init__(self, platform):
        self.Nf = 3  # Number of fingers
        self.Nq = self.Nf * 3  # Number of joints in hand
        self.platform = platform
        # class with kinematics functions
        self.kinematics = CustomPinocchioUtils(
            self.platform.simfinger.finger_urdf_path,
            self.platform.simfinger.tip_link_names,
            self.platform.simfinger.link_names,
        )

        self.controller = ImpedanceController(self.kinematics)

    def get_ft_pos(self, q):
        """Get fingertip positions given current joint configuration q"""

        ft_pos = np.array(self.kinematics.forward_kinematics(q)).reshape(self.Nq)
        return ft_pos

    def get_torque(self, x_des, dx_des, q_cur, dq_cur):
        return self.controller.get_command_torque(x_des, dx_des, q_cur, dq_cur)
