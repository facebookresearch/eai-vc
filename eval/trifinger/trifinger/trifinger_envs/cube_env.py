import enum
import gym
import numpy as np
import pybullet
import torch
from dataclasses import dataclass

from scipy.spatial.transform import Rotation
import pinocchio as pin

import trifinger_simulation
import trifinger_simulation.visual_objects
from trifinger_simulation import trifingerpro_limits
import trifinger_simulation.tasks.move_cube as task
from trifinger_simulation.trifinger_platform import ObjectType

try:
    import robot_fingers
except ImportError:
    robot_fingers = None

import control.cube_utils as c_utils


class ActionType(enum.Enum):
    """Different action types that can be used to control the robot."""

    #: Use pure torque commands.  The action is a list of torques (one per
    #: joint) in this case.
    TORQUE = enum.auto()
    #: Use joint position commands.  The action is a list of angular joint
    #: positions (one per joint) in this case.  Internally a PD controller is
    #: executed for each action to determine the torques that are applied to
    #: the robot.
    POSITION = enum.auto()
    #: Use both torque and position commands.  In this case the action is a
    #: dictionary with keys "torque" and "position" which contain the
    #: corresponding lists of values (see above).  The torques resulting from
    #: the position controller are added to the torques in the action before
    #: applying them to the robot.
    TORQUE_AND_POSITION = enum.auto()


class BaseCubeEnv(gym.Env):
    """Gym environment for moving cubes with TriFingerPro."""

    def __init__(
        self,
        goal_pose: dict,
        action_type: ActionType = ActionType.POSITION,
        step_size: int = 1,
        difficulty: int = 1,
    ):
        """Initialize.

        Args:
            goal_pose: Goal trajectory for the cube.  If ``None`` a new
                random trajectory is sampled upon reset.
            action_type: Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size:  Number of actual control steps to be performed in one
                call of step().
        """
        # Basic initialization
        # ====================

        # if goal_pose is None:
        #    self.goal = task.sample_goal(difficulty).to_dict()
        # else:
        #    self.goal = goal_pose

        self.action_type = action_type

        self.info = {"difficulty": difficulty}

        self.difficulty = difficulty

        if step_size < 1:
            raise ValueError("step_size cannot be less than 1.")
        self.step_size = step_size

        # will be initialized in reset()
        self.platform = None

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        object_state_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=trifingerpro_limits.object_position.low,
                    high=trifingerpro_limits.object_position.high,
                ),
                "orientation": gym.spaces.Box(
                    low=trifingerpro_limits.object_orientation.low,
                    high=trifingerpro_limits.object_orientation.high,
                ),
            }
        )
        observation_state_space = gym.spaces.Box(
            low=np.append(
                trifingerpro_limits.robot_position.low,
                trifingerpro_limits.object_position.low,
            ),
            high=np.append(
                trifingerpro_limits.robot_position.high,
                trifingerpro_limits.object_position.high,
            ),
        )
        position_error_state_space = gym.spaces.Box(
            low=np.full(3, -999999, dtype=np.float32),
            high=np.full(3, 999999, dtype=np.float32),
        )
        orientation_error_state_space = gym.spaces.Box(
            low=np.full(4, -999999, dtype=np.float32),
            high=np.full(4, 999999, dtype=np.float32),
        )

        goal_state_space = gym.spaces.Box(
            low=np.append(
                trifingerpro_limits.object_position.low,
                trifingerpro_limits.object_orientation.low,
            ),
            high=np.append(
                trifingerpro_limits.object_position.high,
                trifingerpro_limits.object_orientation.high,
            ),
        )

        if self.action_type == ActionType.TORQUE:
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")

        self.observation_space = gym.spaces.Dict(
            {
                "t": gym.spaces.Discrete(task.EPISODE_LENGTH),
                "robot_position": robot_position_space,
                "robot_velocity": robot_velocity_space,
                "robot_torque": robot_torque_space,
                "object_vertices": object_state_space["position"],
                "object_position": object_state_space["position"],
                "object_orientation": object_state_space["orientation"],
                "observation": observation_state_space,
                "action": self.action_space,
                "desired_goal": goal_state_space,
                "achieved_goal": goal_state_space,
                "achieved_goal_position": object_state_space["position"],
                "achieved_goal_orientation": object_state_space["orientation"],
                "achieved_goal_position_error": position_error_state_space,
                "achieved_goal_orientation_error": orientation_error_state_space,
            }
        )

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
        return -task.evaluate_state(
            task.Pose(desired_goal[:3], desired_goal[3:]),  # expects pos + orientation
            task.Pose(achieved_goal[:3], achieved_goal[3:]),
            info["difficulty"],
        )

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
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

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
        return [seed]

    def _goal_orientation(self):
        return self.goal[3:]

    def _goal_pos(self):
        return self.goal[:3]

    def _create_observation(self, t, action):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)
        object_observation = camera_observation.filtered_object_pose

        position_error = np.linalg.norm(object_observation.position - self._goal_pos())

        # From trifinger_simulation tasks/move_cube/__init__.py evaluate_state()
        goal_rot = Rotation.from_quat(self._goal_orientation())
        actual_rot = Rotation.from_quat(object_observation.orientation)
        error_rot = goal_rot.inv() * actual_rot
        orientation_error = error_rot.magnitude()

        # Get cube vertices
        obj_pose = {
            "position": object_observation.position,
            "orientation": object_observation.orientation,
        }
        v_wf_dict = c_utils.get_vertices_wf(obj_pose)
        observation = {
            "t": t,
            "robot_position": robot_observation.position,
            "robot_velocity": robot_observation.velocity,
            "robot_torque": robot_observation.torque,
            "object_position": object_observation.position,
            "object_orientation": object_observation.orientation,
            "observation": np.append(
                robot_observation.position, object_observation.position
            ),
            "object_vertices": v_wf_dict,
            "action": action,
            "desired_goal": self.goal,
            "achieved_goal_position": object_observation.position,
            "achieved_goal_orientation": object_observation.orientation,
            "achieved_goal_position_error": position_error,
            "achieved_goal_orientation_error": orientation_error,
            "achieved_goal": np.append(
                object_observation.position, object_observation.orientation
            ),
        }

        # Save camera observation images
        if self.enable_cameras:
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

    def close(self):
        pybullet.disconnect()
        super().close()


class SimCubeEnv(BaseCubeEnv):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        goal_pose: dict = None,
        action_type: ActionType = ActionType.POSITION,
        step_size: int = 1,
        difficulty: int = 1,
        visualization: bool = False,
        no_collisions: bool = False,
        enable_cameras: bool = False,
        finger_type: str = "trifingerpro",
        camera_delay_steps: int = 90,
        time_step: float = 0.001,
        object_type: ObjectType = ObjectType.COLORED_CUBE,
        enable_shadows: bool = False,
        camera_view: str = "default",
        arena_color: str = "default",
        random_q_init: bool = False,
        fix_cube_base: bool = False,
    ):
        """Initialize.

        Args:
            goal_pose: Goal trajectory for the cube.  If ``None`` a new
                random trajectory is sampled upon reset.
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
            random_q_init (bool): If true, use random intitial joint positions
            fix_cube_base (bool): Fix cube base
        """
        super().__init__(
            goal_pose=goal_pose,
            action_type=action_type,
            step_size=step_size,
            difficulty=difficulty,
        )

        self.visualization = visualization
        self.no_collisions = no_collisions
        self.enable_cameras = enable_cameras
        self.finger_type = finger_type
        self.time_step = time_step
        self.enable_shadows = enable_shadows
        self.camera_view = camera_view
        self.random_q_init = random_q_init

        # initialize simulation
        # initial_robot_position = trifingerpro_limits.robot_position.default
        self.q_nominal = np.array([-0.08, 1.15, -1.5] * 3)

        if self.random_q_init:
            self.initial_robot_position = self.sample_init_robot_position()
        else:
            self.initial_robot_position = self.q_nominal

        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=self.visualization,
            enable_cameras=self.enable_cameras,
            finger_type=self.finger_type,
            time_step_s=self.time_step,
            initial_robot_position=self.initial_robot_position,
            camera_delay_steps=camera_delay_steps,
            object_type=object_type,
            enable_shadows=self.enable_shadows,
            camera_view=self.camera_view,
            arena_color=arena_color,
            fix_cube_base=fix_cube_base,
        )

        # visualize the cube vertices
        if self.visualization and not self.enable_cameras:
            self.draw_verts = True
        else:
            self.draw_verts = False
        self.vert_markers = None

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
        # TODO figure out a better way for this
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if not self.action_space.contains(np.array(action, dtype=np.float32)):
            raise ValueError("Given action is not contained in the action space.")

        num_steps = self.step_size

        # ensure episode length is not exceeded due to step_size
        step_count_after = self.step_count + num_steps
        if step_count_after > task.EPISODE_LENGTH:
            excess = step_count_after - task.EPISODE_LENGTH
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > task.EPISODE_LENGTH:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
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

            observation = self._create_observation(self.info["time_index"], action)

            reward = 0
            reward += self.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.info,
            )

            # Draw cube vertices from observation
            if self.draw_verts:
                v_wf_dict = observation["object_observation"]["vertices"]
                positions = [v_wf for k, v_wf in v_wf_dict.items()]
                self.vert_markers.set_state(positions)

        is_done = self.step_count >= task.EPISODE_LENGTH

        return observation, reward, is_done, self.info

    def reset(
        self,
        goal_pose_dict=None,
        init_pose_dict=None,
        init_robot_position=None,
        random_init_cube_pos=False,
    ):
        """Reset the environment."""

        ##hard-reset simulation
        # del self.platform

        # initialize cube at the centre
        if init_pose_dict is None:
            initial_object_pose = task.sample_goal(difficulty=-1)
            if not random_init_cube_pos:
                # Hardcode init pose to arena center
                initial_object_pose.position = [
                    0,
                    0,
                    task._CUBE_WIDTH / 2,
                ]
        else:
            initial_object_pose = task.Pose.from_dict(init_pose_dict)

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

        if self.no_collisions:
            self.disable_collisions()

        # if no goal is given, sample one randomly
        if goal_pose_dict is None:
            if self.difficulty == 0 or self.difficulty not in [1, 2, 3]:
                self.goal = np.append(
                    initial_object_pose.position, initial_object_pose.orientation
                )
            else:
                pose = task.sample_goal(self.difficulty)
                self.goal = np.append(pose.position, pose.orientation)
        else:
            pose = goal_pose_dict
            self.goal = np.append(pose["position"], pose["orientation"])

        # visualize the goal
        if self.visualization and not self.enable_cameras:
            self.goal_marker = trifinger_simulation.visual_objects.CubeMarker(
                width=task._CUBE_WIDTH,
                position=self._goal_pos(),
                orientation=self._goal_orientation(),
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )

        if self.draw_verts:
            v_wf_dict = c_utils.get_vertices_wf(initial_object_pose.to_dict())
            if self.vert_markers is None:
                self.vert_markers = trifinger_simulation.visual_objects.Marker(
                    8,
                    goal_size=0.005,
                    initial_position=[v_wf for k, v_wf in v_wf_dict.items()],
                )
            else:
                positions = [v_wf for k, v_wf in v_wf_dict.items()]
                self.vert_markers.set_state(positions)

        self.info = {"time_index": -1, "goal": self.goal, "difficulty": self.difficulty}

        self.step_count = 0

        new_obs = self._create_observation(0, self._initial_action)

        return new_obs

    def _set_sim_params(self):
        """Set pybullet GUI params"""

        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_GUI, 0
        )  # Turn off debug camera visuals
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_SHADOWS, self.enable_shadows
        )  # Turn off shadow rendering

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


class SimCubeEnvNYU(SimCubeEnv):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        goal_pose: dict = None,
        action_type: ActionType = ActionType.TORQUE,
        step_size: int = 1,
        difficulty: int = 1,
        visualization: bool = False,
        no_collisions: bool = False,
        enable_cameras: bool = False,
        finger_type: str = "trifingernyu",
        camera_delay_steps: int = 90,
        time_step: float = 0.001,
        init_difficulty=-1,
    ):
        super().__init__(
            goal_pose,
            action_type,
            step_size,
            difficulty,
            visualization,
            no_collisions,
            enable_cameras,
            finger_type,
            camera_delay_steps,
            time_step,
        )
        self.init_difficulty = init_difficulty

    def reset(
        self,
        goal_pose_dict=None,
        init_pose_dict=None,
        init_robot_position=None,
        random_init_cube_pos=True,
        max_goal_orn_diff=np.pi / 2,
    ):
        """Reset the environment."""

        ##hard-reset simulation
        # del self.platform

        # initialize cube at the centre
        if init_pose_dict is None:
            initial_object_pose = task.sample_goal(difficulty=self.init_difficulty)
            if not random_init_cube_pos:
                # Hardcode init pose to arena center
                initial_object_pose.position = [
                    0,
                    0,
                    task._CUBE_WIDTH / 2,
                ]
            # for now make sure object z-axis aligns with world z-axis
            yaw_angle = np.random.uniform(-np.pi, np.pi)
            yaw_rot = Rotation.from_euler("z", yaw_angle)
            initial_object_pose.orientation = yaw_rot.as_quat()
        else:
            initial_object_pose = task.Pose.from_dict(init_pose_dict)

        self.platform.reset(
            initial_object_pose=initial_object_pose,
            initial_robot_position=init_robot_position,
        )

        # Set pybullet GUI params
        self._set_sim_params()

        if self.no_collisions:
            self.disable_collisions()

        # if no goal is given, sample one randomly
        if goal_pose_dict is None:
            if self.difficulty == 0 or self.difficulty not in [1, 2, 3]:
                self.goal = np.append(
                    initial_object_pose.position, initial_object_pose.orientation
                )
            else:
                pose = task.sample_goal(self.difficulty)
                # random yaw orientation
                yaw_angle = np.random.uniform(-max_goal_orn_diff, max_goal_orn_diff)
                yaw_rot = Rotation.from_euler("z", yaw_angle)
                orientation = yaw_rot * Rotation.from_quat(
                    initial_object_pose.orientation
                )
                self.goal = np.append(pose.position, orientation.as_quat())
        else:
            pose = goal_pose_dict
            self.goal = np.append(pose["position"], pose["orientation"])

        # visualize the goal
        if self.visualization and not self.enable_cameras:
            self.goal_marker = trifinger_simulation.visual_objects.CubeMarker(
                width=task._CUBE_WIDTH,
                position=self._goal_pos(),
                orientation=self._goal_orientation(),
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )

        if self.draw_verts:
            v_wf_dict = c_utils.get_vertices_wf(initial_object_pose.to_dict())
            if self.vert_markers is None:
                self.vert_markers = trifinger_simulation.visual_objects.Marker(
                    8,
                    goal_size=0.005,
                    initial_position=[v_wf for k, v_wf in v_wf_dict.items()],
                )
            else:
                positions = [v_wf for k, v_wf in v_wf_dict.items()]
                self.vert_markers.set_state(positions)

        self.info = {"time_index": -1, "goal": self.goal, "difficulty": self.difficulty}

        self.step_count = 0

        new_obs = self._create_observation(0, self._initial_action)

        return new_obs


@dataclass(frozen=True)
class SimCubeEnvParams:
    """
    TODO explain params
    :param force_eval_start_dist: Generate the start positions from the eval offset.
    """

    env: str = ("SimCubeEnv",)
    goal_pose: torch.Tensor = (None,)
    action_type: ActionType = (ActionType.TORQUE,)
    visualization: bool = (False,)
    no_collisions: bool = (True,)
    enable_cameras: bool = (True,)
    finger_type: str = ("trifingerpro",)
    time_step: int = (0,)
    camera_delay_steps: int = 0
