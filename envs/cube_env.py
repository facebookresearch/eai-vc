import enum
import typing
import gym
import numpy as np
import pybullet
from scipy.spatial.transform import Rotation
import os
import sys

import trifinger_simulation
import trifinger_simulation.visual_objects
from trifinger_simulation import trifingerpro_limits
import trifinger_simulation.tasks.move_cube as task

try:
    import robot_fingers
except ImportError:
    robot_fingers = None

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

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

        if goal_pose is None:
            self.goal = task.sample_goal(difficulty).to_dict()
        else:
            self.goal = goal_pose

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
                "robot_observation": gym.spaces.Dict(
                    {
                        "position": robot_position_space,
                        "velocity": robot_velocity_space,
                        "torque": robot_torque_space,
                    }
                ),
                "object_observation": gym.spaces.Dict(
                    {
                        "position": object_state_space["position"],
                        "orientation": object_state_space["orientation"],
                    }
                ),
                "action": self.action_space,
                "desired_goal": object_state_space["position"],
                "achieved_goal": object_state_space["position"],
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
            task.Pose.from_dict(desired_goal),
            task.Pose.from_dict(achieved_goal),
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

    def _create_observation(self, t, action):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)
        object_observation = camera_observation.filtered_object_pose
        
        position_error = np.linalg.norm(object_observation.position - self.goal["position"])

        # From trifinger_simulation tasks/move_cube/__init__.py evaluate_state()
        goal_rot = Rotation.from_quat(self.goal["orientation"])
        actual_rot = Rotation.from_quat(object_observation.orientation)
        error_rot = goal_rot.inv() * actual_rot
        orientation_error = error_rot.magnitude()

        # Get cube vertices
        obj_pose = {"position": object_observation.position, "orientation": object_observation.orientation}
        v_wf_dict = c_utils.get_vertices_wf(obj_pose)

        observation = {
            "t": t,
            "robot_observation": {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
            },
            "object_observation": {
                "position": object_observation.position,
                "orientation": object_observation.orientation,
                "vertices": v_wf_dict,
            },
            "action": action,
            "desired_goal": self.goal,
            "achieved_goal": {
                "position": object_observation.position,
                "orientation": object_observation.orientation,
                "position_error": position_error,
                "orientation_error": orientation_error,
            },
        }

        # Save camera observation images
        if self.enable_cameras:
            camera_observation_dict = {
                                        "camera60": {"image"    : camera_observation.cameras[0].image,
                                                     "timestamp": camera_observation.cameras[0].timestamp},
                                        "camera180": {"image"    : camera_observation.cameras[1].image,
                                                     "timestamp": camera_observation.cameras[1].timestamp},
                                        "camera300": {"image"    : camera_observation.cameras[2].image,
                                                     "timestamp": camera_observation.cameras[2].timestamp},
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

        # initialize simulation
        #initial_robot_position = trifingerpro_limits.robot_position.default
        initial_robot_position = [-0.08, 1.15, -1.5] * 3

        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=self.visualization,
            enable_cameras=self.enable_cameras,
            finger_type=self.finger_type,
            time_step_s=self.time_step,
            initial_robot_position=initial_robot_position,
            camera_delay_steps=camera_delay_steps,
        )

        # visualize the cube vertices
        if self.visualization and not self.enable_cameras:
            self.draw_verts = True
        else:
            self.draw_verts = False
        self.vert_markers = None

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

        if not self.action_space.contains(np.array(action, dtype=np.float32)):
            raise ValueError(
                "Given action is not contained in the action space."
            )

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
            #self.info["time_index"] = t

            observation = self._create_observation(
                self.info["time_index"], action
            )

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

    def reset(self, goal_pose_dict=None, init_pose_dict=None):
        """ Reset the environment. """

        ##hard-reset simulation
        #del self.platform

        # initialize cube at the centre
        if init_pose_dict is None:
            initial_object_pose = task.sample_goal(difficulty=-1)
            initial_object_pose.position = [0,0,task._CUBE_WIDTH/2] # TODO hardcoded init pose to arena center
        else:
            initial_object_pose = task.Pose.from_dict(init_pose_dict)

        self.platform.reset(initial_object_pose)

        # Set pybullet GUI params
        self._set_sim_params()

        if self.no_collisions:
            self.disable_collisions()

        # if no goal is given, sample one randomly
        if goal_pose_dict is None:
            self.goal = task.sample_goal(self.difficulty).to_dict()
        else:
            self.goal = goal_pose_dict

        # visualize the goal
        if self.visualization and not self.enable_cameras:
            self.goal_marker = trifinger_simulation.visual_objects.CubeMarker(
                width=task._CUBE_WIDTH,
                position=self.goal["position"],
                orientation=self.goal["orientation"],
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )


        if self.draw_verts:
            v_wf_dict = c_utils.get_vertices_wf(initial_object_pose.to_dict())
            if self.vert_markers is None:
                self.vert_markers = trifinger_simulation.visual_objects.Marker(
                    8, goal_size = 0.005,
                    initial_position = [v_wf for k, v_wf in v_wf_dict.items()],
                )
            else:
                positions = [v_wf for k, v_wf in v_wf_dict.items()]
                self.vert_markers.set_state(positions)

        self.info = {"time_index": -1, "goal": self.goal, "difficulty": self.difficulty}

        self.step_count = 0

        return self._create_observation(0, self._initial_action)

    def _set_sim_params(self):
        """ Set pybullet GUI params """

        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0) # Turn off debug camera visuals

    def disable_collisions(self):
        """ Disable collisions between finger and object, for debugging finger controllers """
        
        obj_id = self.platform.cube._object_id
        robot_id = self.platform.simfinger.finger_id
        obj_link_id = -1
        finger_link_ids = self.platform.simfinger.pybullet_link_indices +\
                            self.platform.simfinger.pybullet_tip_link_indices

        for link_id in finger_link_ids:
            pybullet.setCollisionFilterPair(robot_id, obj_id, link_id, obj_link_id, enableCollision=0)

        # Make object invisible 
        #pybullet.changeVisualShape(obj_id, obj_link_id, rgbaColor=[0,0,0,0])

