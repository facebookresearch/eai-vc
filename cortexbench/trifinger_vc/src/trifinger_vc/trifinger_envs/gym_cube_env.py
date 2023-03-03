#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import enum
import gym
import numpy as np
import pybullet
import torch
from dataclasses import dataclass

from scipy.spatial.transform import Rotation

import trifinger_simulation
import trifinger_simulation.visual_objects
from trifinger_simulation import trifingerpro_limits
import trifinger_simulation.tasks.move_cube as task
from trifinger_simulation.trifinger_platform import ObjectType
from trifinger_simulation import camera

from trifinger_vc.trifinger_envs.action import ActionType
from trifinger_vc.control.impedance_controller import ImpedanceController
from trifinger_vc.control.custom_pinocchio_utils import CustomPinocchioUtils
import trifinger_vc.control.cube_utils as c_utils

MOVE_CUBE_EPISODE = 1000


class BaseCubeEnv(gym.Env):
    """Gym environment for moving cubes with TriFingerPro."""

    def __init__(
        self,
        goal_pose: dict,
        action_type: ActionType = ActionType.POSITION,
        step_size: int = 100,
        difficulty: int = 1,
        visual_observation: bool = False,
        seq_eval: bool = True,
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
        self.max_episode_len = MOVE_CUBE_EPISODE

        if step_size < 1:
            raise ValueError("step_size cannot be less than 1.")
        self.step_size = step_size
        self.visual_observation = visual_observation
        self.camera_id = 0

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
        observation_state_space = gym.spaces.Box(
            low=np.concatenate(
                (
                    trifingerpro_limits.robot_position.low,
                    trifingerpro_limits.object_position.low,
                    trifingerpro_limits.object_orientation.low,
                    goal_state_space.low,
                ),
                axis=0,
            ),
            high=np.concatenate(
                (
                    trifingerpro_limits.robot_position.high,
                    trifingerpro_limits.object_position.high,
                    trifingerpro_limits.object_orientation.high,
                    goal_state_space.high,
                ),
                axis=0,
            ),
        )

        observation_state_space = gym.spaces.Box(
            low=np.ones(23) * -10,
            high=np.ones(23) * 10,
        )

        self.robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )

        if self.action_type == ActionType.TORQUE:
            #     self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            #     self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            #     self.action_space = gym.spaces.Dict(
            #         {
            #             "torque": robot_torque_space,
            #             "position": robot_position_space,
            #         }
            #     )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")

        self.action_space = gym.spaces.Box(
            low=-2 * np.ones(9),
            high=2 * np.ones(9),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "t": gym.spaces.Discrete(MOVE_CUBE_EPISODE + 1),
                "state_obs": observation_state_space,
            }
        )
        if self.visual_observation:
            self.img_size = (270, 270, 3)
            self.image_observation_space = gym.spaces.Box(
                low=np.zeros(self.img_size), high=np.ones(self.img_size) * 255
            )

            self.success_space = gym.spaces.Box(low=-1 * np.ones(1), high=np.ones(1))
            self.observation_space = gym.spaces.Dict(
                {
                    "t": gym.spaces.Discrete(MOVE_CUBE_EPISODE + 1),
                    "state_obs": observation_state_space,
                    "scaled_success": self.success_space,
                    "scaled_success_reach": self.success_space,
                    "pixels": self.image_observation_space,
                }
            )
        self.train_start_list = self.get_start_list()
        self.train_goal_list = self.get_goal_list()

        self.eval_start_list = self.get_eval_start_list()
        self.eval_goal_list = self.get_eval_goal_list()
        self.eval_start_count = 0
        self.eval_count = 0
        # goes through hardcoded eval goal values in order rather than randomly choosing
        self.sequential_eval = seq_eval

    def get_start_list(self):
        return [
            np.array([2.04097663, -1.65100083, 3.25]),
            np.array([4.10744807, 7.23903795, 3.25]),
            np.array([2.21298789, -1.77792618, 3.25]),
            np.array([2.64859868, -2.20080816, 3.25]),
            np.array([-4.34777044, 6.33049147, 3.25]),
            np.array([-4.44752978, -1.9291824, 3.25]),
            np.array([-4.77374649, -2.92058681, 3.25]),
            np.array([4.13839802, -7.93056262, 3.25]),
            np.array([-3.6339912, -3.78330752, 3.25]),
            np.array([1.67778578, -1.07522108, 3.25]),
            np.array([0.72958577, 5.62491041, 3.25]),
            np.array([5.50109756, 1.60540866, 3.25]),
            np.array([-4.84623381, -1.91969908, 3.25]),
            np.array([-2.64804082, 0.25612383, 3.25]),
            np.array([-6.75746318, -0.83377183, 3.25]),
            np.array([-3.81740024, -3.40561904, 3.25]),
            np.array([-5.17478337, -6.27176169, 3.25]),
            np.array([-3.94487777, 1.97520862, 3.25]),
            np.array([-2.18946437, 3.83887034, 3.25]),
            np.array([-8.45837181, 1.69275326, 3.25]),
            np.array([-6.47569175, -2.13925613, 3.25]),
            np.array([-3.10575608, -5.55079685, 3.25]),
            np.array([3.46376521, -1.65422878, 3.25]),
            np.array([-0.0720884, 6.85211944, 3.25]),
            np.array([0.23277159, 6.48953965, 3.25]),
            np.array([0.35250774, 7.69888375, 3.25]),
            np.array([-1.53017535, -3.94902122, 3.25]),
            np.array([5.46574845, -4.00952579, 3.25]),
            np.array([-6.32064986, -2.72127592, 3.25]),
            np.array([1.09125718, -4.08004056, 3.25]),
            np.array([-3.6541273, 4.97720398, 3.25]),
            np.array([6.11267395, 6.43009359, 3.25]),
            np.array([0.69486026, -8.91990217, 3.25]),
            np.array([2.60528523, 4.81703968, 3.25]),
            np.array([-1.92844214, -2.97537717, 3.25]),
            np.array([-5.35549988, -4.30591255, 3.25]),
            np.array([-5.57041867, 6.64359229, 3.25]),
            np.array([-5.87918698, 5.4926953, 3.25]),
            np.array([-0.64131894, 6.00955903, 3.25]),
            np.array([-2.48863439, -0.31338188, 3.25]),
            np.array([-0.02733371, -3.19647573, 3.25]),
            np.array([-4.4459109, 7.33152599, 3.25]),
            np.array([-2.58218984, -0.85153104, 3.25]),
            np.array([-0.53642423, -2.85615106, 3.25]),
            np.array([-1.94631083, 3.88030117, 3.25]),
            np.array([4.53668622, -5.11221288, 3.25]),
            np.array([-2.77463316, 0.71408483, 3.25]),
            np.array([-2.8336516, -3.67925051, 3.25]),
            np.array([-0.45671894, 4.32993726, 3.25]),
            np.array([2.79136047, 7.29243927, 3.25]),
            np.array([-0.6892756, 3.96817383, 3.25]),
            np.array([4.99552183, 3.56101594, 3.25]),
            np.array([5.16958045, -7.02891967, 3.25]),
            np.array([1.23990442, -1.38083498, 3.25]),
            np.array([5.92869115, 6.2522862, 3.25]),
            np.array([-3.14521847, -8.13946438, 3.25]),
            np.array([2.9719716, -6.96319138, 3.25]),
            np.array([5.07185006, -1.16377918, 3.25]),
            np.array([1.66742066, 4.02562049, 3.25]),
            np.array([1.77176953, 3.41187981, 3.25]),
            np.array([-0.13260779, -2.68537634, 3.25]),
            np.array([4.33229546, -0.03551759, 3.25]),
            np.array([-1.43365107, -1.84130095, 3.25]),
            np.array([-2.92969646, 5.75421449, 3.25]),
            np.array([1.11222653, 3.18992928, 3.25]),
            np.array([5.25777992, -3.84619755, 3.25]),
            np.array([-5.07620368, -5.58340159, 3.25]),
            np.array([-3.05283113, -7.62402811, 3.25]),
            np.array([1.23449075, 0.44386378, 3.25]),
            np.array([-2.03197261, 5.92553343, 3.25]),
            np.array([-1.00614565, 1.65717695, 3.25]),
            np.array([6.94632315, 3.60978841, 3.25]),
            np.array([-3.53368917, 8.10776891, 3.25]),
            np.array([0.2204234, 5.20549202, 3.25]),
            np.array([-5.29871847, -2.50313875, 3.25]),
            np.array([-1.18429566, -3.25836533, 3.25]),
            np.array([7.021721, -1.37745048, 3.25]),
            np.array([-4.61213103, -3.81696923, 3.25]),
            np.array([-1.80475419, -2.29072473, 3.25]),
            np.array([-7.17524205, -0.65156247, 3.25]),
            np.array([-4.55399435, -3.30533432, 3.25]),
            np.array([-0.05460599, -5.58954694, 3.25]),
            np.array([4.19168691, -7.49274173, 3.25]),
            np.array([4.84372648, 4.82713899, 3.25]),
            np.array([6.63102781, 5.70623944, 3.25]),
            np.array([7.59700729, -0.83047598, 3.25]),
            np.array([4.46110769, 4.83956357, 3.25]),
            np.array([-4.6037906, 0.19172261, 3.25]),
            np.array([-7.18088318, -1.33220808, 3.25]),
            np.array([1.06310965, 2.41328782, 3.25]),
            np.array([-0.49105523, -1.11458754, 3.25]),
            np.array([0.01794725, 3.06635785, 3.25]),
            np.array([-5.38248375, 1.22571585, 3.25]),
            np.array([-4.5219725, -5.00797691, 3.25]),
            np.array([1.64514413, 4.37356647, 3.25]),
            np.array([-2.13024822, 0.58961604, 3.25]),
            np.array([-1.91045255, 2.92433814, 3.25]),
            np.array([5.69786521, -3.72389571, 3.25]),
            np.array([-4.26038794, -0.25427055, 3.25]),
            np.array([-3.73057202, -7.6881122, 3.25]),
        ]

    def get_eval_start_list(self):
        return [
            np.array([6.67169073, -2.96553179, 3.25]),
            np.array([4.53332389, 2.98308279, 3.25]),
            np.array([-2.91775021, 2.57252752, 3.25]),
            np.array([8.93065598, -0.15437427, 3.25]),
            np.array([-8.19208537, -1.94309468, 3.25]),
            np.array([1.8349047, -4.78840247, 3.25]),
            np.array([-0.29920792, 5.39048065, 3.25]),
            np.array([8.02817476, -2.77101145, 3.25]),
            np.array([6.75243009, -5.60007531, 3.25]),
            np.array([-1.2305441, -1.93330211, 3.25]),
            np.array([-4.16567822, 4.60573848, 3.25]),
            np.array([1.68092937, -0.61479163, 3.25]),
            np.array([-1.93641802, -2.23759902, 3.25]),
            np.array([3.75552483, 4.99247274, 3.25]),
            np.array([-2.63227948, 1.02710679, 3.25]),
            np.array([-0.73785682, 6.72614777, 3.25]),
            np.array([-5.98990161, 1.40376386, 3.25]),
            np.array([-4.14701302, -7.64395404, 3.25]),
            np.array([-2.68738883, -0.86268445, 3.25]),
            np.array([3.56820047, -4.01970462, 3.25]),
            np.array([4.73531203, -7.38510796, 3.25]),
            np.array([4.54052887, -1.01960825, 3.25]),
            np.array([-8.56401189, 0.82893131, 3.25]),
            np.array([-3.23477287, -5.12156484, 3.25]),
            np.array([-3.8107995, 2.98017638, 3.25]),
        ]

    def get_goal_list(self):
        return [
            np.array([-6.85620415, 5.69309662, 3.4868318]),
            np.array([6.89543661, 0.20638839, 3.60589458]),
            np.array([-4.76185274, -3.57138597, 3.48521864]),
            np.array([4.48172165, 2.30776027, 3.40143134]),
            np.array([-10.097758, -2.05704158, 3.50874507]),
            np.array([-6.21063435, 5.96678709, 3.49914875]),
            np.array([-0.85843888, 0.26477303, 3.51648274]),
            np.array([-1.53639816, -1.34207088, 3.35050419]),
            np.array([2.4713391, -8.3362068, 3.40881575]),
            np.array([1.76395876e-03, 1.59974155e00, 3.34845197e00]),
            np.array([-2.44383359, 7.52655064, 3.34270859]),
            np.array([1.09045117, -1.26148746, 3.45028295]),
            np.array([4.2388288, 8.1671043, 3.42516367]),
            np.array([1.88647559, -7.03245503, 3.4258199]),
            np.array([0.11318267, 2.57698791, 3.44239848]),
            np.array([4.10511002, 2.40155972, 3.55802448]),
            np.array([-0.23120615, -1.45758424, 3.47215934]),
            np.array([-3.05966982, 10.02575994, 3.34350474]),
            np.array([1.73366214, 10.70642224, 3.43047809]),
            np.array([1.68763431, -0.56803548, 3.39711601]),
            np.array([-9.77245964, -1.42591748, 3.34540121]),
            np.array([-3.71715436, 0.15941034, 3.33814527]),
            np.array([0.89186381, -10.34613863, 3.544193]),
            np.array([-0.57973103, 10.59727006, 3.38286818]),
            np.array([-10.70692197, 0.85174816, 3.48813104]),
            np.array([3.74088445, -4.07057836, 3.58707664]),
            np.array([-6.51509437, 3.33729785, 3.41168711]),
            np.array([9.92651822, -5.09583286, 3.3516998]),
            np.array([-9.71215617, 0.43383868, 3.3529111]),
            np.array([-7.48044561, -7.8204012, 3.35138153]),
            np.array([-6.77449691, -2.21448351, 3.4748631]),
            np.array([5.24973063, 7.75546124, 3.39087428]),
            np.array([5.7441052, -9.48213409, 3.44377653]),
            np.array([-1.65363983, 6.93396322, 3.34352824]),
            np.array([1.72672181, -2.20423246, 3.34493667]),
            np.array([-6.32620696, -6.15006496, 3.34785745]),
            np.array([-7.25481784, -2.84468915, 3.40973936]),
            np.array([3.48910405, 0.27649298, 3.33779743]),
            np.array([-7.29880413, -1.67084031, 3.47002878]),
            np.array([-5.39445235, 5.24321575, 3.34222376]),
            np.array([3.27466144, 0.63430133, 3.39329086]),
            np.array([1.84325319, 6.99002939, 3.36439045]),
            np.array([-6.83167302, -5.41291579, 3.36950817]),
            np.array([-0.91039109, -0.63790262, 3.34861123]),
            np.array([6.51689054, 1.39720148, 3.44225852]),
            np.array([-4.96093917, -6.83616067, 3.46017926]),
            np.array([1.84286209, 2.71032173, 3.33851569]),
            np.array([-9.25094037, -2.60808305, 3.34171691]),
            np.array([-4.39315839, 5.4206937, 3.34240775]),
            np.array([7.79844963, 1.21241137, 3.54043111]),
            np.array([7.9784517, -1.04042639, 3.35562883]),
            np.array([9.74992113, -0.05703117, 3.34283087]),
            np.array([-1.80699541, 0.62056611, 3.52324641]),
            np.array([-3.33449495, -3.27455263, 3.35692825]),
            np.array([1.91787857, -1.55797992, 3.49959849]),
            np.array([-8.18887959, -6.95140586, 3.34517562]),
            np.array([-6.55092508, -5.36105026, 3.51953136]),
            np.array([-4.65692181, 5.00578746, 3.57180856]),
            np.array([-10.01640723, 0.09701515, 3.47691971]),
            np.array([6.08581384, -6.55555138, 3.51073652]),
            np.array([6.37559629, -1.39824096, 3.38839112]),
            np.array([4.22314207, 10.60955302, 3.40776734]),
            np.array([10.49006752, -0.25291699, 3.6091191]),
            np.array([-5.5563258, -0.45557905, 3.56926722]),
            np.array([-0.59690022, 0.23712072, 3.34728676]),
            np.array([1.54201962, 0.53821618, 3.41215915]),
            np.array([3.91624165, -3.5208636, 3.65523469]),
            np.array([-9.66192239, -1.57394663, 3.35618386]),
            np.array([-10.28422427, 3.20408299, 3.52148926]),
            np.array([-5.86194317, 7.78183548, 3.34852961]),
            np.array([-3.41343808, 10.86818437, 3.35983464]),
            np.array([10.88775929, 1.83811875, 3.36764426]),
            np.array([1.64951292, 7.73225581, 3.35893576]),
            np.array([-3.87361636, 10.68113917, 3.38532573]),
            np.array([-10.40482953, -2.83170933, 3.36578927]),
            np.array([1.61077724, 4.92156534, 3.33918436]),
            np.array([0.17828444, -5.5765294, 3.34321059]),
            np.array([5.167725, -1.28080891, 3.64031652]),
            np.array([-8.55232423, 1.28828846, 3.37625187]),
            np.array([-9.78914147, -4.66853043, 3.40276421]),
            np.array([-5.83961344, -0.53358555, 3.34591576]),
            np.array([7.90392253, 4.11711935, 3.54403815]),
            np.array([0.77248579, -5.16369315, 3.34268256]),
            np.array([1.58398011, 0.66349796, 3.34651256]),
            np.array([2.96027527, -3.30153252, 3.52695208]),
            np.array([-3.32688568, -5.9893656, 3.38640985]),
            np.array([-2.38823957, 1.22484347, 3.51193319]),
            np.array([-0.67132962, 9.86015055, 3.41217951]),
            np.array([-0.67080763, -6.43749339, 3.35517908]),
            np.array([-5.63190129, -6.7793298, 3.5780783]),
            np.array([-3.90313746, 9.41344458, 3.4665348]),
            np.array([-4.88213205, 6.32855783, 3.35855582]),
            np.array([-8.22583522, 4.5383908, 3.34817245]),
            np.array([-0.62195955, 0.33415983, 3.33682747]),
            np.array([10.65720498, 3.41036641, 3.50191099]),
            np.array([-3.30091672, 9.51880107, 3.47082805]),
            np.array([-10.51216611, 3.15678105, 3.42093078]),
            np.array([6.69407137, -0.58780311, 3.35043057]),
            np.array([-6.7290203, -8.85983436, 3.54240275]),
            np.array([6.44124682, -4.54900372, 3.50910745]),
        ]

    def get_eval_goal_list(self):
        return [
            np.array([5.14456575, 9.53934744, 3.4736776]),
            np.array([9.47314765, -1.05800597, 3.38940632]),
            np.array([-7.89212926, 5.73950083, 3.34253909]),
            np.array([5.25182976, -0.06633719, 3.34521151]),
            np.array([-11.45348978, -1.08593413, 3.34631526]),
            np.array([-2.49468065, -10.71326428, 3.42221313]),
            np.array([-7.46557298, -5.45556846, 3.6661241]),
            np.array([8.33472767, -7.27369026, 3.34793479]),
            np.array([-6.54476041, -6.11756091, 3.61223536]),
            np.array([-10.06022672, -2.42743655, 3.36778525]),
            np.array([-2.85501714, -2.09537331, 3.55102278]),
            np.array([-2.34413951, -6.80405336, 3.38061399]),
            np.array([-6.53886547, -2.29299191, 3.37285069]),
            np.array([-1.87206664, 1.74855269, 3.34257076]),
            np.array([6.5523002, -6.84960049, 3.45466889]),
            np.array([7.64386918, 5.86611545, 3.42190653]),
            np.array([-1.29261219, 7.50578918, 3.41643612]),
            np.array([-0.46343966, -3.91072315, 3.4125123]),
            np.array([6.85678941, 2.0588009, 3.58958953]),
            np.array([-3.10926912, -2.49296228, 3.43359971]),
            np.array([-7.3301309, -5.06979915, 3.39754574]),
            np.array([-7.61911634, -6.00939488, 3.57340908]),
            np.array([-2.88103846, 10.77367604, 3.34477527]),
            np.array([1.11187448, 4.50634239, 3.39748213]),
            np.array([-5.39123021, 9.35176932, 3.3435149]),
        ]

    def _get_fingertip_pos(self, t):
        # r_obs = self.platform.get_robot_observation(t)
        r_obs = self.platform.simfinger.get_observation(t)
        # pass joint pos for xyz coordinates
        return self.hand_kinematics.get_ft_pos(r_obs.position)

    def compute_reward(
        self,
        fingertip_pos,
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
        # reward wrt xy position, not taking z into account
        arena_radius = 0.195
        xy_dist = np.linalg.norm(desired_goal[:2] - achieved_goal[:2])
        scaled_dist = xy_dist / (2 * arena_radius)
        start_scaled_dist = xy_dist / self.start_dist
        reward = 0
        # additional reward the closer the fingertips are to the cube
        ftip_dist_to_cube = 0
        for i in range(3):
            ftip_dist_to_cube += np.linalg.norm(
                fingertip_pos[(3 * i) : (3 * i) + 3] - achieved_goal[0:3]
            )
        if ftip_dist_to_cube < 0.15:
            reward += 50.0
            for i in range(3):
                ftip_dist_to_target = np.linalg.norm(
                    fingertip_pos[(3 * i) : (3 * i) + 3] - desired_goal[0:3]
                )
            cube_dist_to_target = np.linalg.norm(achieved_goal[0:3] - desired_goal[0:3])
            reward += -200 * cube_dist_to_target + -10 * ftip_dist_to_target

        reward += -100 * (ftip_dist_to_cube) + 50
        self.prev_finger_dist = ftip_dist_to_cube

        if xy_dist < 0.07:
            reward += 10
        if xy_dist < 0.04:
            reward += 20
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

    def render(self, mode="human"):
        camera_observation = None
        cam_imgs = self.platform.get_camera_observation(self.step_count)
        if mode == "rgb_array":
            # 0:camera 60, 1:camera180, 2:camera300
            if self.camera_id == -1:  # return ALL images
                camera_observation = np.concatenate(
                    (cam_imgs[0].image, cam_imgs[1].image, cam_imgs[2].image), axis=2
                )
            else:
                camera_observation = torch.tensor(
                    cam_imgs.cameras[self.camera_id].image
                )
        elif mode == "eval":
            camera_observation = cam_imgs.cameras[0].image
        else:
            raise NotImplementedError

        return camera_observation

    def get_success(self, curr, goal):
        success = 1 - (np.linalg.norm(curr - goal) / self.start_dist)
        if success < 0:
            success = 0
        return success

    def get_success_reach(self, curr, goal):
        dist = 0
        for i in range(3):
            dist += np.linalg.norm(curr[(3 * i) : (3 * i) + 3] - goal[0:3])
        success = 1 - (dist / self.start_finger_dist)
        if success < 0:
            success = 0
        return success

    def _create_observation(self, t):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)
        object_observation = camera_observation.filtered_object_pose

        if self.visual_observation:
            observation = {
                "t": t,
                "pixels": self.render("rgb_array"),
                "state_obs": np.concatenate(
                    (
                        robot_observation.position,
                        object_observation.position,
                        object_observation.orientation,
                        self.goal,
                    ),
                    axis=0,
                ),
                "scaled_success": self.get_success(
                    object_observation.position, self.goal[0:3]
                ),
                "scaled_success_reach": self.get_success_reach(
                    self.hand_kinematics.get_ft_pos(robot_observation.position),
                    object_observation.position,
                ),
            }
        else:
            observation = {
                "t": t,
                "state_obs": np.concatenate(
                    (
                        robot_observation.position,
                        object_observation.position,
                        object_observation.orientation,
                        self.goal,
                    ),
                    axis=0,
                ),
            }

        if not self.run_rl_policy:
            position_error = np.linalg.norm(
                object_observation.position - self._goal_pos()
            )
            # Get cube vertices
            obj_pose = {
                "position": object_observation.position,
                "orientation": object_observation.orientation,
            }

            # From trifinger_simulation tasks/move_cube/__init__.py evaluate_state()
            goal_rot = Rotation.from_quat(self._goal_orientation())
            actual_rot = Rotation.from_quat(object_observation.orientation)
            error_rot = goal_rot.inv() * actual_rot
            orientation_error = error_rot.magnitude()

            # Add new observation fields
            ft_pos_cur = self.hand_kinematics.get_ft_pos(robot_observation.position)
            v_wf_dict = c_utils.get_vertices_wf(obj_pose)
            observation["robot_position"] = robot_observation.position
            observation["object_position"] = object_observation.position
            observation["object_orientation"] = object_observation.orientation
            observation["object_vertices"] = v_wf_dict
            observation["desired_goal"] = self.goal
            observation["achieved_goal_position_error"] = position_error
            observation["achieved_goal_orientation_error"] = orientation_error
            observation["ft_pos_cur"] = ft_pos_cur
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

    def close(self):
        pybullet.disconnect()
        super().close()


class MoveCubeEnv(BaseCubeEnv):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        goal_pose: dict = None,
        action_type: ActionType = ActionType.TORQUE,
        step_size: int = 100,
        difficulty: int = 1,
        visualization: bool = False,
        goal_visualization: bool = False,
        no_collisions: bool = False,
        enable_cameras: bool = False,
        finger_type: str = "trifinger_meta",
        camera_delay_steps: int = 90,
        time_step: float = 0.004,
        object_type: ObjectType = ObjectType.COLORED_CUBE,
        enable_shadows: bool = False,
        camera_view: str = "default",
        arena_color: str = "default",
        random_q_init: bool = False,
        visual_observation: bool = False,
        run_rl_policy: bool = True,
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
            run_rl_policy (bool): If true, don't add extra observation fields used for bc policy
        """
        super().__init__(
            goal_pose=goal_pose,
            action_type=action_type,
            step_size=step_size,
            difficulty=difficulty,
            visual_observation=visual_observation,
        )

        self.visualization = visualization
        self.goal_visualization = goal_visualization
        self.no_collisions = no_collisions
        self.enable_cameras = enable_cameras
        self.finger_type = finger_type
        self.time_step = time_step
        self.enable_shadows = enable_shadows
        self.camera_view = camera_view
        self.random_q_init = random_q_init
        self.visual_observation = visual_observation
        self.run_rl_policy = run_rl_policy

        if self.visual_observation:
            self.enable_cameras = True

        # initialize simulation
        # initial_robot_position = trifingerpro_limits.robot_position.default
        self.q_nominal = np.array([-0.08, 1.15, -1.5] * 3)

        if self.random_q_init:
            self.initial_robot_position = self.sample_init_robot_position()
        else:
            self.initial_robot_position = np.array(
                [
                    -0.0809731,
                    1.1499023,
                    -1.50172085,
                    -0.08046894,
                    1.14986721,
                    -1.50067745,
                    -0.07987084,
                    1.14964149,
                    -1.50124104,
                ]
            )

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
            fix_cube_base=False,
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
            self.goal_visualization = True

        else:
            self.tricamera = None

        # visualize the cube vertices
        if self.visualization and not self.enable_cameras:
            self.draw_verts = True
        else:
            self.draw_verts = False
        self.vert_markers = None
        self.start_dist = 1000

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
        """
        Run one timestep of the environment's dynamics, which is
        self.step_size number of simulation steps

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
            action = np.clip(action, -0.02, 0.02)

        if not self.action_space.contains(np.array(action, dtype=np.float32)):
            raise ValueError("Given action is not contained in the action space.")

        num_steps = self.step_size
        # ensure episode length is not exceeded due to step_size
        step_count_after = self.step_count + num_steps
        if step_count_after > MOVE_CUBE_EPISODE:
            excess = step_count_after - MOVE_CUBE_EPISODE
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _i in range(num_steps):
            # Get current robot state
            robot_obs = self.platform.get_robot_observation(self.step_count)
            joint_position = robot_obs.position
            joint_velocity = robot_obs.velocity

            self.step_count += 1
            if self.step_count > MOVE_CUBE_EPISODE:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # Update desired position and velocity
            x_des_i = self.x_des_plan + (_i + 1) * (action / num_steps)
            dx_des_i = action / (self.step_size * self.time_step)

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

        # visualize the goal
        if self.goal_visualization:
            self.goal_marker = trifinger_simulation.visual_objects.CubeMarker(
                width=task._CUBE_WIDTH,
                position=self._goal_pos(),
                orientation=self._goal_orientation(),
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )
        observation = self._create_observation(self.info["time_index"])

        # Update plan with action
        self.x_des_plan += action

        # Compute reward
        reward = 0
        reward += self.compute_reward(
            self.hand_kinematics.get_ft_pos(observation["state_obs"][:9]),
            observation["state_obs"][9:16],
            observation["state_obs"][16:],
            self.info,
        )

        # Draw cube vertices from observation
        if self.draw_verts:
            v_wf_dict = observation["object_observation"]["vertices"]
            positions = [v_wf for k, v_wf in v_wf_dict.items()]
            self.vert_markers.set_state(positions)

        is_done = self.step_count >= MOVE_CUBE_EPISODE

        return observation, reward, is_done, self.info

    def choose_start_from_demos(self, eval=False):
        start_pos_list = self.train_start_list
        if eval:
            self.eval_start_count += 1
            if self.eval_start_count == len(self.eval_start_list):
                self.eval_start_count = 0
            start_pos_list = self.eval_start_list
        else:
            start_pos_list = self.train_start_list

        if self.sequential_eval and eval:
            idx = self.eval_start_count
        else:
            idx = np.random.randint(0, len(start_pos_list))
        return start_pos_list[idx] / 100.0

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
        return goal_pos_list[idx] / 100.0

    def reset(
        self,
        goal_pose_dict=None,
        init_pose_dict=None,
        init_robot_position=None,
        random_init_cube_pos=False,
        eval_mode=False,
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
            if self.run_rl_policy:
                initial_object_pose.position = self.choose_start_from_demos(
                    eval=eval_mode
                )
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
            if self.run_rl_policy:
                self.goal[0:3] = self.choose_goal_from_demos(eval=eval_mode)
        else:
            pose = goal_pose_dict
            self.goal = np.append(pose["position"], pose["orientation"])

        # visualize the goal
        if self.goal_visualization:
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

        # Reset state for policy execution

        self.x_des_plan = torch.FloatTensor(
            self.hand_kinematics.get_ft_pos(init_robot_position).copy()
        )

        self.info = {"time_index": -1, "goal": self.goal, "difficulty": self.difficulty}

        self.step_count = 0

        self.start_dist = np.linalg.norm(initial_object_pose.position - self.goal[0:3])
        self.start_finger_dist = 0
        for i in range(3):
            self.start_finger_dist += np.linalg.norm(
                self.x_des_plan[(3 * i) : (3 * i) + 3]
                - initial_object_pose.position[0:3]
            )
        self.prev_finger_dist = self.start_finger_dist

        self.platform._camera_update()
        new_obs = self._create_observation(0)

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
