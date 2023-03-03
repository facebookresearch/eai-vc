#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import numpy as np
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import RGBSensor, Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import NavigationEpisode


# fmt: off
@registry.register_sensor
class ImageGoalRotationSensor(Sensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.
    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "imagegoalrotation"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                f"ImageGoalNav requires one RGB sensor, {len(rgb_sensor_uuids)} detected"
            )

        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._rgb_sensor_uuid
        ]

    def _get_pointnav_episode_image_goal(self, episode: NavigationEpisode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        # Add rotation to episode ** NEW **
        if self.config.SAMPLE_ANGLE == True:
            angle = np.random.uniform(0, 2 * np.pi)
        else:
            # to be sure that the rotation is the same for the same episode_id
            # since the task is currently using pointnav Dataset.
            seed = abs(hash(episode.episode_id)) % (2**32)
            rng = np.random.RandomState(seed)
            angle = rng.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        episode.goals[0].rotation = source_rotation

        goal_observation = self._sim.get_observations_at(
            position=goal_position.tolist(), rotation=source_rotation
        )
        return goal_observation[self._rgb_sensor_uuid]

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_pointnav_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal
