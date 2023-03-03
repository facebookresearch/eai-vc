#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from gym import spaces

from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def get_habitat_sim_action(action):
    if action == "TURN_RIGHT":
        return HabitatSimActions.TURN_RIGHT
    elif action == "TURN_LEFT":
        return HabitatSimActions.TURN_LEFT
    elif action == "MOVE_FORWARD":
        return HabitatSimActions.MOVE_FORWARD
    elif action == "LOOK_UP":
        return HabitatSimActions.LOOK_UP
    elif action == "LOOK_DOWN":
        return HabitatSimActions.LOOK_DOWN
    return HabitatSimActions.STOP


@registry.register_sensor(name="DemonstrationSensor")
class DemonstrationSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "demonstration"
        self.observation_space = spaces.Discrete(1)
        self.timestep = 0
        self.prev_action = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs
    ):
        # Fetch next action as observation
        if task._is_resetting:  # reset
            self.timestep = 1

        if self.timestep < len(episode.reference_replay):
            action_name = episode.reference_replay[self.timestep].action
            action = get_habitat_sim_action(action_name)
        else:
            action = 0

        self.timestep += 1
        return action

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_sensor(name="InflectionWeightSensor")
class InflectionWeightSensor(Sensor):
    def __init__(self, config: Config, **kwargs):
        self.uuid = "inflection_weight"
        self.observation_space = spaces.Discrete(1)
        self._config = config
        self.timestep = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs
    ):
        if task._is_resetting:  # reset
            self.timestep = 0

        inflection_weight = 1.0
        if self.timestep == 0:
            inflection_weight = 1.0
        elif self.timestep >= len(episode.reference_replay):
            inflection_weight = 1.0
        elif (
            episode.reference_replay[self.timestep - 1].action
            != episode.reference_replay[self.timestep].action
        ):
            inflection_weight = self._config.INFLECTION_COEF
        self.timestep += 1
        return inflection_weight

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)
