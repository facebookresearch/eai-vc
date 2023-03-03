#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import numpy as np
import quaternion
from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationEpisode, Success, DistanceToGoal
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_coeff,
)
from habitat.tasks.nav.object_nav_task import ObjectGoal


@registry.register_measure
class AngleToGoal(Measure):
    """The measure calculates an angle towards the goal. Note: this measure is
    only valid for single goal tasks (e.g., ImageNav)
    """

    cls_uuid: str = "angle_to_goal"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        super().__init__()
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = None
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode: NavigationEpisode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        current_rotation = self._sim.get_agent_state().rotation
        if not isinstance(current_rotation, quaternion.quaternion):
            current_rotation = quaternion_from_coeff(current_rotation)

        assert len(episode.goals) > 0, "Episode must have goals"

        is_semantic_nav = isinstance(episode.goals[0], ObjectGoal)
        if not is_semantic_nav:
            goal_rotation = episode.goals[0].rotation
        else:
            # Hack to save time. We dont need to calculate the angle to goal if we are outside the goal radius
            if task.measurements.measures[DistanceToGoal.cls_uuid].get_metric() > 0.1:
                self._metric = np.pi
                return

            current_position = self._sim.get_agent_state().position

            nearest_goal = self.get_closest_goal(episode, current_position)

            # find angle between current_position and nearest_goal position
            goal_vector = nearest_goal.position - current_position
            goal_angle = np.arctan2(goal_vector[2], goal_vector[0])
            goal_rotation = quaternion.from_rotation_vector([0, goal_angle, 0])

        if not isinstance(goal_rotation, quaternion.quaternion):
            goal_rotation = quaternion_from_coeff(goal_rotation)

        self._metric = angle_between_quaternions(current_rotation, goal_rotation)

    def get_closest_goal(self, episode, agent_position):
        min_dist = float("inf")
        closest_goal = None
        for goal in episode.goals:
            # snapped_point = self._sim.path_finder.snap_point(goal.position)
            euclidean_dist = np.linalg.norm(
                np.array(agent_position) - np.array(goal.position)
            )
            if euclidean_dist >= min_dist:
                continue
            distance = self._sim.geodesic_distance(
                agent_position,
                [goal.position],
                episode,
            )
            if distance < min_dist:
                closest_goal = goal
                min_dist = distance
        return closest_goal


@registry.register_measure
class AngleSuccess(Measure):
    """Weather or not the agent is within an angle tolerance."""

    cls_uuid: str = "angle_success"

    def __init__(self, config: Config, *args: Any, **kwargs: Any):
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        dependencies = [AngleToGoal.cls_uuid]
        if self._config.USE_TRAIN_SUCCESS:
            dependencies.append(TrainSuccess.cls_uuid)
        else:
            dependencies.append(Success.cls_uuid)
        task.measurements.check_measure_dependencies(self.uuid, dependencies)
        self.update_metric(task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        if self._config.USE_TRAIN_SUCCESS:
            success = task.measurements.measures[TrainSuccess.cls_uuid].get_metric()
        else:
            success = task.measurements.measures[Success.cls_uuid].get_metric()
        angle_to_goal = task.measurements.measures[AngleToGoal.cls_uuid].get_metric()

        if success and np.rad2deg(angle_to_goal) < self._config.SUCCESS_ANGLE:
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class TrainSuccess(Success):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "train_success"

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called  # type: ignore
            and distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0
