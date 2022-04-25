from typing import Any

import numpy as np
import quaternion
from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationEpisode, Success
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_coeff,
)


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

    def update_metric(self, episode: NavigationEpisode, *args: Any, **kwargs: Any):
        current_rotation = self._sim.get_agent_state().rotation
        if not isinstance(current_rotation, quaternion.quaternion):
            current_rotation = quaternion_from_coeff(current_rotation)

        goal_rotation = episode.goals[0].rotation
        if not isinstance(goal_rotation, quaternion.quaternion):
            goal_rotation = quaternion_from_coeff(goal_rotation)

        self._metric = angle_between_quaternions(current_rotation, goal_rotation)


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
        task.measurements.check_measure_dependencies(
            self.uuid, [Success.cls_uuid, AngleToGoal.cls_uuid]
        )
        self.update_metric(task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        success = task.measurements.measures[Success.cls_uuid].get_metric()
        angle_to_goal = task.measurements.measures[AngleToGoal.cls_uuid].get_metric()

        if success and np.rad2deg(angle_to_goal) < self._config.SUCCESS_ANGLE:
            self._metric = 1.0
        else:
            self._metric = 0.0
