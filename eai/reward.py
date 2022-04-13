from typing import Any, Optional

from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import DistanceToGoal, Success

from eai.measures import AngleSuccess, AngleToGoal


@registry.register_measure
class SimpleReward(Measure):
    cls_uuid: str = "simple_reward"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config
        self._previous_dtg: Optional[float] = None
        self._previous_atg: Optional[float] = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self,
        *args: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                Success.cls_uuid,
                DistanceToGoal.cls_uuid,
                AngleToGoal.cls_uuid,
                AngleSuccess.cls_uuid,
            ],
        )
        self._metric = None
        self._previous_dtg = None
        self._previous_atg = None
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        # success
        success = task.measurements.measures[Success.cls_uuid].get_metric()
        success_reward = self._config.SUCCESS_REWARD if success else 0.0

        # distance-to-goal
        dtg = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        if self._previous_dtg is None:
            self._previous_dtg = dtg
        add_dtg = self._config.USE_DTG_REWARD
        dtg_reward = self._previous_dtg - dtg if add_dtg else 0.0
        self._previous_dtg = dtg

        # angle-to-goal
        atg = task.measurements.measures[AngleToGoal.cls_uuid].get_metric()
        if self._previous_atg is None:
            self._previous_atg = atg
        add_atg = (
            self._config.USE_ATG_REWARD and dtg <= self._config.ATG_REWARD_DISTANCE
        )
        atg_reward = self._previous_atg - atg if add_atg else 0.0
        self._previous_atg = atg

        # angle success
        angle_success = task.measurements.measures[AngleSuccess.cls_uuid].get_metric()
        angle_success_reward = (
            self._config.ANGLE_SUCCESS_REWARD if angle_success else 0.0
        )

        # slack penalty
        slack_penalty = self._config.SLACK_PENALTY

        self._metric = (
            success_reward
            + dtg_reward
            + atg_reward
            + angle_success_reward
            + slack_penalty
        )
