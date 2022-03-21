from typing import Any, Optional

from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import DistanceToGoal, Success


@registry.register_measure
class SimpleReward(Measure):
    cls_uuid: str = "simple_reward"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config
        self._previous_dtg: Optional[float] = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self,
        *args: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ):
        task.measurements.check_measure_dependencies(self.uuid, [Success.cls_uuid])
        if self._config.USE_DTG_REWARD:
            task.measurements.check_measure_dependencies(
                self.uuid, [DistanceToGoal.cls_uuid]
            )
        self._metric = 0.0
        self._previous_dtg = None
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        success = task.measurements.measures[Success.cls_uuid].get_metric()
        dtg = None
        if self._config.USE_DTG_REWARD:
            dtg = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()

        if self._previous_dtg is None:
            self._previous_dtg = dtg

        success_reward = self._config.SUCCESS_REWARD if success else 0.0
        reward_shaping = 0.0
        if self._config.USE_DTG_REWARD:
            reward_shaping = self._previous_dtg - dtg
        slack_penalty = self._config.SLACK_PENALTY

        self._metric = success_reward + reward_shaping + slack_penalty

        self._previous_dtg = dtg
