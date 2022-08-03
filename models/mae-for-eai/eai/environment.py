from typing import Optional

import habitat
import numpy as np
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_env(name="SimpleRLEnv")
class SimpleRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)
        self.config = config

    def get_reward_range(self):
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        return self._env.get_metrics()[self.config.RL.REWARD_MEASURE]

    def get_done(self, observations):
        if self._env.episode_over:
            return True
        if self._env.get_metrics()[self.config.RL.SUCCESS_MEASURE]:
            return True
        return False

    def get_info(self, observations):
        return self._env.get_metrics()
