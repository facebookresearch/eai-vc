#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, List

import attr
import os

from habitat.tasks.nav.nav import NavigationTask, NavigationEpisode
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator


@attr.s(auto_attribs=True, kw_only=True)
class AgentStateSpec:
    r"""Agent data specifications that capture states of agent and sensor in replay state."""
    position: Optional[List[float]] = attr.ib(default=None)
    rotation: Optional[List[float]] = attr.ib(default=None)
    sensor_data: Optional[dict] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ReplayActionSpec:
    r"""Replay specifications that capture metadata associated with action."""
    action: str = attr.ib(default=None, validator=not_none_validator)
    agent_state: Optional[AgentStateSpec] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode
    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None
    reference_replay: Optional[List[ReplayActionSpec]] = None
    scene_state = None
    is_thda: Optional[bool] = False
    scene_dataset: Optional[str] = "mp3d"
    scene_dataset_config: Optional[str] = ""
    additional_obj_config_paths: Optional[List] = []
    attempts: Optional[int] = 1

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@registry.register_task(name="ObjectNav-v2")
class ObjectNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
    _is_episode_active: bool
    _prev_action: int
    _is_resetting: bool

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._is_episode_active = False
        self._is_resetting = False

    def reset(self, episode):
        self._is_resetting = True
        obs = super().reset(episode)
        self._is_resetting = False
        return obs
