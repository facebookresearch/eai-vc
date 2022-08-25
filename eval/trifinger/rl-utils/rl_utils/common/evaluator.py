import os
import os.path as osp
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from rl_utils.common.core_utils import compress_and_filter_dict
from rl_utils.common.viz_utils import save_mp4
from rl_utils.envs.vec_env.vec_env import VecEnv
from rl_utils.interfaces import BasePolicy


class Evaluator:
    """
    Dataset save format is meant to be consistent with https://github.com/rail-berkeley/d4rl/blob/master/d4rl/offline_env.py
    """

    def __init__(
        self,
        envs: VecEnv,
        rnn_hxs_dim: int,
        num_render: Optional[int],
        vid_dir: str,
        fps: int,
        save_traj_name: Optional[str] = None,
        **kwargs,
    ):
        """
        :param save_traj_name: The full file path (for example "data/trajs/data.pth") to save the evaluated trajectories to.
        """
        self._envs = envs
        self._rnn_hxs_dim = rnn_hxs_dim
        self._num_render = num_render
        self._vid_dir = vid_dir
        self._fps = fps
        self._should_save_trajs = save_traj_name is not None
        self._save_traj_name = save_traj_name

    def _clear_save_trajs(self):
        self._save_trajs_obs = defaultdict(list)
        self._save_trajs_actions = defaultdict(list)
        self._save_trajs_rewards = defaultdict(list)
        self._save_trajs_done = defaultdict(list)
        self._save_trajs_info = defaultdict(list)

        self._all_traj_obs = []
        self._all_traj_actions = []
        self._all_traj_rewards = []
        self._all_traj_done = []
        self._all_traj_info = []

    def _add_transition_to_save(self, env_i, obs, action, reward, done, info):
        self._save_trajs_obs[env_i].append(obs[env_i])
        self._save_trajs_actions[env_i].append(action[env_i])
        self._save_trajs_rewards[env_i].append(reward[env_i])
        self._save_trajs_done[env_i].append(done[env_i])
        self._save_trajs_info[env_i].append(info[env_i])

    def _flush_trajectory_to_save(self, env_i):
        self._all_traj_obs.extend(self._save_trajs_obs[env_i])
        self._all_traj_actions.extend(self._save_trajs_actions[env_i])
        self._all_traj_rewards.extend(self._save_trajs_rewards[env_i])
        self._all_traj_done.extend(self._save_trajs_done[env_i])
        self._all_traj_info.extend(self._save_trajs_info[env_i])

        self._save_trajs_obs[env_i].clear()
        self._save_trajs_actions[env_i].clear()
        self._save_trajs_rewards[env_i].clear()
        self._save_trajs_done[env_i].clear()
        self._save_trajs_info[env_i].clear()

    @property
    def eval_trajs_obs(self):
        return self._all_traj_obs

    @property
    def eval_trajs_dones(self):
        return self._all_traj_done

    def _save_trajs(self):
        assert self._save_traj_name is not None
        obs = torch.stack(self._all_traj_obs, dim=0).detach()
        actions = torch.stack(self._all_traj_actions, dim=0).detach()
        rewards = torch.stack(self._all_traj_rewards, dim=0).detach()
        terminals = torch.stack(self._all_traj_done, dim=0).detach()

        num_steps = obs.shape[0]
        assert (
            actions.shape[0] == num_steps and len(actions.shape) == 2
        ), f"Action shape wrong {actions.shape}"

        rewards = rewards.view(-1)
        terminals = terminals.view(-1)
        assert rewards.size(0) == num_steps, f"Reward is wrong shape {rewards.shape}"
        assert (
            terminals.size(0) == num_steps
        ), f"Terminals is wrong shape {terminals.shape}"

        os.makedirs(osp.dirname(self._save_traj_name), exist_ok=True)

        torch.save(
            {
                "observations": obs,
                "actions": actions,
                "rewards": rewards,
                "terminals": terminals,
                "infos": self._all_traj_info,
            },
            self._save_traj_name,
        )
        print(f"Saved trajectories to {self._save_traj_name}")
        self._clear_save_trajs()

    def evaluate(
        self, policy: BasePolicy, num_episodes: int, eval_i: int
    ) -> Dict[str, float]:
        self._clear_save_trajs()

        if isinstance(policy, nn.Module):
            device = next(policy.parameters()).device
        else:
            device = torch.device("cpu")

        num_envs = self._envs.num_envs
        obs = self._envs.reset()
        rnn_hxs = torch.zeros(num_envs, self._rnn_hxs_dim).to(device)
        eval_masks = torch.zeros(num_envs, 1, device=device)

        evals_per_proc = num_episodes // num_envs
        left_over_evals = num_episodes % num_envs
        num_evals = [evals_per_proc for _ in range(num_envs)]
        num_evals[-1] += left_over_evals

        all_frames = []
        accum_stats = defaultdict(list)
        total_evaluated = 0
        if self._num_render is None:
            num_render = num_episodes
        else:
            num_render = self._num_render

        while sum(num_evals) != 0:
            act_data = policy.act(obs, rnn_hxs, eval_masks, deterministic=True)
            next_obs, rewards, done, info = self._envs.step(act_data["action"])
            rnn_hxs = act_data["recurrent_hidden_states"]

            if total_evaluated < num_render:
                frames = self._envs.render(mode="rgb_array")
                all_frames.append(frames)

            for env_i in range(num_envs):
                self._add_transition_to_save(
                    env_i, obs, act_data["action"], rewards, done, info
                )

                if done[env_i]:
                    total_evaluated += 1
                    if num_evals[env_i] > 0:
                        self._flush_trajectory_to_save(env_i)
                        for k, v in compress_and_filter_dict(info[env_i]).items():
                            accum_stats[k].append(v)
                        num_evals[env_i] -= 1
            obs = next_obs

        if len(all_frames) > 0:
            save_mp4(all_frames, self._vid_dir, f"eval_{eval_i}", self._fps)

        if self._should_save_trajs:
            self._save_trajs()

        return {k: np.mean(v) for k, v in accum_stats.items()}
