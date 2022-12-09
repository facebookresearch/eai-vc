import os
import os.path as osp
from collections import defaultdict
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn

from rl_utils.common.viz_utils import save_mp4
from rl_utils.envs.vec_env.vec_env import VecEnv
from rl_utils.interfaces import BasePolicy
from tensordict import TensorDict


def compute_ftip_distances(obs):
    ftip_dists = []
    num_fingers = 3
    for i in range(num_fingers):
        ftip_dists.append(
            torch.norm(
                obs[:, 3 * i : 3 + (3 * i)] - obs[:, 9 + 3 * i : 12 + (3 * i)], dim=1
            )
        )
    return ftip_dists


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
        num_envs: int,
        num_steps: int,
        info_keys: List[str],
        device,
        save_traj_name: Optional[str] = None,
        **kwargs,
    ):
        """
        :param save_traj_name: The full file path (for example "data/trajs/data.pth") to save the evaluated trajectories to.
        """
        self._num_envs = num_envs
        self._num_steps = num_steps
        self._device = device
        self._info_keys = info_keys
        self._envs = envs
        self._rnn_hxs_dim = rnn_hxs_dim
        self._num_render = num_render
        self._vid_dir = vid_dir
        self._parent_dir = os.path.join("data/logs/", os.path.split(vid_dir)[1])
        self._eval_log = "eval.log"
        self._write_to_file = True
        self._fps = fps
        self._should_save_trajs = save_traj_name is not None
        self._save_traj_name = save_traj_name

        self._save_traj_td = TensorDict(
            {}, batch_size=[self._num_envs, self._num_steps], device=self._device
        )
        self._all_traj_td = None

        if not os.path.exists("data/logs/"):
            os.mkdir("data/logs/")
        os.mkdir(self._parent_dir)

    def _clear_save_trajs(self, max_evals):

        if self._all_traj_td is None:
            self._all_traj_td = TensorDict(
                {},
                batch_size=[self._num_envs, max_evals, self._num_steps],
                device=self._device,
            )
        self._save_traj_td = self._save_traj_td.empty()
        self._all_traj_td = self._all_traj_td.empty()
        self._mask_traj = torch.zeros(self._num_envs, max_evals)
        self._all_traj_obs = []
        self._all_traj_actions = []
        self._all_traj_rewards = []
        self._all_traj_done = []
        self._all_traj_info = []

    def _add_transition_to_save(self, td, step_idx):
        self._save_traj_td[:, step_idx] = td

    def _flush_trajectory_to_save(self, env_i, num_eval):
        # TODO CLEAN up
        self._save_traj_td = self._save_traj_td.to(
            self._save_traj_td["observation"].device
        )
        self._all_traj_td[env_i, num_eval] = self._save_traj_td[env_i]
        self._mask_traj[env_i, num_eval] = 1

    @property
    def eval_trajs_obs(self):
        return self._all_traj_obs

    @property
    def eval_trajs_dones(self):
        return self._all_traj_done

    def _save_trajs(self):
        assert self._save_traj_name is not None
        all_traj = self._all_traj_td[self._mask_traj]
        obs = all_traj["observation"].detach()
        actions = all_traj["action"].detach()
        rewards = all_traj["reward"].detach()
        terminals = all_traj["done"].detach()

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
        if isinstance(policy, nn.Module):
            device = next(policy.parameters()).device
        else:
            device = torch.device("cpu")

        obs = self._envs.reset()

        rnn_hxs = torch.zeros(self._num_envs, self._rnn_hxs_dim).to(device)
        eval_masks = torch.zeros(self._num_envs, 1, device=device)

        evals_per_proc = num_episodes // self._num_envs
        left_over_evals = num_episodes % self._num_envs
        num_evals = [evals_per_proc for _ in range(self._num_envs)]
        num_evals[-1] += left_over_evals
        self._clear_save_trajs(num_evals[-1] + 1)
        step_num = 0

        all_frames = []
        accum_stats = defaultdict(list)
        total_evaluated = 0

        if self._num_render is None:
            num_render = num_episodes
        else:
            num_render = self._num_render

        obs = obs["observation"]
        while sum(num_evals) != 0:
            td = TensorDict(
                source={"observation": obs, "hxs": rnn_hxs, "mask": eval_masks},
                batch_size=[self._num_envs],
                device=device,
            )

            policy.act(td, deterministic=True)
            self._envs.step(td)

            next_obs, done = td["next"]["observation"], td["done"]

            rnn_hxs = td["recurrent_hidden_states"]
            if total_evaluated < num_render:
                frames = self._envs.render(mode="eval")
                all_frames.append(frames[0])

            self._add_transition_to_save(td, step_num)
            for env_i in range(self._num_envs):
                if done[env_i]:
                    total_evaluated += 1
                    if num_evals[env_i] > 0:
                        self._flush_trajectory_to_save(env_i, num_evals[env_i])
                        for k in self._info_keys:
                            v = td[k][env_i]
                            v = v.item() if v.shape == torch.Size([1]) else v.tolist()
                            accum_stats[k].append(v)
                        num_evals[env_i] -= 1

            step_num += 1
            if step_num == self._num_steps:
                step_num = 0

            if self._envs.is_done:
                completed_td = td["observation"].detach()
                ftip_distances = torch.norm(
                    completed_td[:, :9] - completed_td[:, 9:], dim=1
                )
                per_finger_dists = compute_ftip_distances(completed_td)
                avg_ftip_dist = 0
                for f in per_finger_dists:
                    avg_ftip_dist += f
                avg_ftip_dist = avg_ftip_dist / 3

                if (
                    self._write_to_file
                    and ftip_distances.mean().item() > 0.10
                    and eval_i > 300
                ):
                    # only once per eval
                    self._write_to_file = False
                    for x in range(completed_td.shape[0]):
                        debug_log_file = os.path.join(self._parent_dir, self._eval_log)
                        with open(debug_log_file, "a") as f:
                            # TODO add start pos
                            f.write("Start Postions")
                            f.write(
                                "".join(
                                    str(x)
                                    for x in self._envs.get_start_pos(return_goal=True)
                                )
                            )
                            f.write(
                                f"\nGOAL:{completed_td[x, 9:]}\nFINGERTIP_DIST:{ftip_distances[x]}"
                            )

                # success_rate = percentage done and w/ distance less than X / all storage_td
                success_rate = (
                    ftip_distances < 0.02 * td["done"].T
                ).sum() / ftip_distances.shape[0]

                accum_stats["scaled_success"].append(
                    td["scaled_success"].detach().mean().item()
                )
                accum_stats["success_total_size"].append(ftip_distances.shape[0])
                accum_stats["eval_success_rate"].append(success_rate)
                accum_stats["eval_ftip_distances"].append(ftip_distances.mean().item())

                accum_stats["f0_dist"].append(per_finger_dists[0].mean().item())
                accum_stats["f1_dist"].append(per_finger_dists[1].mean().item())
                accum_stats["f2_dist"].append(per_finger_dists[2].mean().item())

                accum_stats["avg_ftip_distances"].append(avg_ftip_dist.mean().item())

                td.set("reset_workers", td["done"])
                reset_td = self._envs.reset(tensordict=td)
                next_obs = reset_td["observation"]

            obs = next_obs

        if len(all_frames) > 0:
            save_mp4(all_frames, self._vid_dir, f"eval_{eval_i}", self._fps)

        if self._should_save_trajs:
            self._save_trajs()

        # for next time eval gets called
        self._write_to_file = True
        return {k: np.mean(np.array(v)) for k, v in accum_stats.items()}


# TODO get rid of proprocess function
class PretrainedEvaluator(Evaluator):
    def __init__(
        self,
        envs: VecEnv,
        rnn_hxs_dim: int,
        num_render: Optional[int],
        vid_dir: str,
        fps: int,
        num_envs: int,
        num_steps: int,
        info_keys: List[str],
        device,
        pretrained_model=None,
        embedding_size=None,
        preprocess_func=None,
        preprocess_transform=None,
        save_traj_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            envs,
            rnn_hxs_dim,
            num_render,
            vid_dir,
            fps,
            num_envs,
            num_steps,
            info_keys,
            device,
            save_traj_name,
            **kwargs,
        )

    def evaluate(
        self, policy: BasePolicy, num_episodes: int, eval_i: int
    ) -> Dict[str, float]:
        if isinstance(policy, nn.Module):
            device = next(policy.parameters()).device
        else:
            device = torch.device("cpu")

        obs = self._envs.reset()

        rnn_hxs = torch.zeros(self._num_envs, self._rnn_hxs_dim).to(device)
        eval_masks = torch.zeros(self._num_envs, 1, device=device)

        evals_per_proc = num_episodes // self._num_envs
        left_over_evals = num_episodes % self._num_envs
        num_evals = [evals_per_proc for _ in range(self._num_envs)]
        num_evals[-1] += left_over_evals
        self._clear_save_trajs(num_evals[-1] + 1)
        step_num = 0

        all_frames = []
        accum_stats = defaultdict(list)
        total_evaluated = 0

        if self._num_render is None:
            num_render = num_episodes
        else:
            num_render = self._num_render

        obs = obs["pixels"]
        while sum(num_evals) != 0:

            td = TensorDict(
                source={"pixels": obs, "hxs": rnn_hxs, "mask": eval_masks},
                batch_size=[self._num_envs],
                device=device,
            )

            policy.act(td, deterministic=True)
            self._envs.step(td)

            # next_obs, done = td["next_observation"], td["done"]
            # next_obs, done = td["next_pixels"], td["done"]
            next_obs = td["next"]["pixels"]
            done = td["done"]
            rnn_hxs = td["recurrent_hidden_states"]
            if total_evaluated < num_render:
                # TODO try catch for other envs
                frames = self._envs.render(mode="eval")
                all_frames.append(frames[0])

            # self._add_transition_to_save(td, step_num)
            for env_i in range(self._num_envs):
                if done[env_i]:
                    total_evaluated += 1
                    if num_evals[env_i] > 0:

                        # self._flush_trajectory_to_save(env_i, num_evals[env_i])
                        # for k in self._info_keys:
                        #     v = td[k][env_i]
                        #     v = v.item() if v.shape == torch.Size([1]) else v.tolist()
                        #     accum_stats[k].append(v)
                        num_evals[env_i] -= 1

            step_num += 1
            if step_num == self._num_steps:
                step_num = 0

            if self._envs.is_done:
                completed_td = td["next"]["ftip_dist"].detach()
                ftip_distances = torch.norm(
                    completed_td[:, :9] - completed_td[:, 9:], dim=1
                )

                per_finger_dists = compute_ftip_distances(completed_td)
                avg_ftip_dist = 0
                for f in per_finger_dists:
                    avg_ftip_dist += f
                avg_ftip_dist = avg_ftip_dist / 3

                if (
                    self._write_to_file
                    and ftip_distances.mean().item() > 0.10
                    and eval_i > 800
                ):
                    # only once per eval
                    self._write_to_file = False
                    for x in range(completed_td.shape[0]):
                        debug_log_file = os.path.join(self._parent_dir, self._eval_log)
                        with open(debug_log_file, "a") as f:
                            f.write(
                                f"\nGOAL:{completed_td[x, 9:]}\nFINGERTIP_DIST:{ftip_distances[x]}"
                            )

                # success_rate = percentage done and w/ distance less than X / all storage_td
                success_rate = (
                    avg_ftip_dist < 0.02 * td["done"].T
                ).sum() / avg_ftip_dist.shape[0]

                # accum_stats["success_total_size"].append(ftip_distances.shape[0])
                accum_stats["eval_success_rate"].append(
                    success_rate.cpu().mean().item()
                )
                accum_stats["eval_ftip_distances"].append(
                    ftip_distances.cpu().mean().item()
                )
                accum_stats["f0_dist"].append(per_finger_dists[0].cpu().mean().item())
                accum_stats["f1_dist"].append(per_finger_dists[1].cpu().mean().item())
                accum_stats["f2_dist"].append(per_finger_dists[2].cpu().mean().item())
                accum_stats["avg_ftip_distances"].append(
                    avg_ftip_dist.cpu().mean().item()
                )
                accum_stats["scaled_success"].append(
                    td["next"]["scaled_success"].detach().mean().item()
                )

                td.set("reset_workers", td["done"])
                reset_td = self._envs.reset(tensordict=td)
                next_obs = reset_td["pixels"]

            obs = next_obs

        if len(all_frames) > 0:
            save_mp4(all_frames, self._vid_dir, f"eval_{eval_i}", self._fps)

        if self._should_save_trajs:
            self._save_trajs()
        self._write_to_file = True

        # ret_dict = {}
        # for k, v in accum_stats.items():
        #     print(k)
        #     print(v)
        #     ret_dict[k] =  np.mean(np.array(v))
        # return ret_dict
        return {k: np.mean(np.array(v)) for k, v in accum_stats.items()}
