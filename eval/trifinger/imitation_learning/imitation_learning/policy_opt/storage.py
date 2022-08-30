from collections import defaultdict
from typing import Dict, Optional

import torch


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage:
    def __init__(
        self,
        num_steps,
        num_processes,
        obs_shape,
        action_dim,
        action_is_discrete,
        recurrent_hidden_state_size,
        device,
        fetch_final_obs,
    ):
        super().__init__()

        if isinstance(obs_shape, dict):
            self.obs_keys = obs_shape
        else:
            self.obs_keys = {None: obs_shape}

        self.obs: Dict[Optional[str], torch.Tensor] = {}
        for k, space_shape in self.obs_keys.items():
            ob = torch.zeros(num_steps + 1, num_processes, *space_shape)
            self.obs[k] = ob

        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size
        )

        self.actions = torch.zeros(num_steps, num_processes, action_dim)
        if action_is_discrete:
            self.actions = self.actions.long()

        self.masks = torch.zeros(num_steps + 1, num_processes, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        if fetch_final_obs:
            self.final_obs = torch.zeros(num_steps, num_processes, *space_shape)
        else:
            self.final_obs = None

        self.num_steps = num_steps
        self.n_procs = num_processes
        self.step = 0
        self.to(device)

    def compute_masks(self, done, infos):
        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

        bad_masks = torch.FloatTensor(
            [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
        )

        return masks, bad_masks

    def __len__(self):
        return self.num_steps * self.n_procs

    def init_storage(self, obs):
        for k in self.obs_keys:
            if k is None:
                self.obs[k][0].copy_(obs)
            else:
                self.obs[k][0].copy_(obs[k])

        self.masks = self.masks.zero_()
        self.bad_masks = self.bad_masks.zero_()
        self.recurrent_hidden_states = self.recurrent_hidden_states.zero_()

    def to(self, device):
        for k in self.obs_keys:
            self.obs[k] = self.obs[k].to(device)

        if self.final_obs is not None:
            self.final_obs = self.final_obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)

    def insert(
        self,
        next_obs,
        rewards,
        done,
        infos,
        action,
        value_preds,
        action_log_probs,
        recurrent_hidden_states,
        **kwargs,
    ):
        masks, bad_masks = self.compute_masks(done, infos)

        for k in self.obs_keys:
            if k is None:
                self.obs[k][self.step + 1].copy_(next_obs)
            else:
                self.obs[k][self.step + 1].copy_(next_obs[k])

        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)

        if self.final_obs is not None:
            for env_i, info in enumerate(infos):
                if "final_obs" in info:
                    self.final_obs[self.step, env_i].copy_(info["final_obs"])

        self.step = (self.step + 1) % self.num_steps

    def get_obs(self, idx):
        ret_d = {}
        for k in self.obs_keys:
            if k is None:
                return self.obs[k][idx]
            ret_d[k] = self.obs[k][idx]
        return ret_d

    def after_update(self):
        for k in self.obs_keys:
            self.obs[k][0].copy_(self.obs[k][-1])

        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])

    def data_generator(self, num_mini_batch, get_next_obs=False, **add_data):
        if get_next_obs and self.final_obs is None:
            raise ValueError(
                "Must fetch final observations if getting the next observation"
            )
        if get_next_obs and len(self.obs_keys) > 1:
            raise ValueError("Cannot fetch next obseration with dictionary observation")

        num_processes = self.rewards.size(1)
        if num_processes < num_mini_batch:
            raise ValueError(
                f"Number of processes {num_processes} is smaller than num mini batch {num_mini_batch}"
            )

        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            ret_data = {
                "observation": defaultdict(list),
                "hxs": [],
                "action": [],
                "value": [],
                "mask": [],
                "prev_log_prob": [],
                "reward": [],
                "add_batch": defaultdict(list),
            }

            if get_next_obs:
                ret_data["next_obs"] = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                for k in self.obs_keys:
                    ret_data["observation"][k].append(self.obs[k][:-1, ind])

                if get_next_obs:
                    mask = self.masks[1:, ind]
                    final_obs = self.final_obs[:, ind]
                    # This code assumes observation dict has only 1 key.
                    first_key = next(iter(self.obs_keys))
                    obs = self.obs[first_key][1:, ind]
                    ret_data["next_obs"].append((mask * obs) + ((1 - mask) * final_obs))

                ret_data["hxs"].append(self.recurrent_hidden_states[0:1, ind])

                ret_data["action"].append(self.actions[:, ind])
                ret_data["value"].append(self.value_preds[:-1, ind])
                ret_data["reward"].append(self.rewards[:, ind])
                ret_data["mask"].append(self.masks[:-1, ind])
                ret_data["prev_log_prob"].append(self.action_log_probs[:, ind])
                for k, v in add_data.items():
                    ret_data["add_batch"][k].append(v[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            for k, v in ret_data.items():
                if k == "hxs":
                    ret_data[k] = torch.stack(v, 1).view(N, -1)
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        ret_data[k][sub_k] = _flatten_helper(
                            T, N, torch.stack(sub_v, 1)
                        )
                else:
                    ret_data[k] = _flatten_helper(T, N, torch.stack(v, 1))

            # Pop the add batch keys out a level
            add_batch = ret_data.pop("add_batch")
            ret_data.update(add_batch)

            # No need to return obs dict if there's only one thing in
            # dictionary
            if len(ret_data["observation"]) == 1:
                ret_data["observation"] = next(iter(ret_data["observation"].values()))
            yield ret_data
