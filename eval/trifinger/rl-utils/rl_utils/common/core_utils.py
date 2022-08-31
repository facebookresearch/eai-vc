"""
Helpers for dealing with vectorized environments.
"""

import hashlib
import os
import os.path as osp
import pickle
import random
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List

import gym
import gym.spaces as spaces
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Sets the seed for numpy, python random, and pytorch.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def group_trajectories(
    dones: torch.Tensor, **other_data: Dict[str, torch.Tensor]
) -> List[Dict[str, torch.Tensor]]:
    """
    :param dones: An (N, 1) tensor
    """
    for k, v in other_data.items():
        if v.size(0) != dones.size(0):
            raise ValueError(
                f"Key {k} is improper shape {v.shape} for dones of shape {dones.shape}"
            )

    trajs = []
    cur_traj = defaultdict(list)
    for i, done in enumerate(dones):
        for k, v in other_data.items():
            cur_traj[k].append(v[i])
        if done:
            trajs.append({k: torch.stack(v, dim=0) for k, v in cur_traj.items()})
            cur_traj = defaultdict(list)
    return trajs


def compress_dict(d: Dict[str, Any], pre="", sep=".") -> Dict[str, Any]:
    """
    Compresses a dictionary to only have one "level"
    """
    ret_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            ret_d.update(compress_dict(v, f"{pre}{k}{sep}"))
        else:
            ret_d[pre + k] = str(v)
    return ret_d


def compress_and_filter_dict(d: Dict[str, Any], pre="") -> Dict[str, Any]:
    """
    Compresses a dictionary to only have one "level"
    """
    ret_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            ret_d.update(compress_and_filter_dict(v, f"{k}."))
        elif isinstance(v, (float, int, np.float32, np.float64, np.uint, np.int32)):
            ret_d[pre + k] = v
    return ret_d


def copy_obs_dict(obs):
    """
    Deep-copy an observation dict.
    """
    return {k: np.copy(v) for k, v in obs.items()}


def dict_to_obs(obs_dict):
    """
    Convert an observation dict into a raw array if the
    original observation space was not a Dict space.
    """
    if set(obs_dict.keys()) == {None}:
        return obs_dict[None]
    return obs_dict


def obs_space_info(obs_space):
    """
    Get dict-structured information about a gym.Space.

    Returns:
      A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    if isinstance(obs_space, gym.spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict)
        subspaces = obs_space.spaces
    else:
        subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes


def obs_to_dict(obs):
    """
    Convert an observation into a dict.
    """
    if isinstance(obs, dict):
        return obs
    return {None: obs}


def reshape_obs_space(obs_space, new_shape):
    assert isinstance(obs_space, gym.spaces.Box)
    return gym.spaces.Box(
        shape=new_shape,
        high=obs_space.low.reshape(-1)[0],
        low=obs_space.high.reshape(-1)[0],
        dtype=obs_space.dtype,
    )


def is_dict_obs(ob_space):
    return isinstance(ob_space, gym.spaces.Dict)


def update_obs_space(cur_space, update_obs_space):
    if is_dict_obs(cur_space):
        new_obs_space = cur_space.spaces
        new_obs_space["observation"] = update_obs_space
        return gym.spaces.Dict(new_obs_space)
    else:
        return update_obs_space


class StackHelper:
    """
    A helper for stacking observations.
    """

    def __init__(self, ob_shape, n_stack, device, n_procs=None):
        self.input_dim = ob_shape[0]
        self.n_procs = n_procs
        self.real_shape = (n_stack * self.input_dim, *ob_shape[1:])
        if self.n_procs is not None:
            self.stacked_obs = torch.zeros((n_procs, *self.real_shape))
            if device is not None:
                self.stacked_obs = self.stacked_obs.to(device)
        else:
            self.stacked_obs = np.zeros(self.real_shape)

    def update_obs(self, obs, dones=None, infos=None):
        """
        - obs: torch.tensor
        """
        if self.n_procs is not None:
            self.stacked_obs[:, : -self.input_dim] = self.stacked_obs[
                :, self.input_dim :
            ].clone()
            for (i, new) in enumerate(dones):
                if new:
                    self.stacked_obs[i] = 0
            self.stacked_obs[:, -self.input_dim :] = obs

            # Update info so the final observation frame stack has the final
            # observation as the final frame in the stack.
            for i in range(len(infos)):
                if "final_obs" in infos[i]:
                    new_final = torch.zeros(*self.stacked_obs.shape[1:])
                    new_final[:-1] = self.stacked_obs[i][1:]
                    new_final[-1] = torch.tensor(infos[i]["final_obs"]).to(
                        self.stacked_obs.device
                    )
                    infos[i]["final_obs"] = new_final
            return self.stacked_obs.clone(), infos
        else:
            self.stacked_obs[: -self.input_dim] = self.stacked_obs[
                self.input_dim :
            ].copy()
            self.stacked_obs[-self.input_dim :] = obs

            return self.stacked_obs.copy(), infos

    def reset(self, obs):
        if self.n_procs is not None:
            if torch.backends.cudnn.deterministic:
                self.stacked_obs = torch.zeros(self.stacked_obs.shape)
            else:
                self.stacked_obs.zero_()
            self.stacked_obs[:, -self.input_dim :] = obs
            return self.stacked_obs.clone()
        else:
            self.stacked_obs = np.zeros(self.stacked_obs.shape)
            self.stacked_obs[-self.input_dim :] = obs
            return self.stacked_obs.copy()

    def get_shape(self):
        return self.real_shape


def get_size_for_space(space: spaces.Space) -> int:
    """
    Returns the number of dimensions to represent a Gym space. If the space is discrete, this requires only 1 dimension.
    """
    if isinstance(space, spaces.Discrete):
        return 1
    elif isinstance(space, spaces.Box):
        return space.shape[0]
    else:
        raise ValueError(f"Space {space} not supported")


class CacheHelper:
    CACHE_PATH = "./data/cache"

    def __init__(self, cache_name, lookup_val, def_val=None, verbose=False, rel_dir=""):
        self.use_cache_path = osp.join(CacheHelper.CACHE_PATH, rel_dir)
        if not osp.exists(self.use_cache_path):
            os.makedirs(self.use_cache_path)
        sec_hash = hashlib.md5(str(lookup_val).encode("utf-8")).hexdigest()
        cache_id = f"{cache_name}_{sec_hash}.pickle"
        self.cache_id = osp.join(self.use_cache_path, cache_id)
        self.def_val = def_val
        self.verbose = verbose

    def exists(self):
        return osp.exists(self.cache_id)

    def load(self, load_depth=0):
        if self.exists():
            try:
                with open(self.cache_id, "rb") as f:
                    if self.verbose:
                        print("Loading cache @", self.cache_id)
                    return pickle.load(f)
            except EOFError as e:
                if load_depth == 32:
                    raise e
                # try again soon
                print(
                    "Cache size is ", osp.getsize(self.cache_id), "for ", self.cache_id
                )
                time.sleep(1.0 + np.random.uniform(0.0, 1.0))
                return self.load(load_depth + 1)
            return self.def_val
        else:
            return self.def_val

    def save(self, val):
        with open(self.cache_id, "wb") as f:
            if self.verbose:
                print("Saving cache @", self.cache_id)
            pickle.dump(val, f)
