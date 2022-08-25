import gym
import numpy as np
import torch

import rl_utils.common.core_utils as utils
from rl_utils.envs.vec_env.vec_env import VecEnvWrapper


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info["bad_transition"] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def _data_convert(self, arr):
        if isinstance(arr, np.ndarray) and arr.dtype == np.float64:
            return arr.astype(np.float32)
        return arr

    def reset(self):
        obs = self.venv.reset()
        return self._trans_obs(obs)

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def _trans_obs(self, obs):
        # Support for dict observations
        def _convert_obs(x):
            x = self._data_convert(x)
            x = torch.Tensor(x)
            return x.to(self.device)

        if isinstance(obs, dict):
            for k in obs:
                obs[k] = _convert_obs(obs[k])
        else:
            return _convert_obs(obs)
        return obs

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = self._trans_obs(obs)

        reward = torch.Tensor(reward).unsqueeze(dim=1)
        # Reward is sometimes a Double. Observation is considered to always be
        # float32
        reward = reward.float()
        return obs, reward, done, info


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    """
    For now, this will only stack the "observation" key in dictionary
    observation spaces.
    """

    def __init__(self, venv, nstack, device):
        self.venv = venv
        self.nstack = nstack

        ob_space = venv.observation_space

        self.stacked_obs = utils.StackHelper(
            ob_space.shape, nstack, device, venv.num_envs
        )
        new_obs_space = utils.update_obs_space(
            venv.observation_space,
            utils.reshape_obs_space(ob_space, self.stacked_obs.get_shape()),
        )

        VecEnvWrapper.__init__(self, venv, observation_space=new_obs_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        stacked_obs, infos = self.stacked_obs.update_obs(obs, news, infos)

        return stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        stacked_obs = self.stacked_obs.reset(obs)
        return utils.set_def_obs(obs, stacked_obs)

    def close(self):
        self.venv.close()
