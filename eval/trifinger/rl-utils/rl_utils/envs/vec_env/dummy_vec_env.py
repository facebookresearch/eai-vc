from collections.abc import Iterable

import numpy as np

from rl_utils.common.core_utils import copy_obs_dict, dict_to_obs, obs_space_info

from .vec_env import VecEnv


class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """

    def __init__(self, env_fns):
        """
        Arguments:

        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = {
            k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])
            for k in self.keys
        }
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.spec = self.envs[0].spec

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert (
                self.num_envs == 1
            ), "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(
                actions, self.num_envs
            )
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            # if isinstance(self.envs[e].action_space, spaces.Discrete):
            #    action = int(action)

            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[
                e
            ].step(action)
            if self.buf_dones[e]:
                final_obs = obs
                if isinstance(obs, dict) and "observation" in obs:
                    final_obs = obs["observation"]
                self.buf_infos[e]["final_obs"] = final_obs
                obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            self.buf_infos.copy(),
        )

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self, **kwargs):
        return [env.render(mode="rgb_array", **kwargs) for env in self.envs]

    def render(self, mode="human", **kwargs):
        if self.num_envs == 1:
            use_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    use_kwargs[k] = {
                        k_j: (v_j[0] if isinstance(v_j, Iterable) else v_j)
                        for k_j, v_j in v.items()
                    }
                elif isinstance(v, Iterable):
                    use_kwargs[k] = kwargs[k][0]
                else:
                    use_kwargs[k] = kwargs[k]
            return self.envs[0].render(mode=mode, **use_kwargs)
        else:
            return super().render(mode=mode, **kwargs)
