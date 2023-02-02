import numpy as np

from . import VecEnvWrapper


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0

    def reset(self):
        self.eprets = np.zeros(self.num_envs, "f")
        self.eplens = np.zeros(self.num_envs, "i")

        return self.venv.reset()

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1
        newinfos = []
        for i, (done, ret, eplen, info) in enumerate(
            zip(dones, self.eprets, self.eplens, infos)
        ):
            info = info.copy()
            if done:
                epinfo = {
                    "reward": ret,
                    "length": eplen,
                }
                info["episode"] = epinfo
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
            newinfos.append(info)

        return obs, rews, dones, newinfos
