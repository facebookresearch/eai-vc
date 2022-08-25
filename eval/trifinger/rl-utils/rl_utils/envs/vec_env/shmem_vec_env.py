"""
An interface for asynchronous vectorized environments.
"""

import ctypes
import multiprocessing as mp
from collections.abc import Iterable

import numpy as np

from rl_utils.common.core_utils import dict_to_obs, obs_space_info, obs_to_dict

from .vec_env import FINAL_OBS_KEY, CloudpickleWrapper, VecEnv, clear_mpi_env_vars

_NP_TO_CT = {
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
    np.int32: ctypes.c_int32,
    np.int8: ctypes.c_int8,
    np.uint8: ctypes.c_char,
    bool: ctypes.c_bool,
}


class ShmemVecEnv(VecEnv):
    """
    Optimized version of SubprocVecEnv that uses shared variables to communicate observations.
    """

    def __init__(self, env_fns, spaces=None, context="spawn"):
        """
        If you don't specify observation_space, we'll have to create a dummy
        environment to get it.
        """
        ctx = mp.get_context(context)
        if spaces:
            observation_space, action_space = spaces
        else:
            dummy = env_fns[0]()
            observation_space, action_space = (
                dummy.observation_space,
                dummy.action_space,
            )
            dummy.close()
            del dummy
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(
            observation_space
        )
        self.obs_bufs = [
            {
                k: ctx.Array(
                    _NP_TO_CT[self.obs_dtypes[k].type], int(np.prod(self.obs_shapes[k]))
                )
                for k in self.obs_keys
            }
            for _ in env_fns
        ]
        self.parent_pipes = []
        self.procs = []
        with clear_mpi_env_vars():
            for env_fn, obs_buf in zip(env_fns, self.obs_bufs):
                wrapped_fn = CloudpickleWrapper(env_fn)
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(
                    target=_subproc_worker,
                    args=(
                        child_pipe,
                        parent_pipe,
                        wrapped_fn,
                        obs_buf,
                        self.obs_shapes,
                        self.obs_dtypes,
                        self.obs_keys,
                    ),
                )
                proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()
                child_pipe.close()
        self.waiting_step = False
        self.viewer = None

    def reset(self):
        if self.waiting_step:
            print("Called reset() while waiting for the step to complete")
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(("reset", None))
        return self._decode_obses([pipe.recv() for pipe in self.parent_pipes])

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        for pipe, act in zip(self.parent_pipes, actions):
            pipe.send(("step", act))

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        obs, rews, dones, infos = zip(*outs)
        return self._decode_obses(obs), np.array(rews), np.array(dones), infos

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(("close", None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def get_images(self, mode="human", **kwargs):
        N = len(self.parent_pipes)
        all_pipe_kwargs = []
        for i in range(N):
            pipe_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, Iterable):
                    pipe_kwargs[k] = kwargs[k][i]
                else:
                    pipe_kwargs[k] = kwargs[k]
            all_pipe_kwargs.append(pipe_kwargs)

        for pipe, p_kwargs in zip(self.parent_pipes, all_pipe_kwargs):
            pipe.send(("render", (mode, p_kwargs)))
        return [pipe.recv() for pipe in self.parent_pipes]

    def _decode_obses(self, obs):
        result = {}
        for k in self.obs_keys:
            bufs = [b[k] for b in self.obs_bufs]
            o = [
                np.frombuffer(b.get_obj(), dtype=self.obs_dtypes[k]).reshape(
                    self.obs_shapes[k]
                )
                for b in bufs
            ]
            result[k] = np.array(o)
        return dict_to_obs(result)


def _subproc_worker(
    pipe, parent_pipe, env_fn_wrapper, obs_bufs, obs_shapes, obs_dtypes, keys
):
    """
    Control a single environment instance using IPC and
    shared memory.
    """

    def _write_obs(maybe_dict_obs):
        flatdict = obs_to_dict(maybe_dict_obs)
        for k in keys:
            dst = obs_bufs[k].get_obj()
            dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(
                obs_shapes[k]
            )  # pylint: disable=W0212
            np.copyto(dst_np, flatdict[k])

    env = env_fn_wrapper.x()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == "reset":
                pipe.send(_write_obs(env.reset()))
            elif cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:
                    final_obs = obs
                    if isinstance(obs, dict):
                        final_obs = obs["observation"]
                    info[FINAL_OBS_KEY] = final_obs
                    obs = env.reset()
                pipe.send((_write_obs(obs), reward, done, info))
            elif cmd == "render":
                pipe.send(env.render(mode=data[0], **data[1]))
            elif cmd == "close":
                pipe.send(None)
                break
            else:
                raise RuntimeError("Got unrecognized cmd %s" % cmd)
    except KeyboardInterrupt:
        print("ShmemVecEnv worker: got KeyboardInterrupt")
    finally:
        env.close()
