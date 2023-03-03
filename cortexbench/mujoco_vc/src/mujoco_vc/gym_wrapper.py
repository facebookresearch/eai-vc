#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import gym
from mjrl.utils.gym_env import GymEnv
from gym.spaces.box import Box
from mujoco_vc.model_loading import load_pretrained_model
from mujoco_vc.supported_envs import ENV_TO_SUITE
from typing import Union, Tuple


class MuJoCoPixelObsWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        width,
        height,
        camera_name,
        device_id=-1,
        depth=False,
        *args,
        **kwargs
    ):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0.0, high=255.0, shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id

    def get_image(self):
        if self.spec.id.startswith("dmc"):
            # dmc backend
            # dmc expects camera_id as an integer and not name
            if self.camera_name == None or self.camera_name == "None":
                self.camera_name = 0
            img = self.env.unwrapped.render(
                mode="rgb_array",
                width=self.width,
                height=self.height,
                camera_id=int(self.camera_name),
            )
        else:
            # mujoco-py backend
            img = self.sim.render(
                width=self.width,
                height=self.height,
                depth=self.depth,
                camera_name=self.camera_name,
                device_id=self.device_id,
            )
            img = img[::-1, :, :]
        return img

    def observation(self, observation):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        # Output format is (H, W, 3)
        return self.get_image()


class FrozenEmbeddingWrapper(gym.ObservationWrapper):
    """
    This wrapper places a frozen vision model over the image observation.

    Args:
        env (Gym environment): the original environment
        suite (str): category of environment ["dmc", "adroit", "metaworld"]
        embedding_name (str): name of the embedding to use (name of config)
        history_window (int, 1) : timesteps of observation embedding to incorporate into observation (state)
        embedding_fusion (callable, 'None'): function for fusing the embeddings into a state.
            Defaults to concatenation if not specified
        obs_dim (int, 'None') : dimensionality of observation space. Inferred if not specified.
            Required if function != None. Defaults to history_window * embedding_dim
        add_proprio (bool, 'False') : flag to specify if proprioception should be appended to observation
        device (str, 'cuda'): where to allocate the model.
    """

    def __init__(
        self,
        env,
        embedding_name: str,
        suite: str,
        history_window: int = 1,
        fuse_embeddings: callable = None,
        obs_dim: int = None,
        device: str = "cuda",
        seed: int = None,
        add_proprio: bool = False,
        *args,
        **kwargs
    ):
        gym.ObservationWrapper.__init__(self, env)

        self.embedding_buffer = (
            []
        )  # buffer to store raw embeddings of the image observation
        self.obs_buffer = []  # temp variable, delete this line later
        self.history_window = history_window
        self.fuse_embeddings = fuse_embeddings
        if device == "cuda" and torch.cuda.is_available():
            print("Using CUDA.")
            device = torch.device("cuda")
        else:
            print("Not using CUDA.")
            device = torch.device("cpu")
        self.device = device

        # get the embedding model
        embedding, embedding_dim, transforms, metadata = load_pretrained_model(
            embedding_name=embedding_name, seed=seed
        )
        embedding.to(device=self.device)
        # freeze the PVR
        for p in embedding.parameters():
            p.requires_grad = False
        self.embedding, self.embedding_dim, self.transforms = (
            embedding,
            embedding_dim,
            transforms,
        )

        # proprioception
        if add_proprio:
            self.get_proprio = lambda: get_proprioception(self.unwrapped, suite)
            proprio = self.get_proprio()
            self.proprio_dim = 0 if proprio is None else proprio.shape[0]
        else:
            self.proprio_dim = 0
            self.get_proprio = None

        # final observation space
        obs_dim = (
            obs_dim
            if obs_dim != None
            else int(self.history_window * self.embedding_dim + self.proprio_dim)
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

    def observation(self, observation):
        # observation shape : (H, W, 3)
        inp = self.transforms(
            observation
        )  # numpy to PIL to torch.Tensor. Final dimension: (1, 3, H, W)
        inp = inp.to(self.device)
        with torch.no_grad():
            emb = (
                self.embedding(inp)
                .view(-1, self.embedding_dim)
                .to("cpu")
                .numpy()
                .squeeze()
            )
        # update observation buffer
        if len(self.embedding_buffer) < self.history_window:
            # initialization
            self.embedding_buffer = [emb.copy()] * self.history_window
        else:
            # fixed size buffer, replace oldest entry
            for i in range(self.history_window - 1):
                self.embedding_buffer[i] = self.embedding_buffer[i + 1].copy()
            self.embedding_buffer[-1] = emb.copy()

        # fuse embeddings to obtain observation
        if self.fuse_embeddings != None:
            obs = self.fuse_embeddings(self.embedding_buffer)
        else:
            # print("Fuse embedding function not given. Defaulting to concat.")
            obs = np.array(self.embedding_buffer).ravel()

        # add proprioception if necessary
        if self.proprio_dim > 0:
            proprio = self.get_proprio()
            obs = np.concatenate([obs, proprio])
        return obs

    def get_obs(self):
        return self.observation(self.env.observation(None))

    def get_image(self):
        return self.env.get_image()

    def reset(self):
        self.embedding_buffer = []  # reset to empty buffer
        return super().reset()


def env_constructor(
    env_name: str,
    pixel_based: bool = True,
    device: str = "cuda",
    image_width: int = 256,
    image_height: int = 256,
    camera_name: str = None,
    embedding_name: str = "resnet50",
    history_window: int = 1,
    fuse_embeddings: callable = None,
    render_gpu_id: int = -1,
    seed: int = 123,
    add_proprio=False,
    *args,
    **kwargs
) -> GymEnv:
    # construct basic gym environment
    assert env_name in ENV_TO_SUITE.keys()
    suite = ENV_TO_SUITE[env_name]
    if suite == "metaworld":
        # Meta world natively misses many specs. We will explicitly add them here.
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        from collections import namedtuple

        e = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]()
        e._freeze_rand_vec = False
        e.spec = namedtuple("spec", ["id", "max_episode_steps"])
        e.spec.id = env_name
        e.spec.max_episode_steps = 500
    else:
        e = gym.make(env_name)
    # seed the environment for reproducibility
    e.seed(seed)

    # get correct camera name
    camera_name = (
        None if (camera_name == "None" or camera_name == "default") else camera_name
    )
    # Use appropriate observation wrapper
    if pixel_based:
        e = MuJoCoPixelObsWrapper(
            env=e,
            width=image_width,
            height=image_height,
            camera_name=camera_name,
            device_id=0,
        )
        e = FrozenEmbeddingWrapper(
            env=e,
            embedding_name=embedding_name,
            suite=suite,
            history_window=history_window,
            fuse_embeddings=fuse_embeddings,
            device=device,
            seed=seed,
            add_proprio=add_proprio,
        )
        e = GymEnv(e)
    else:
        e = GymEnv(e)

    # Output wrapped env
    e.set_seed(seed)
    return e


def get_proprioception(env: gym.Env, suite: str) -> Union[np.ndarray, None]:
    assert isinstance(env, gym.Env)
    if suite == "metaworld":
        return env.unwrapped._get_obs()[:4]
    elif suite == "adroit":
        # In adroit, in-hand tasks like pen lock the base of the hand
        # while other tasks like relocate allow for movement of hand base
        # as if attached to an arm
        if env.unwrapped.spec.id == "pen-v0":
            return env.unwrapped.get_obs()[:24]
        elif env.unwrapped.spec.id == "relocate-v0":
            return env.unwrapped.get_obs()[:30]
        else:
            print("Unsupported environment. Proprioception is defaulting to None.")
            return None
    elif suite == "dmc":
        # no proprioception used for dm-control
        return None
    else:
        print("Unsupported environment. Proprioception is defaulting to None.")
        return None
