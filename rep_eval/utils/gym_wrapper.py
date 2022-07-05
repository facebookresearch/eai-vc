import os
import numpy as np
import torch
import gym
from mjrl.utils.gym_env import GymEnv
from rep_eval.utils.model_loading import load_pvr_model
from gym.spaces.box import Box
from torch._C import device
from torch.nn.modules.linear import Identity


def set_seed(seed=None):
    """Set all seeds to make results reproducible
    :param seed: an integer to your choosing (default: None)
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


class MuJoCoPixelObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, width, height, camera_name, device_id=-1, depth=False, *args, **kwargs):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0., high=255., shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id

    def get_image(self):
        if self.spec.id.startswith('dmc'):
            # dmc backend
            # dmc expects camera_id as an integer and not name
            if self.camera_name == None or self.camera_name == 'None':
                self.camera_name = 0
            img = self.env.unwrapped.render(mode='rgb_array', width=self.width, 
                                            height=self.height, camera_id=int(self.camera_name))
        else:
            # mujoco-py backend
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                                  camera_name=self.camera_name, device_id=self.device_id)
            img = img[::-1,:,:]
        return img

    def observation(self, observation):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        return self.get_image()
        

class FrozenEmbeddingWrapper(gym.ObservationWrapper):
    """
    This wrapper places a frozen vision model over the image observation.

    Args:
        env (Gym environment): the original environment,
        embedding_name (str): name of the embedding to use
        history_window (int, 1) : timesteps of observation embedding to incorporate into observation (state)
        embedding_fusion (callable, 'None'): function for fusing the embeddings into a state.
            Defaults to concatenation if not specified
        obs_dim (int, 'None') : dimensionality of observation space. Inferred if not specified. 
            Required if function != None. Defaults to history_window * embedding_dim
        device (str, 'cuda'): where to allocate the model.

    """
    def __init__(self, env,
                 embedding_name : str,
                 history_window : int = 1,
                 fuse_embeddings : callable = None,
                 obs_dim : int = None,
                 device : str = 'cuda',
                 seed : int = None,
                 *args, **kwargs):

        gym.ObservationWrapper.__init__(self, env)

        self.embedding_buffer = []  # buffer to store raw embeddings of the image observation
        self.obs_buffer = [] # temp variable, delete this line later
        self.history_window = history_window
        self.fuse_embeddings = fuse_embeddings
        if device == 'cuda' and torch.cuda.is_available():
            print('Using CUDA.')
            device = torch.device('cuda')
        else:
            print('Not using CUDA.')
            device = torch.device('cpu')
        self.device = device

        # get the embedding model
        embedding, embedding_dim, transforms = load_pvr_model(embedding_name=embedding_name, seed=seed)
        embedding.to(device=self.device)
        # freeze the PVR
        for p in embedding.parameters():
            p.requires_grad = False
        self.embedding, self.embedding_dim, self.transforms = embedding, embedding_dim, transforms
        obs_dim = obs_dim if obs_dim != None else int(self.history_window * self.embedding_dim)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

    def observation(self, observation):
        # observation shape : (H, W, 3)
        inp = self.transforms(observation)
        inp = inp.to(self.device)
        with torch.no_grad():
            emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
        # update observation buffer
        if len(self.embedding_buffer) < self.history_window:
            # initialization
            self.embedding_buffer = [emb.copy()] * self.history_window
        else:
            # fixed size buffer, replace oldest entry
            for i in range(self.history_window-1):
                self.embedding_buffer[i] = self.embedding_buffer[i+1].copy()
            self.embedding_buffer[-1] = emb.copy()

        # fuse embeddings to obtain observation
        if self.fuse_embeddings != None:
            obs = self.fuse_embeddings(self.embedding_buffer)
        else:
            print("Fuse embedding function not give. Defaulting to concat.")
            obs = np.array(self.embedding_buffer).ravel()

        return obs

    def get_obs(self):
        return self.observation(self.env.observation(None))

    def get_image(self):
        return self.env.get_image()

    def reset(self):
        self.embedding_buffer = []    # reset to empty buffer
        return super().reset()


def env_constructor(env_name: str,
                    pixel_based : bool = True,
                    device: str = 'cuda', 
                    image_width: int = 256, image_height: int = 256,
                    camera_name: str = None, 
                    embedding_name : str = 'resnet50',
                    history_window : int = 1,
                    fuse_embeddings : callable = None,
                    render_gpu_id : int = - 1, 
                    seed : int = 123,
                    *args, **kwargs) -> GymEnv:
    # get correct camera name
    camera_name = None if (camera_name == 'None' or camera_name == 'default') else camera_name
    e = gym.make(env_name)
    e.seed(seed)
    # Use appropriate observation wrapper
    if pixel_based:
        e = MuJoCoPixelObsWrapper(env=e, width=image_width, height=image_height, 
                                  camera_name=camera_name, device_id=render_gpu_id)
        e = FrozenEmbeddingWrapper(env=e, embedding_name=embedding_name, history_window=history_window,
                                   fuse_embeddings=fuse_embeddings, device=device, seed=seed)
        e = GymEnv(e)
    else:
        e = GymEnv(e)
    # Output wrapped env
    e.set_seed(seed)
    return e