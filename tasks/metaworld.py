from collections import deque, defaultdict
import numpy as np
import torch
import gym
from gym.wrappers import TimeLimit
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from encode_dataset import encode


class MetaWorldWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self._num_frames = cfg.get('frame_stack', 1)
		self._frames = deque([], maxlen=self._num_frames)
		if cfg.modality == 'pixels':
			self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self._num_frames*3, cfg.img_size, cfg.img_size), dtype=np.uint8)
		elif cfg.modality == 'features':
			features_to_dim = defaultdict(lambda: 2048) # default to RN50
			features_to_dim.update({
				'clip': 512,
				'maehoi': 384,
				'mocoego18': 100352,
			})
			self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._num_frames*features_to_dim[cfg.features],), dtype=np.float32)
		else: # state
			self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space
		self.camera_name = 'corner2'
		self.env.model.cam_pos[2]=[0.75, 0.075, 0.7]
		self.success = False
	
	@property
	def state(self):
		state = self._state_obs.astype(np.float32)
		return np.concatenate((state[:4], state[18:18+4]))
	
	def _get_pixel_obs(self):
		return self.render(width=self.cfg.img_size, height=self.cfg.img_size).transpose(2, 0, 1)
	
	def _get_feature_obs(self):
		obs = torch.from_numpy(self._get_pixel_obs()).unsqueeze(0).view(-1, 3, self.cfg.img_size, self.cfg.img_size)
		obs = encode(obs, self.cfg).view(-1).cpu().numpy()
		return obs

	def _stacked_obs(self):
		assert len(self._frames) == self._num_frames
		return np.concatenate(list(self._frames), axis=0)
	
	def reset(self):
		self.success = False
		obs = self.env.reset()
		self._state_obs = obs
		if self.cfg.modality == 'pixels':
			obs = self._get_pixel_obs()
		elif self.cfg.modality == 'features':
			obs = self._get_feature_obs()
		for _ in range(self._num_frames):
			self._frames.append(obs)
		return self._stacked_obs()
	
	def step(self, action):
		reward = 0
		for _ in range(self.cfg.action_repeat):
			obs, r, _, info = self.env.step(action)
			reward += r
		self._state_obs = obs
		if self.cfg.modality == 'pixels':
			obs = self._get_pixel_obs()
		elif self.cfg.modality == 'features':
			obs = self._get_feature_obs()
		self._frames.append(obs)
		self.success = self.success or bool(info['success'])
		return self._stacked_obs(), reward * self.cfg.get('reward_scale', 1.), False, info

	def render(self, mode='rgb_array', width=None, height=None, camera_id=None):
		return self.env.render(offscreen=True, resolution=(width, height), camera_name=self.camera_name)

	def observation_spec(self):
		return self.observation_space

	def action_spec(self):
		return self.action_space

	def __getattr__(self, name):
		return getattr(self._env, name)


def make_metaworld_env(cfg):
	env_id = cfg.task.split('-', 1)[-1] + '-v2-goal-observable'
	env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=cfg.seed)
	env = MetaWorldWrapper(env, cfg)
	env = TimeLimit(env, max_episode_steps=cfg.episode_length)
	cfg.state_dim = 39
	return env
