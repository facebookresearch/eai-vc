from collections import defaultdict
import numpy as np
import torch
import random
import gym
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


class DefaultDictWrapper(gym.Wrapper):
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)
		self.env = env

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		return obs, reward, done, defaultdict(float, info)


class FrameEmitterWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		gym.Wrapper.__init__(self, env)
		self.env = env
		self.cfg = cfg
		self._domain = cfg.task.split('-')[0]
		self._enabled = cfg.get('demo', False)
		self._frames = []

	@property
	def unwrapped(self):
		return self.env.unwrapped
	
	@property
	def frames(self):
		return self._frames
	
	def _save_frame(self):
		if self._enabled:
			if self._domain == 'rlb':
				raise NotImplementedError()
			elif self._domain == 'mw':
				frame = self.env.render(mode='rgb_array', height=84, width=84)
			else:
				frame = self.env.render(mode='rgb_array', height=84, width=84, camera_id=2 if self._domain == 'quadruped' else 0)
			self._frames.append(frame)
	
	def reset(self):
		self._frames = []
		obs = self.env.reset()
		self._save_frame()
		return obs
	
	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		self._save_frame()
		return obs, reward, done, info


def make_env(cfg):
	"""
	Make environment for TD-MPC experiments.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	domain = cfg.task.split('-')[0]

	if domain == 'rlb': # RLBench
		from tasks.rlbench import make_rlbench_env
		env = make_rlbench_env(cfg)
	elif domain == 'mw': # Meta-World
		from tasks.metaworld import make_metaworld_env
		env = make_metaworld_env(cfg)
	else: # DMControl
		from tasks.dmcontrol import make_dmcontrol_env
		env = make_dmcontrol_env(cfg)

	env = DefaultDictWrapper(env)
	env = FrameEmitterWrapper(env, cfg)
	cfg.domain = domain
	cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
	cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
	cfg.action_dim = env.action_space.shape[0]

	return env
