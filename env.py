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


class MultitaskWrapper(gym.Wrapper):
	def __init__(self, envs, cfg):
		gym.Wrapper.__init__(self, envs[0])
		self._envs = envs
		self._cfg = cfg
		self._tasks = cfg.task_list
		self._task_id = 0
		self._env = self._envs[self._task_id]

	@property
	def task(self):
		return self._tasks[self._task_id]

	@property
	def tasks(self):
		return self._tasks
	
	@property
	def task_id(self):
		return self._task_id
	
	@task_id.setter
	def task_id(self, value):
		assert value is None or 0 <= value < len(self._envs), 'task_id must be in [0, {}) or None'.format(len(self._envs))
		self._task_id = np.random.randint(len(self._envs)) if value is None else value
		self._env = self._envs[self._task_id]
	
	@property
	def task_vec(self):
		vec = np.zeros(len(self.tasks), dtype=np.float32)
		vec[self.task_id] = 1.
		return vec

	@property
	def observation_space(self):
		obs_shapes = np.array([env.observation_space.shape[0] for env in self._envs])
		return self._envs[np.argmax(obs_shapes)].observation_space

	@property
	def action_space(self):
		action_shapes = np.array([env.action_space.shape[0] for env in self._envs])
		return self._envs[np.argmax(action_shapes)].action_space

	def _reshape_obs(self, obs):
		if self._cfg.modality == 'state' and obs.shape[0] < self.observation_space.shape[0]:
			return np.concatenate([obs, np.zeros((self.observation_space.shape[0] - obs.shape[0],))])
		return obs
	
	def _reshape_action(self, action):
		if action.shape[0] > self._env.action_space.shape[0]:
			return action[:self._env.action_space.shape[0]]
		return action
	
	def reset(self):
		return self._reshape_obs(self._env.reset())

	def step(self, action):
		obs, reward, done, info = self._env.step(self._reshape_action(action))
		return self._reshape_obs(obs), reward, done, info
	
	def render(self, *args, **kwargs):
		self._env.render(*args, **kwargs)


def make_multitask_env(cfg):
	__task = cfg.task
	domain, task = cfg.task.split('-', 1)
	assert task.startswith('mt'), 'Expected MT task name, got {}'.format(task)
	num_tasks = int(task[2:])

	if domain == 'rlb': # RLBench
		raise NotImplementedError()
	elif domain == 'mw': # Meta-World
		if num_tasks == 5:
			tasks = ['mw-drawer-close', 'mw-drawer-open', 'mw-hammer', 'mw-box-close', 'mw-pick-place']
		else:
			raise NotImplementedError()
	else: # DMControl
		if num_tasks == 5:
			tasks = ['cup-catch', 'finger-spin', 'cheetah-run', 'walker-run', 'quadruped-run']
		else:
			raise NotImplementedError()

	envs = []
	for _task in tasks:
		cfg.task = _task
		envs.append(make_env(cfg))
	cfg.task = __task
	cfg.task_list = tasks
	env = MultitaskWrapper(envs, cfg)
	cfg.domain = domain
	cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
	cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
	cfg.action_dim = env.action_space.shape[0]
	return env


def make_env(cfg):
	"""
	Make environment for TD-MPC experiments.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	domain, task = cfg.task.split('-', 1)
	if task.startswith('mt'):
		return make_multitask_env(cfg)
	elif domain == 'rlb': # RLBench
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
