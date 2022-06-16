from collections import deque, defaultdict
from typing import Any, NamedTuple
import dm_env
import numpy as np
import torch
from tasks import walker, cheetah
from dm_control import suite
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
import gym
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from encode_dataset import encode_clip


class ExtendedTimeStep(NamedTuple):
	step_type: Any
	reward: Any
	discount: Any
	observation: Any
	action: Any

	def first(self):
		return self.step_type == StepType.FIRST

	def mid(self):
		return self.step_type == StepType.MID

	def last(self):
		return self.step_type == StepType.LAST


class ActionRepeatWrapper(dm_env.Environment):
	def __init__(self, env, num_repeats):
		self._env = env
		self._num_repeats = num_repeats

	def step(self, action):
		reward = 0.0
		discount = 1.0
		for i in range(self._num_repeats):
			time_step = self._env.step(action)
			reward += (time_step.reward or 0.0) * discount
			discount *= time_step.discount
			if time_step.last():
				break

		return time_step._replace(reward=reward, discount=discount)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def reset(self):
		return self._env.reset()

	def __getattr__(self, name):
		return getattr(self._env, name)
		

class FrameStackWrapper(dm_env.Environment):
	def __init__(self, env, num_frames, pixels_key='pixels'):
		self._env = env
		self._num_frames = num_frames
		self._frames = deque([], maxlen=num_frames)
		self._pixels_key = pixels_key

		wrapped_obs_spec = env.observation_spec()
		assert pixels_key in wrapped_obs_spec

		pixels_shape = wrapped_obs_spec[pixels_key].shape
		if len(pixels_shape) == 4:
			pixels_shape = pixels_shape[1:]
		self._obs_spec = specs.BoundedArray(shape=np.concatenate(
			[[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
											dtype=np.uint8,
											minimum=0,
											maximum=255,
											name='observation')

	def _transform_observation(self, time_step):
		assert len(self._frames) == self._num_frames
		obs = np.concatenate(list(self._frames), axis=0)
		return time_step._replace(observation=obs)

	def _extract_pixels(self, time_step):
		pixels = time_step.observation[self._pixels_key]
		if len(pixels.shape) == 4:
			pixels = pixels[0]
		return pixels.transpose(2, 0, 1).copy()

	def reset(self):
		time_step = self._env.reset()
		pixels = self._extract_pixels(time_step)
		for _ in range(self._num_frames):
			self._frames.append(pixels)
		return self._transform_observation(time_step)

	def step(self, action):
		time_step = self._env.step(action)
		pixels = self._extract_pixels(time_step)
		self._frames.append(pixels)
		return self._transform_observation(time_step)

	def observation_spec(self):
		return self._obs_spec

	def action_spec(self):
		return self._env.action_spec()

	def __getattr__(self, name):
		return getattr(self._env, name)


class FeaturesWrapper(dm_env.Environment):
	def __init__(self, env, num_frames, features=None):
		assert features is not None
		self._env = env
		self._num_frames = num_frames
		self._features = features
		features_to_dim = {
			'clip': 512,
			'rn50': 2048,
		}
		self._obs_spec = specs.BoundedArray(shape=np.array([num_frames*features_to_dim[features],]),
											dtype=np.float32,
											minimum=-np.inf,
											maximum=np.inf,
											name='observation')
	
	def observation_spec(self):
		return self._obs_spec

	def action_spec(self):
		return self._env.action_spec()

	def _encode(self, time_step):
		if self._features is None:
			return time_step
		_obs = torch.from_numpy(time_step.observation).unsqueeze(0)
		_obs = _obs.view(-1, 3, 84, 84).permute(0, 2, 3, 1)
		_obs = encode_clip(_obs).view(-1)
		
		return ExtendedTimeStep(observation=_obs.cpu().numpy(),
								step_type=time_step.step_type,
								action=time_step.action,
								reward=time_step.reward or 0.0,
								discount=time_step.discount or 1.0)

	def reset(self):
		return self._encode(self._env.reset())
	
	def step(self, action):
		return self._encode(self._env.step(action))

	def __getattr__(self, name):
		return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
	def __init__(self, env, dtype):
		self._env = env
		wrapped_action_spec = env.action_spec()
		self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
											   dtype,
											   wrapped_action_spec.minimum,
											   wrapped_action_spec.maximum,
											   'action')

	def step(self, action):
		action = action.astype(self._env.action_spec().dtype)
		return self._env.step(action)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._action_spec

	def reset(self):
		return self._env.reset()

	def __getattr__(self, name):
		return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
	def __init__(self, env):
		self._env = env

	def reset(self):
		time_step = self._env.reset()
		return self._augment_time_step(time_step)

	def step(self, action):
		time_step = self._env.step(action)
		return self._augment_time_step(time_step, action)

	def _augment_time_step(self, time_step, action=None):
		if action is None:
			action_spec = self.action_spec()
			action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
		return ExtendedTimeStep(observation=time_step.observation,
								step_type=time_step.step_type,
								action=action,
								reward=time_step.reward or 0.0,
								discount=time_step.discount or 1.0)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def __getattr__(self, name):
		return getattr(self._env, name)


class TimeStepToGymWrapper(object):
	def __init__(self, env, domain, task, cfg):
		try: # pixels
			obs_shp = env.observation_spec().shape
			assert cfg.modality in {'pixels', 'features'}
		except: # state
			obs_shp = []
			for v in env.observation_spec().values():
				try:
					shp = v.shape[0]
				except:
					shp = 1
				obs_shp.append(shp)
			obs_shp = (np.sum(obs_shp),)
			assert cfg.modality != 'pixels'
		act_shp = env.action_spec().shape
		self.observation_space = gym.spaces.Box(
			low=np.full(obs_shp, -np.inf if cfg.modality != 'pixels' else env.observation_spec().minimum),
			high=np.full(obs_shp, np.inf if cfg.modality != 'pixels' else env.observation_spec().maximum),
			shape=obs_shp,
			dtype=np.float32 if cfg.modality != 'pixels' else np.uint8)
		self.action_space = gym.spaces.Box(
			low=np.full(act_shp, env.action_spec().minimum),
			high=np.full(act_shp, env.action_spec().maximum),
			shape=act_shp,
			dtype=env.action_spec().dtype)
		self.env = env
		self.domain = domain
		self.task = task
		self.ep_len = cfg.episode_length
		self.modality = cfg.modality
		self.t = 0
	
	@property
	def unwrapped(self):
		return self.env.unwrapped

	@property
	def reward_range(self):
		return None

	@property
	def metadata(self):
		return None
	
	def _obs_to_array(self, obs):
		if self.modality == 'state':
			return np.concatenate([v.flatten() for v in obs.values()])
		return obs

	def reset(self):
		self.t = 0
		return self._obs_to_array(self.env.reset().observation)
	
	def step(self, action):
		self.t += 1
		time_step = self.env.step(action)
		return self._obs_to_array(time_step.observation), time_step.reward, time_step.last() or self.t == self.ep_len, defaultdict(float)

	def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
		camera_id = dict(quadruped=2).get(self.domain, camera_id)
		return self.env.physics.render(height, width, camera_id)


class DefaultDictWrapper(gym.Wrapper):
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		return obs, reward, done, defaultdict(float, info)


class FrameEmitterWrapper(gym.Wrapper):
	def __init__(self, env, domain, enabled=False):
		gym.Wrapper.__init__(self, env)
		self._enabled = enabled
		self._frames = []
	
	@property
	def frames(self):
		return self._frames
	
	def _save_frame(self):
		if self._enabled:
			self._frames.append(self.env.render(
				mode='rgb_array',
				height=84,
				width=84,
				camera_id=dict(quadruped=2).get(self.domain, 0)
			))
	
	def reset(self):
		self._frames = []
		obs = self.env.reset()
		self._save_frame()
		return obs
	
	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		self._save_frame()
		return obs, reward, done, info


class MultitaskEnv(object):
	def __init__(self, cfg):
		domain, task = cfg.task.replace('-', '_').split('_', 1)
		domain = dict(cup='ball_in_cup').get(domain, domain)
		self.multitask = cfg.multitask
		if not cfg.multitask:
			tasks = [(domain, task)]
		elif domain in {'walker', 'cheetah'}:
			tasks = [(domain, _task) for _task in suite.TASKS_BY_DOMAIN[domain]]
			assert cfg.num_tasks <= len(tasks), f'num_tasks={cfg.num_tasks} but only {len(tasks)} tasks available'
			tasks = tasks[:cfg.num_tasks]
		self._tasks = ['-'.join(tup).replace('_', '-') for tup in tasks]
		task_kwargs = {'random': cfg.seed}
		if cfg.get('infinite_horizon', False):
			task_kwargs.update({'time_limit': float('inf')})
		self._envs = [suite.load(domain, task, task_kwargs=task_kwargs, visualize_reward=False) for domain, task in tasks]
		self._task_id = 0
		self._env = self._envs[self._task_id]
	
	@property
	def task(self):
		return self._tasks[self._task_id]

	@property
	def tasks(self):
		return self._tasks

	@property
	def unwrapped(self):
		return self
	
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
	
	def reset(self):
		return self._env.reset()
	
	def step(self, action):
		return self._env.step(action)
	
	def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
		return self._env.render(mode, width, height, camera_id)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def __getattr__(self, name):
		return getattr(self._env, name)


def make_env(cfg):
	"""
	Make environment for TD-MPC experiments.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	domain, task = cfg.task.replace('-', '_').split('_', 1)
	domain = dict(cup='ball_in_cup').get(domain, domain)

	if cfg.get('multitask', False):
		env = MultitaskEnv(cfg)
	elif cfg.get('distractors', False):
		from distracting_control import suite as dc_suite
		env = dc_suite.load(
			domain,
			task,
			task_kwargs={'random': cfg.seed},
			visualize_reward=False,
			dynamic=True,
			difficulty=0.2,
			background_dataset_paths=['/home/nh/data/DAVIS/JPEGImages/480p'],
			background_dataset_videos='train',
		)
	else:
		env = suite.load(domain,
						task,
						task_kwargs={'random': cfg.seed},
						visualize_reward=False)
	env = ActionDTypeWrapper(env, np.float32)
	env = ActionRepeatWrapper(env, cfg.action_repeat)
	env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

	if cfg.modality in {'pixels', 'features'}:
		if (domain, task) in suite.ALL_TASKS:
			camera_id = dict(quadruped=2).get(domain, 0)
			render_kwargs = dict(height=84, width=84, camera_id=camera_id)
			env = pixels.Wrapper(env,
								pixels_only=True,
								render_kwargs=render_kwargs)
		env = FrameStackWrapper(env, cfg.get('frame_stack', 1))
	env = ExtendedTimeStepWrapper(env)
	if cfg.modality == 'features':
		env = FeaturesWrapper(env, cfg.get('frame_stack', 1), cfg.get('features', None))
	env = TimeStepToGymWrapper(env, domain, task, cfg)

	env = DefaultDictWrapper(env)
	env = FrameEmitterWrapper(env, domain, enabled=cfg.get('demo', False))
	cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
	cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
	cfg.action_dim = env.action_space.shape[0]

	return env
