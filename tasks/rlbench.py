import numpy as np
import torch
import random
import gym
from gym.wrappers import TimeLimit
import string
import os
# os.environ['DISPLAY'] = ':0.0'
os.environ['DISPLAY'] = ':99'
os.environ['XDG_RUNTIME_DIR'] = '/private/home/nihansen/code/tdmpc2/xvfb/xvfb-run-' + ''.join(random.choices(string.ascii_letters + string.digits, k=8))
from logger import make_dir
make_dir(os.environ['XDG_RUNTIME_DIR'])
import subprocess
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget, PushButton, CloseDrawer, CloseMicrowave, StraightenRope


class RLBench(object):
	def __init__(self, cfg):
		self.cfg = cfg
		self.domain, self.task = cfg.task.replace('-', '_').split('_', 1)
		assert self.domain == 'rlb', 'domain must be rlb (RLBench)'
		if self.cfg.modality == 'state':
			self.cameras = ['front_rgb']
			# self.cameras = []
		else:
			self.cameras = [
				# 'left_shoulder_rgb',
				# 'right_shoulder_rgb',
				# 'overhead_rgb',
				'wrist_rgb',
				'front_rgb',
			]
		img_size = cfg.get('img_size', 84)
		obs_cfg = ObservationConfig()
		camera_cfgs = [c for c in obs_cfg.__dict__ if c.endswith('camera') and c.replace('_camera', '_rgb') in self.cameras]
		unused_camera_cfgs = [c for c in obs_cfg.__dict__ if c.endswith('camera') and c.replace('_camera', '_rgb') not in self.cameras]
		for c in camera_cfgs:
			camera = getattr(obs_cfg, c)
			camera.image_size = (img_size, img_size)
			camera.depth = False
			camera.point_cloud = False
			camera.mask = False
		for c in unused_camera_cfgs:
			camera = getattr(obs_cfg, c)
			camera.set_all(False)
		obs_cfg.task_low_dim_state = True
		subprocess.Popen(['Xvfb', os.environ['DISPLAY'], '-screen', '0', '1024x768x24', '+extension', 'GLX', '+render', '-noreset'])
		# Xvfb :99 -screen 0 1024x768x24 +extension GLX +render -noreset &
		print('init self.sim')
		self.sim = Environment(
			action_mode=MoveArmThenGripper(
				arm_action_mode=JointPosition(absolute_mode=False), gripper_action_mode=Discrete()),
			obs_config=obs_cfg,
			shaped_rewards=self.task=='reach_target',
			headless=True)
		print('self.sim.launch()')
		self.sim.launch()
		tasks = {
			'reach_target': ReachTarget,
			'push_button': PushButton,
			'close_drawer': CloseDrawer,
			'close_microwave': CloseMicrowave,
			'straighten_rope': StraightenRope
		}
		print('self.env = self.sim.get_task(tasks[self.task])')
		self.env = self.sim.get_task(tasks[self.task])
		if cfg.modality == 'pixels':
			self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=np.array([len(self.cameras)*3, img_size, img_size]), dtype='uint8')
		elif cfg.modality == 'features':
			raise NotImplementedError()
		else:
			print('self.env.reset()')
			self.env.reset()
			print('self.env.get_low_dim_state()')
			state = self.env._task.get_low_dim_state()
			print('state.shape', state.shape)
			self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=np.array([state.shape[0]]), dtype='float32')
		self.action_space = gym.spaces.Box(-1., 1., shape=(int(self.sim.action_shape[0]),), dtype='float32')
		self._current_frame = None
		self._prev_obs = None
	
	@property
	def unwrapped(self):
		return self.env.unwrapped
	
	def _get_obs(self, obs):
		# frames = torch.cat([torch.tensor(obs.__dict__[o]) for o in self.cameras], dim=-1).permute(2, 0, 1)
		# self._current_frame = frames[-3:].permute(1,2,0)
		# if self.cfg.modality == 'pixels':
			# obs = frames
		# elif self.cfg.modality == 'features':
			# raise NotImplementedError()
		# else:
		obs = obs.task_low_dim_state
		self._prev_obs = obs
		return obs
	
	def reset(self):
		_, obs = self.env.reset()
		return self._get_obs(obs)

	def step(self, action):
		reward = 0
		try:
			for _ in range(self.cfg.action_repeat):
				obs, r, done = self.env.step(action)
				if self.task not in {'reach_target'}:
					r = float(self.env._task.success()[0])
				reward += r
				if done:
					break
		except (IKError, ConfigurationPathError, InvalidActionError):
			return self._prev_obs, reward, False, {'success': self.env._task.success()[0]}
		return self._get_obs(obs), reward, False, {'success': self.env._task.success()[0]}

	def render(self, mode='rgb_array', width=None, height=None, camera_id=None):
		# return self._current_frame
		return np.zeros((84, 84, 3), dtype='uint8')

	def observation_spec(self):
		return self._obs_spec

	def action_spec(self):
		return self._action_spec
	
	@property
	def spec(self):
		return None

	def __getattr__(self, name):
		return getattr(self._env, name)


def make_rlbench_env(cfg):
	env = RLBench(cfg)
	cfg.cameras = env.cameras
	env = TimeLimit(env, max_episode_steps=cfg.episode_length)
	return env
