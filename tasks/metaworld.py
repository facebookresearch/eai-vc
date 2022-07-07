import gym
from gym.wrappers import TimeLimit
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN


class MetaWorldWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		if cfg.modality == 'pixels':
			raise NotImplementedError()
		elif cfg.modality == 'features':
			raise NotImplementedError()
		else: # state
			self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space
		self.camera_name = 'corner2'
		self.env.model.cam_pos[2]=[0.75, 0.075, 0.7]
		self.success = False
	
	def _get_pixel_obs(self):
		return self.render(width=self.cfg.img_size, height=self.cfg.img_size)
	
	def reset(self):
		self.success = False
		obs = self.env.reset()
		if self.cfg.modality == 'pixels':
			obs = self._get_pixel_obs()
		return obs
	
	def step(self, action):
		reward = 0
		for _ in range(self.cfg.action_repeat):
			obs, r, _, info = self.env.step(action)
			reward += r
		if self.cfg.modality == 'pixels':
			obs = self._get_pixel_obs()
		self.success = self.success or bool(info['success'])
		return obs, reward, False, info
	
	def render(self, mode='rgb_array', width=None, height=None, camera_id=None):
		return self.env.render(offscreen=True, resolution=(width, height), camera_name=self.camera_name)

	def observation_spec(self):
		return self.observation_space

	def action_spec(self):
		return self.action_space

	def __getattr__(self, name):
		return getattr(self._env, name)


def make_metaworld_env(cfg):
	env_id = cfg.task.split('-', 1)[-1] + '-v2-goal-hidden'
	env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id](seed=cfg.seed)
	env = MetaWorldWrapper(env, cfg)
	env = TimeLimit(env, max_episode_steps=cfg.episode_length)
	return env
