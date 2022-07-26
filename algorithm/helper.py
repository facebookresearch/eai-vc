from email.policy import default
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import defaultdict
from pathlib import Path
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from torch.utils.data import IterableDataset


__REDUCE__ = lambda b: 'mean' if b else 'none'


def l1(pred, target, reduce=False):
	"""Computes the L1-loss between predictions and targets."""
	return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def _get_out_shape(in_shape, layers):
	"""Utility function. Returns the output shape of a network for a given input shape."""
	x = torch.randn(*in_shape).unsqueeze(0)
	return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape


def orthogonal_init(m):
	"""Orthogonal layer initialization."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	elif isinstance(m, nn.Conv2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if m.bias is not None:
			nn.init.zeros_(m.bias)


def soft_update_params(m, m_target, tau):
	"""Update slow-moving average of online network (target network) at rate tau."""
	with torch.no_grad():
		for p, p_target in zip(m.parameters(), m_target.parameters()):
			p_target.data.lerp_(p.data, tau)


def set_requires_grad(net, value):
	"""Enable/disable gradients for a given (sub)network."""
	for param in net.parameters():
		param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
	"""Utility class implementing the truncated normal distribution."""
	def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
		super().__init__(loc, scale, validate_args=False)
		self.low = low
		self.high = high
		self.eps = eps

	def _clamp(self, x):
		clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
		x = x - x.detach() + clamped_x.detach()
		return x

	def sample(self, clip=None, sample_shape=torch.Size()):
		shape = self._extended_shape(sample_shape)
		eps = _standard_normal(shape,
							   dtype=self.loc.dtype,
							   device=self.loc.device)
		eps *= self.scale
		if clip is not None:
			eps = torch.clamp(eps, -clip, clip)
		x = self.loc + eps
		return self._clamp(x)


class NormalizeImg(nn.Module):
	"""Module that divides (pixel) observations by 255."""
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.)


class Flatten(nn.Module):
	"""Module that flattens its input to a (batched) vector."""
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


class Unflatten(nn.Module):
	def __init__(self, size):
		super(Unflatten, self).__init__()
		self.size = size

	def forward(self, x):
		return x.view(-1, *self.size)


class Flare(nn.Module):
	"""Flow of latents."""
	def __init__(self, latent_dim, num_frames):
		super().__init__()
		assert num_frames in {2, 3}
		self.latent_dim = latent_dim
		self.num_frames = num_frames
	
	def forward(self, x):
		assert x.shape[-1] == self.latent_dim*self.num_frames
		x = x.view(x.size(0), self.num_frames, self.latent_dim)
		deltas = x[:, 1:] - x[:, :-1]
		if self.num_frames == 3:
			ddelta = (x[:, -1] - x[:, 0]).unsqueeze(1)
			dddelta = (deltas[:, -1] - deltas[:, 0]).unsqueeze(1)
			return torch.cat([x, deltas, ddelta, dddelta], dim=1).view(x.size(0), -1)
		elif self.num_frames == 2:
			return torch.cat([x, deltas], dim=1).view(x.size(0), -1)
		else:
			raise ValueError('Invalid number of frames: {}'.format(self.num_frames))


class FeatureFuse(nn.Module):
	"""Feature fusion and encoder layer."""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		assert cfg.modality == 'features'
		features_to_dim = defaultdict(lambda: 2048)
		features_to_dim.update({
			'clip': 512,
			'random18': 1024,
		})
		self.fn = nn.Sequential(
			nn.Linear(features_to_dim[cfg.features], cfg.enc_dim), nn.ELU(),
			nn.Linear(cfg.enc_dim, cfg.enc_dim))
		if cfg.frame_stack == 1:
			self.layers = nn.Sequential(
				nn.ELU(), nn.Linear(cfg.enc_dim, cfg.enc_dim), nn.ELU(),
				nn.Linear(cfg.enc_dim, cfg.latent_dim))
		else:
			self.layers = nn.Sequential(
				Flare(cfg.enc_dim, cfg.frame_stack),
				nn.Linear(cfg.enc_dim*(7 if cfg.frame_stack == 3 else 3), cfg.enc_dim), nn.ELU(),
				nn.Linear(cfg.enc_dim, cfg.latent_dim))

	def forward(self, x):
		b = x.size(0)
		x = x.view(b*self.cfg.frame_stack, x.size(1)//self.cfg.frame_stack)
		x = self.fn(x)
		x = x.view(b, self.cfg.enc_dim*self.cfg.frame_stack)
		return self.layers(x)


def enc(cfg):
	"""Returns a TOLD encoder."""
	if cfg.modality == 'pixels':
		if cfg.encoder == 'default+':
			C = int(3*cfg.frame_stack)
			layers = [NormalizeImg(),
					nn.Conv2d(C, cfg.num_channels, 3, stride=2), nn.ReLU(),
					nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=1), nn.ReLU(),
					nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=1), nn.ReLU(),
					nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=1), nn.ReLU()]
			out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
			layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.latent_dim)])
		elif cfg.encoder == 'default':
			C = int(3*cfg.frame_stack)
			layers = [NormalizeImg(),
					nn.Conv2d(C, cfg.num_channels, 7, stride=2), nn.ReLU(),
					nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2), nn.ReLU(),
					nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU(),
					nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU()]
			out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
			layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.latent_dim)])
		else:
			raise ValueError('Unknown encoder arch: {}'.format(cfg.encoder))
	elif cfg.modality == 'features':
		layers = [FeatureFuse(cfg)]
	else:
		layers = [nn.Linear(cfg.obs_shape[0]+(cfg.latent_dim if cfg.get('multitask', False) else 0), cfg.enc_dim), nn.ELU(),
				  nn.Linear(cfg.enc_dim, cfg.latent_dim)]
	print('Encoder parameters: {}'.format(sum(p.numel() for p in nn.Sequential(*layers).parameters())))
	return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
	def __init__(self, C):
		super().__init__()
		self.layers = nn.Sequential(
			nn.ConvTranspose2d(C, C, 3, stride=1, padding=1), nn.ReLU(),
			nn.ConvTranspose2d(C, C, 3, stride=1, padding=1), nn.ReLU())
	
	def forward(self, x):
		return x + self.layers(x)


def dec(cfg):
	"""Returns a TOLD decoder."""
	if cfg.target_modality == 'pixels':
		layers = [nn.Linear(cfg.latent_dim, cfg.mlp_dim), nn.ELU(),
				  nn.Linear(cfg.mlp_dim, cfg.enc_dim), nn.ELU(),
				  nn.Linear(cfg.enc_dim, 128), nn.ELU(),
				  nn.Linear(128, 32*16*16), nn.ReLU(), Unflatten((32, 16, 16)),
				  nn.ConvTranspose2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
				  nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
				  ResidualBlock(128),
				  nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1), nn.ReLU(),
				  ResidualBlock(64),
				  nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1), nn.Sigmoid()]
	elif cfg.target_modality == 'features':
		features_to_dim = defaultdict(lambda: 2048)
		features_to_dim.update({
			'clip': 512,
			'random18': 1024,
		})
		layers = [nn.Linear(cfg.latent_dim, cfg.mlp_dim), nn.ELU(),
				  nn.Linear(cfg.mlp_dim, cfg.enc_dim), nn.ELU(),
				  nn.Linear(cfg.enc_dim, features_to_dim[cfg.features]*cfg.frame_stack)]
	else:
		layers = [nn.Linear(cfg.latent_dim, cfg.mlp_dim), nn.ELU(),
				  nn.Linear(cfg.mlp_dim, cfg.enc_dim), nn.ELU(),
				  nn.Linear(cfg.enc_dim, cfg.state_dim)]
	print('Decoder parameters: {}'.format(sum(p.numel() for p in nn.Sequential(*layers).parameters())))
	return nn.Sequential(*layers)

def task_enc(cfg):
	"""Returns a task encoder."""
	if not cfg.multitask:
		return None
	return nn.Sequential(
		nn.Linear(cfg.num_tasks, cfg.enc_dim), nn.ELU(),
		nn.Linear(cfg.enc_dim, cfg.latent_dim)
	)


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
	"""Returns an MLP."""
	if isinstance(mlp_dim, int):
		mlp_dim = [mlp_dim, mlp_dim]
	return nn.Sequential(
		nn.Linear(in_dim, mlp_dim[0]), act_fn,
		nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
		nn.Linear(mlp_dim[1], out_dim))


def q(cfg, act_fn=nn.ELU()):
	"""Returns a Q-function that uses Layer Normalization."""
	return nn.Sequential(nn.Linear(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim), nn.LayerNorm(cfg.mlp_dim), nn.Tanh(),
						 nn.Linear(cfg.mlp_dim, cfg.mlp_dim), act_fn,
						 nn.Linear(cfg.mlp_dim, 1))


class RandomShiftsAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, cfg):
		super().__init__()
		self.pad = int(cfg.img_size/21) if cfg.modality == 'pixels' else None

	def forward(self, x):
		if not self.pad:
			return x
		n, c, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class Episode(object):
	"""Storage object for a single episode."""
	def __init__(self, cfg, init_obs):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		dtype = torch.uint8 if cfg.modality == 'pixels' else torch.float32
		if not cfg.get('lazy_load', False):
			self.obs = torch.empty((cfg.episode_length+1, *init_obs.shape), dtype=dtype, device=self.device)
			self.obs[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
		self.action = torch.empty((cfg.episode_length, cfg.action_dim), dtype=torch.float32, device=self.device)
		self.reward = torch.empty((cfg.episode_length,), dtype=torch.float32, device=self.device)
		self.cumulative_reward = 0
		self.done = False
		self._idx = 0

	@classmethod
	def from_trajectory(cls, cfg, obs, action, reward, done=None):
		"""Constructs an episode from a trajectory."""
		if cfg.get('lazy_load', False):
			episode = cls(cfg, None)
		else:
			episode = cls(cfg, obs[0])
			episode.obs[1:] = torch.tensor(obs[1:], dtype=episode.obs.dtype, device=episode.device)
		episode.action = torch.tensor(action, dtype=episode.action.dtype, device=episode.device)
		episode.reward = torch.tensor(reward, dtype=episode.reward.dtype, device=episode.device)
		episode.cumulative_reward = torch.sum(episode.reward)
		episode.done = True
		episode._idx = cfg.episode_length+1
		return episode
	
	def __len__(self):
		return self._idx

	@property
	def first(self):
		return len(self) == 0
	
	def __add__(self, transition):
		self.add(*transition)
		return self

	def add(self, obs, action, reward, done):
		self.obs[self._idx+1] = torch.tensor(obs, dtype=self.obs.dtype, device=self.obs.device)
		self.action[self._idx] = action
		self.reward[self._idx] = reward
		self.cumulative_reward += reward
		self.done = done
		self._idx += 1


class ReplayBuffer():
	"""
	Storage and sampling functionality for training TD-MPC / TOLD.
	The replay buffer is stored in GPU memory when training from state.
	Uses prioritized experience replay by default.
	"""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		self.capacity = (cfg.num_tasks if cfg.get('multitask', False) else 1)*1_000_000 + 1
		dtype = torch.uint8 if cfg.modality == 'pixels' else torch.float32
		obs_shape = (3, *cfg.obs_shape[-2:]) if cfg.modality == 'pixels' else cfg.obs_shape
		if self.cfg.get('lazy_load', False):
			self._fps = []
		else:
			self._obs = torch.empty((self.capacity+1, *obs_shape), dtype=dtype, device=self.device)
			self._last_obs = torch.empty((self.capacity//cfg.episode_length, *cfg.obs_shape), dtype=dtype, device=self.device)
		self._action = torch.empty((self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device)
		self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
		self._state = torch.empty((self.capacity, 24), dtype=torch.float32, device=self.device) if cfg.modality != 'state' else None
		self._last_state = torch.empty((self.capacity//cfg.episode_length, 24), dtype=torch.float32, device=self.device) if cfg.modality != 'state' else None
		self._task_vec = torch.empty((self.capacity, cfg.num_tasks), dtype=torch.uint8, device=self.device) if cfg.get('multitask', False) else None
		self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
		self._eps = 1e-6
		self._state_dim = None
		self._full = False
		self.idx = 0
	
	@property
	def full(self):
		return self._full

	def __add__(self, episode: Episode):
		self.add(episode)
		return self

	def add(self, episode: Episode):
		assert not self.full, 'Replay buffer is full'
		if self.cfg.get('lazy_load', False):
			self._fps.append(episode.filepath)
		else:
			self._obs[self.idx:self.idx+self.cfg.episode_length] = episode.obs[:-1, -3:] if self.cfg.modality == 'pixels' else episode.obs[:-1]
			self._last_obs[self.idx//self.cfg.episode_length] = episode.obs[-self.cfg.frame_stack:].view(self.cfg.frame_stack*3, *self.cfg.obs_shape[-2:]) \
				if self.cfg.modality == 'pixels' and episode.obs.shape[1] == 3 else episode.obs[-1]
		self._action[self.idx:self.idx+self.cfg.episode_length] = episode.action
		self._reward[self.idx:self.idx+self.cfg.episode_length] = episode.reward
		if self.cfg.modality != 'state' and 'states' in episode.__dict__:
			states = torch.tensor(episode.metadata['states'])
			self._state_dim = states.size(1)
			self._state[self.idx:self.idx+self.cfg.episode_length, :self._state_dim] = states[:-1]
			self._last_state[self.idx//self.cfg.episode_length, :self._state_dim] = states[-1]
		if self.cfg.multitask:
			self._task_vec[self.idx:self.idx+self.cfg.episode_length] = episode.task_vec.unsqueeze(0).repeat(self.cfg.episode_length, 1)
		if self._full:
			max_priority = self._priorities.max().to(self.device).item()
		else:
			max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
		mask = torch.arange(self.cfg.episode_length) >= self.cfg.episode_length-self.cfg.horizon
		new_priorities = torch.full((self.cfg.episode_length,), max_priority, device=self.device)
		new_priorities[mask] = 0
		self._priorities[self.idx:self.idx+self.cfg.episode_length] = new_priorities
		self.idx = (self.idx + self.cfg.episode_length) % self.capacity
		self._full = self._full or self.idx == 0

	def update_priorities(self, idxs, priorities):
		self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

	def _get_obs(self, arr, idxs):
		if self.cfg.modality != 'pixels':
			return arr[idxs]
		obs = torch.empty((self.cfg.batch_size, 3*self.cfg.frame_stack, *arr.shape[-2:]), dtype=arr.dtype, device=torch.device('cuda'))
		obs[:, -3:] = arr[idxs].cuda()
		_idxs = idxs.clone()
		mask = torch.ones_like(_idxs, dtype=torch.bool)
		for i in range(1, self.cfg.frame_stack):
			mask[_idxs % self.cfg.episode_length == 0] = False
			_idxs[mask] -= 1
			obs[:, -(i+1)*3:-i*3] = arr[_idxs].cuda()
		return obs.float()

	def _lazy_load_obs(self, idxs):
		episode_idxs = idxs // self.cfg.episode_length
		features = torch.empty((self.cfg.batch_size, self.cfg.episode_length+1, self.cfg.obs_shape[0]), dtype=torch.float32)
		for i, episode_idx in enumerate(episode_idxs):
			fp = self._fps[episode_idx]
			feature_dir = Path(os.path.dirname(fp)) / 'features' / self.cfg.features
			features[i] = torch.from_numpy(torch.load(feature_dir / os.path.basename(fp)))
		return features[episode_idxs, idxs % self.cfg.episode_length]

	def sample(self):
		probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
		probs /= probs.sum()
		total = len(probs)
		idxs = torch.from_numpy(np.random.choice(total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
		weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
		weights /= weights.max()

		if self.cfg.get('lazy_load', False):
			obs = self._lazy_load_obs(idxs)
			next_obs_shape = (self.cfg.obs_shape[0],)
		else:
			obs = self._get_obs(self._obs, idxs)
			next_obs_shape = (3*self.cfg.frame_stack, *self._last_obs.shape[-2:]) if self.cfg.modality == 'pixels' else self._last_obs.shape[1:]
		next_obs = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *next_obs_shape), dtype=obs.dtype, device=obs.device)
		action = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *self._action.shape[1:]), dtype=torch.float32, device=self.device)
		reward = torch.empty((self.cfg.horizon+1, self.cfg.batch_size), dtype=torch.float32, device=self.device)
		state = self._state[idxs, :self._state_dim] if self.cfg.modality != 'state' and self._state_dim is not None else None
		next_state = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *state.shape[1:]), dtype=state.dtype, device=state.device) if state is not None else None
		task_vec = self._task_vec[idxs].float() if self.cfg.multitask else None
		for t in range(self.cfg.horizon+1):
			_idxs = idxs + t
			if self.cfg.get('lazy_load', False):
				next_obs[t] = self._lazy_load_obs(_idxs+1)
			else:
				next_obs[t] = self._get_obs(self._obs, _idxs+1)
			action[t] = self._action[_idxs]
			reward[t] = self._reward[_idxs]
			if state is not None:
				next_state[t] = self._state[_idxs+1, :self._state_dim]

		mask = (_idxs+1) % self.cfg.episode_length == 0
		if not self.cfg.get('lazy_load', False):
			next_obs[-1, mask] = self._last_obs[_idxs[mask]//self.cfg.episode_length].to(next_obs.device).float()
		if task_vec is not None:
			task_vec = task_vec.cuda()
		if state is not None:
			state = state.cuda()
			next_state[-1, mask] = self._last_state[_idxs[mask]//self.cfg.episode_length, :self._state_dim].to(next_state.device).float()
			next_state = next_state.cuda()

		return obs.cuda(), next_obs.cuda(), action.cuda(), reward.cuda().unsqueeze(2), state, next_state, task_vec, idxs.cuda(), weights.cuda()


class LazyReplayBufferIterable(IterableDataset):
	"""
	Child process for sampling from disk.
	"""
	def __init__(self, cfg, num_workers=1, max_episodes=10, fetch_every=4):
		self.cfg = cfg
		self.num_workers = num_workers
		self.max_episodes = max_episodes
		self.fetch_every = fetch_every
		self.obs_dtype = torch.uint8 if cfg.modality == 'pixels' else torch.float32
		self.obs_shape = (3, *cfg.obs_shape[-2:]) if cfg.modality == 'pixels' else cfg.obs_shape
		self._fps = []
		self._episodes = [None] * max_episodes
		self._i = 0
		self._worker_id = None

	def set_fps(self, fps):
		self._fps = fps

	def _sample(self):
		# Initialize the worker if first time
		if self._worker_id is None:
			try:
				self._worker_id = torch.utils.data.get_worker_info().id
			except:
				self._worker_id = 0
			total_fps = len(self._fps)
			self._fps = self._fps[self._worker_id::self.num_workers]
			print(f'Worker {self._worker_id} sampling from {len(self._fps)}/{total_fps} episodes')

		# Load episode from disk
		num_load = self.max_episodes if self._i == 0 else int(self._i % self.fetch_every == 0)
		# print('Worker {} loading {} episodes'.format(self._worker_id, num_load))
		for i in range(num_load):
			fp = self._fps[np.random.randint(len(self._fps))]
			self._episodes[(self._i + i) % self.max_episodes] = torch.load(fp)
		self._i += 1

		# Sample a trajectory
		data = self._episodes[np.random.randint(self.max_episodes)]
		idxs = np.random.randint(self.cfg.episode_length-self.cfg.horizon)
		obs = torch.from_numpy(data['states'][idxs])
		next_obs = torch.from_numpy(np.stack(data['states'][idxs+1:idxs+self.cfg.horizon+1]))
		action = torch.from_numpy(np.stack(data['actions'][idxs:idxs+self.cfg.horizon])).float().clip(-1, 1)
		reward = torch.from_numpy(np.stack(data['rewards'][idxs:idxs+self.cfg.horizon])).float()
		return obs, next_obs, action, reward

	def __iter__(self):
		while True:
			yield self._sample()


def _worker_init_fn(worker_id):
	seed = np.random.get_state()[1][0] + worker_id
	np.random.seed(seed)
	random.seed(seed)


class LazyReplayBuffer(ReplayBuffer):
	"""
	Replay buffer that uses a lazy loading mechanism for large buffer sizes.
	Unlike the regular replay buffer, this one does not require the entire buffer to be loaded at once,
	but also does not support prioritized sampling due to the buffer living in multiple processes.
	"""
	def __init__(self, cfg):
		self.cfg = cfg
		self.capacity = 1_000_000
		self.num_workers = 32

	def init(self, fps):
		self._fps = fps
		iterable = LazyReplayBufferIterable(self.cfg,
											num_workers=self.num_workers)
		iterable.set_fps(fps)
		self._loader = torch.utils.data.DataLoader(iterable,
												   batch_size=self.cfg.batch_size,
											 	   num_workers=self.num_workers,
											 	   pin_memory=True,
											 	   worker_init_fn=_worker_init_fn)
		self._iter = None
	
	@property
	def full(self):
		return False

	def __add__(self, episode: Episode):
		return self

	def add(self, episode: Episode):
		return self

	def update_priorities(self, idxs, priorities):
		pass

	@property
	def iter(self):
		if self._iter is None:
			self._iter = iter(self._loader)
		return self._iter

	def sample(self):
		obs, next_obs, action, reward = next(self.iter)
		return obs.cuda().float(), \
			   next_obs.permute(1,0,2).cuda().float(), \
			   action.permute(1,0,2).cuda(), \
			   reward.unsqueeze(2).permute(1,0,2).cuda(), \
			   None, None, None, None, 1.


class RendererBuffer(ReplayBuffer):
	"""
	Storage and sampling functionality for training a renderer."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.capacity = 1_000_001
		self.pixels_shape = (3, cfg.img_size, cfg.img_size)
		self.features_shape = (2048*cfg.frame_stack,)
		self.state_shape = (cfg.state_dim,)
		self._pixels = torch.empty((self.capacity+1, *self.pixels_shape), dtype=torch.uint8)
		self._features = torch.empty((self.capacity+1, *self.features_shape), dtype=torch.float32)
		self._state = torch.empty((self.capacity+1, *self.state_shape), dtype=torch.float32)
		self._action = torch.empty((self.capacity, cfg.action_dim), dtype=torch.float32)
		self._last_pixels = torch.empty((self.capacity//cfg.episode_length, 3*self.cfg.frame_stack, *self.pixels_shape[-2:]), dtype=torch.uint8)
		self._last_features = torch.empty((self.capacity//cfg.episode_length, *self.features_shape), dtype=torch.float32)
		self._last_state = torch.empty((self.capacity//cfg.episode_length, *self.state_shape), dtype=torch.float32)
		self._priorities = torch.ones((self.capacity,), dtype=torch.float32)
		self._full = False
		self.idx = 0

	def add(self, episode: Episode):
		assert not self.full, 'Replay buffer is full'
		self._pixels[self.idx:self.idx+self.cfg.episode_length] = torch.from_numpy(episode.metadata['pixels'][:-1, -3:])
		self._features[self.idx:self.idx+self.cfg.episode_length] = torch.from_numpy(episode.metadata['features'][:-1])
		self._state[self.idx:self.idx+self.cfg.episode_length] = torch.from_numpy(np.array(episode.metadata['states'][:-1]))
		self._action[self.idx:self.idx+self.cfg.episode_length] = episode.action
		self._last_pixels[self.idx//self.cfg.episode_length] = torch.from_numpy(episode.metadata['pixels'][-self.cfg.frame_stack:]).reshape(3*self.cfg.frame_stack, *self.pixels_shape[-2:])
		self._last_features[self.idx//self.cfg.episode_length] = torch.from_numpy(episode.metadata['features'][-1])
		self._last_state[self.idx//self.cfg.episode_length] = torch.from_numpy(episode.metadata['states'][-1])
		mask = torch.arange(self.cfg.episode_length) >= self.cfg.episode_length-self.cfg.horizon
		new_priorities = torch.full((self.cfg.episode_length,), 1.)
		new_priorities[mask] = 0
		self._priorities[self.idx:self.idx+self.cfg.episode_length] = new_priorities
		self.idx = (self.idx + self.cfg.episode_length) % self.capacity
		self._full = self._full or self.idx == 0
	
	def _get_pixel_obs(self, arr, idxs):
		obs = torch.empty((self.cfg.batch_size, 3*self.cfg.frame_stack, *arr.shape[-2:]), dtype=arr.dtype, device=torch.device('cuda'))
		obs[:, -3:] = arr[idxs].cuda()
		_idxs = idxs.clone()
		mask = torch.ones_like(_idxs, dtype=torch.bool)
		for i in range(1, self.cfg.frame_stack):
			mask[_idxs % self.cfg.episode_length == 0] = False
			_idxs[mask] -= 1
			obs[:, -(i+1)*3:-i*3] = arr[_idxs].cuda()
		return obs.float()

	def sample(self, modalities):
		probs = self._priorities[:self.idx] ** self.cfg.per_alpha
		probs /= probs.sum()
		total = len(probs)
		idxs = torch.from_numpy(np.random.choice(total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=False))

		if 'pixels' in modalities:
			pixels = self._get_pixel_obs(self._pixels, idxs)
			next_pixels = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, 3*self.cfg.frame_stack, *self.pixels_shape[-2:]), dtype=torch.float32)
		if 'features' in modalities:
			features = self._features[idxs]
			next_features = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *self.features_shape), dtype=torch.float32)
		if 'state' in modalities:
			state = self._state[idxs]
			next_state = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *self.state_shape), dtype=torch.float32)
		action = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, self.cfg.action_dim), dtype=torch.float32)

		for t in range(self.cfg.horizon+1):
			_idxs = idxs + t
			if 'pixels' in modalities:
				next_pixels[t] = self._get_pixel_obs(self._pixels, _idxs+1)
			if 'features' in modalities:
				next_features[t] = self._features[_idxs+1]
			if 'state' in modalities:
				next_state[t] = self._state[_idxs+1]
			action[t] = self._action[_idxs]

		dictionary = {'action': action.cuda()}
		mask = (_idxs+1) % self.cfg.episode_length == 0
		if 'pixels' in modalities:
			next_pixels[-1, mask] = self._last_pixels[_idxs[mask]//self.cfg.episode_length].float()
			dictionary['pixels'] = pixels.cuda()
			dictionary['next_pixels'] = next_pixels.cuda()
		if 'features' in modalities:
			next_features[-1, mask] = self._last_features[_idxs[mask]//self.cfg.episode_length].float()
			dictionary['features'] = features.cuda()
			dictionary['next_features'] = next_features.cuda()
		if 'state' in modalities:
			next_state[-1, mask] = self._last_state[_idxs[mask]//self.cfg.episode_length].float()
			dictionary['state'] = state.cuda()
			dictionary['next_state'] = next_state.cuda()
		return dictionary
	

def make_buffer(cfg):
	if cfg.get('lazy_load', False):
		return LazyReplayBuffer(cfg)
	else:
		return ReplayBuffer(cfg)


def linear_schedule(schdl, step):
	"""Outputs values following a linear decay schedule"""
	try:
		return float(schdl)
	except ValueError:
		match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
		if match:
			init, final, duration = [float(g) for g in match.groups()]
			mix = np.clip(step / duration, 0.0, 1.0)
			return (1.0 - mix) * init + mix * final
	raise NotImplementedError(schdl)
