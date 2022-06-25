import numpy as np
import torch
import torch.nn as nn
import algorithm.helper as h
import torchvision.transforms as transforms
from env import make_env


class Unflatten(nn.Module):
	def __init__(self, size):
		super(Unflatten, self).__init__()
		self.size = size

	def forward(self, x):
		return x.view(-1, *self.size)


class StateToPixels(nn.Module):
	"""Predict pixels from state observation."""
	def __init__(self):
		super().__init__()
		self._encoder = nn.Sequential(
			nn.Linear(24, 1024), nn.ELU(),
			nn.Linear(1024, 512), nn.ELU(),
			nn.Linear(512, 50), nn.ELU())
		self._pred = nn.Sequential(nn.Linear(50, 32*16*16), nn.ReLU(), Unflatten((32, 16, 16)),
			nn.ConvTranspose2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
			nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
			nn.ConvTranspose2d(128, 256, 1), nn.ReLU(),
			nn.ConvTranspose2d(256, 128, 3, padding=1), nn.ReLU(),
			nn.ConvTranspose2d(128, 64, 3, padding=1), nn.ReLU(),
			nn.ConvTranspose2d(64, 32, 3, padding=1), nn.ReLU(),
			nn.ConvTranspose2d(32, 16, 3, padding=1), nn.ReLU(),
			nn.ConvTranspose2d(16, 3, 3, padding=1), nn.ReLU())
		self.load_state_dict(torch.load('/private/home/nihansen/code/tdmpc2/recon/renderer_27000.pt')['model'])

	def forward(self, state):
		return torch.tanh(self._pred(self._encoder(state)))


class LatentToState(nn.Module):
	"""Predict state observation from latent representation."""
	def __init__(self, cfg):
		super().__init__()
		self._encoder = h.enc(cfg)
		self._pred = h.mlp(cfg.latent_dim, 1024, 24)
		self.apply(h.orthogonal_init)
	
	def forward(self, latent):
		return self._pred(self._encoder(latent))


class Renderer():
	"""Rendering states from latent representations."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.latent2state = LatentToState(cfg).to(self.device)
		self.state2pixels = StateToPixels().to(self.device)
		self.optim = torch.optim.Adam(self.latent2state.parameters(), lr=self.cfg.lr)
		self.latent2state.eval()
		self.state2pixels.eval()
		self.env = make_env(cfg)
		self.unwrapped_env = self.env.env.env.env._env._env._env._env._env._env
		if cfg.modality == 'features':
			self.unwrapped_env = self.unwrapped_env._env
		self._resize = transforms.Resize((64, 64))

	def state_dict(self):
		return {'latent2state': self.latent2state.state_dict()}

	def save(self, fp):
		torch.save(self.state_dict(), fp)
	
	def load(self, fp):
		d = torch.load(fp)
		self.model.load_state_dict(d['latent2state'])

	@torch.no_grad()
	def render(self, input, from_state=False):
		input = input.to(self.device)
		if not from_state:
			input = self.latent2state(input)
		return self.state2pixels(input).cpu()

	# def render(self, state):
	# 	if state.is_cuda:
	# 		state = state.cpu()
	# 	self.env.reset()
	# 	self.unwrapped_env.physics.set_state(state.numpy()[:18])
	# 	return torch.from_numpy(self.env.render(mode='rgb_array', width=384, height=384).copy()).permute(2, 0, 1)/255.

	# @torch.no_grad()
	# def render_from_latent(self, z):
	# 	state = self.model.pred(self.model.h(z.unsqueeze(0))).squeeze(0)
	# 	return self.render(state)

	# @torch.no_grad()
	# def render(self, state):
	# 	return self.model.pred(self.model.h(state.cuda())).cpu()

	def update_renderer(self, replay_buffer, step=int(1e6)):
		obs, _, _, _, state, _, _, _ = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Compute loss
		obs_pred = self.model.pred(self.model.h(state))
		obs_target = self._resize(obs[:, -3:] / 255.)
		total_loss = h.l1(obs_pred, obs_target).mean()

		# Optimize model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.optim.step()

		self.model.eval()
		return {'total_loss': float(total_loss.mean().item()),
				'grad_norm': float(grad_norm)}

	def update(self, replay_buffer, step=int(1e6)):
		latent, _, _, _, state_target, _, _, _ = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		self.latent2state.train()

		# Compute loss
		state_pred = self.latent2state(latent)
		total_loss = h.mse(state_pred, state_target).mean()

		# Optimize model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.latent2state.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.optim.step()

		self.latent2state.eval()
		return {'total_loss': float(total_loss.mean().item()),
				'grad_norm': float(grad_norm)}
