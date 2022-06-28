import numpy as np
import torch
import torch.nn as nn
import algorithm.helper as h
from env import make_env


class Unflatten(nn.Module):
	def __init__(self, size):
		super(Unflatten, self).__init__()
		self.size = size

	def forward(self, x):
		return x.view(-1, *self.size)


class LatentToState(nn.Module):
	"""Predict state observation from latent representation."""
	def __init__(self, cfg):
		super().__init__()
		self._encoder = h.enc(cfg)
		self._pred = h.mlp(cfg.latent_dim, 1024, 18)
		self.apply(h.orthogonal_init)
	
	def forward(self, latent):
		return self._pred(self._encoder(latent))


class Renderer():
	"""Rendering states from latent representations."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.latent2state = LatentToState(cfg).cuda()
		self.optim = torch.optim.Adam(self.latent2state.parameters(), lr=self.cfg.lr)
		self.latent2state.eval()
		self.env = make_env(cfg)
		self.unwrapped_env = self.env.env.env.env._env._env._env._env._env._env
		if cfg.modality == 'features':
			self.unwrapped_env = self.unwrapped_env._env

	def state_dict(self):
		return {'latent2state': self.latent2state.state_dict()}

	def save(self, fp):
		torch.save(self.state_dict(), fp)
	
	def load(self, fp):
		d = torch.load(fp)
		self.latent2state.load_state_dict(d['latent2state'])

	@torch.no_grad()
	def render(self, input, from_state=False):
		if not from_state:
			input = self.latent2state(input.cuda())
		input = input.cpu().numpy()
		pixels = []
		for i in range(input.shape[0]):
			state = input[i]
			self.unwrapped_env.physics.set_state(state)
			self.unwrapped_env.physics.forward()
			pixels.append(torch.from_numpy(self.env.render(height=384, width=384).copy()))
		return torch.stack(pixels).permute(0,3,1,2)/255.

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
