import numpy as np
import torch
import torch.nn as nn
import algorithm.helper as h
from env import make_env


class StatePredictor(nn.Module):
	"""Predict GT state from latent representation"""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = h.enc(cfg)
		self._task_encoder = h.task_enc(cfg)
		self._pred = h.mlp(cfg.latent_dim, cfg.mlp_dim, 18)
		self.apply(h.orthogonal_init)

	def h(self, obs):
		return self._encoder(obs)

	def pred(self, z):
		return torch.tanh(self._pred(z))


class Renderer():
	"""Rendering states from latent representations."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.model = StatePredictor(cfg).cuda()
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
		self.model.eval()
		self.env = make_env(cfg)
		self.unwrapped_env = self.env.env.env.env._env._env._env._env._env._env._env

	def state_dict(self):
		return {'model': self.model.state_dict()}

	def save(self, fp):
		torch.save(self.state_dict(), fp)
	
	def load(self, fp):
		d = torch.load(fp)
		self.model.load_state_dict(d['model'])

	def render(self, state):
		if state.is_cuda:
			state = state.cpu()
		self.env.reset()
		self.unwrapped_env.physics.set_state(state.numpy()[:18])
		return torch.from_numpy(self.env.render(mode='rgb_array', width=384, height=384).copy()).permute(2, 0, 1)/255.

	@torch.no_grad()
	def render_from_latent(self, z):
		state = self.model.pred(self.model.h(z.unsqueeze(0))).squeeze(0)
		return self.render(state)

	def update(self, replay_buffer, step=int(1e6)):
		obs, _, _, _, state, _, _, _ = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Compute loss
		state_pred = self.model.pred(self.model.h(obs))
		total_loss = h.mse(state_pred, state[:,:18]).mean(dim=1)

		# Optimize model
		total_loss.mean().backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.optim.step()

		self.model.eval()
		return {'total_loss': float(total_loss.mean().item()),
				'grad_norm': float(grad_norm)}
