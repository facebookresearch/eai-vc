import numpy as np
import torch
import torch.nn as nn
import algorithm.helper as h


class Policy(nn.Module):
	"""Deterministic policy."""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = h.enc(cfg)
		self._task_encoder = h.task_enc(cfg)
		self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
		self.apply(h.orthogonal_init)

	def h(self, obs):
		return self._encoder(obs)

	def pi(self, z, task_vec=None, std=0):
		assert task_vec is not None
		if task_vec is not None:
			z = z + self._task_encoder(task_vec)
		return torch.tanh(self._pi(z))


class BC():
	"""Behavior cloning."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.model = Policy(cfg).cuda()
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
		self.aug = h.RandomShiftsAug(cfg)
		self.model.eval()

	def state_dict(self):
		return {'model': self.model.state_dict()}

	def save(self, fp):
		torch.save(self.state_dict(), fp)
	
	def load(self, fp):
		d = torch.load(fp)
		self.model.load_state_dict(d['model'])

	@torch.no_grad()
	def plan(self, obs, task_vec=None, eval_mode=False, step=None, t0=True, demo_std_fraction=0):
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		if task_vec is not None:
			task_vec = torch.tensor(task_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
		return self.model.pi(self.model.h(obs), task_vec)

	def update(self, replay_buffer, step=int(1e6)):
		obs, next_obses, action, _, _, _, task_vec, idxs, weights = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Compute loss
		obses = torch.cat((obs.unsqueeze(0), next_obses), dim=0)
		total_loss = 0
		for t in range(self.cfg.horizon):
			a = self.model.pi(self.model.h(self.aug(obses[t])), task_vec[t])
			total_loss += (self.cfg.rho ** t) * h.mse(a, action[t]).mean(dim=1)

		# Optimize model
		weighted_loss = (total_loss.clamp(max=1e4) * weights).mean()
		weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
		weighted_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.optim.step()
		replay_buffer.update_priorities(idxs, total_loss.clamp(max=1e4).detach().unsqueeze(1))

		self.model.eval()
		return {'total_loss': float(total_loss.mean().item()),
				'weighted_loss': float(weighted_loss.mean().item()),
				'grad_norm': float(grad_norm)}
