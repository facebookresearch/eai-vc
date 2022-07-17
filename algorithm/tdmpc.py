import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import algorithm.helper as h


class TOLD(nn.Module):
	"""Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = h.enc(cfg)
		self._task_encoder = h.task_enc(cfg)
		self._dynamics = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
		self._reward = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
		self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
		self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
		self.apply(h.orthogonal_init)
		for m in [self._reward, self._Q1, self._Q2]:
			m[-1].weight.data.fill_(0)
			m[-1].bias.data.fill_(0)

	def track_q_grad(self, enable=True):
		"""Utility function. Enables/disables gradient tracking of Q-networks."""
		for m in [self._Q1, self._Q2]:
			h.set_requires_grad(m, enable)

	def h(self, obs):
		"""Encodes an observation into its latent representation (h)."""
		return self._encoder(obs)

	def next(self, z, a, task_vec=None):
		"""Predicts next latent state (d) and single-step reward (R)."""
		zd, zr = z, z
		if self.cfg.detach_rewval:
			zr = zr.detach()
		if task_vec is not None:
			zt = self._task_encoder(task_vec)
			zd, zr = zd + zt, zr + zt
		xd = torch.cat([zd, a], dim=-1)
		xr = torch.cat([zr, a], dim=-1)
		return self._dynamics(xd), self._reward(xr)

	def pi(self, z, task_vec=None, std=0):
		"""Samples an action from the learned policy (pi)."""
		if task_vec is not None:
			z = z + self._task_encoder(task_vec)
		mu = torch.tanh(self._pi(z))
		if std > 0:
			std = torch.ones_like(mu) * std
			return h.TruncatedNormal(mu, std).sample(clip=0.3)
		return mu

	def Q(self, z, a, task_vec=None):
		"""Predict state-action value (Q)."""
		if task_vec is not None:
			z = z + self._task_encoder(task_vec)
		x = torch.cat([z, a], dim=-1)
		return self._Q1(x), self._Q2(x)


class ReconstructionTOLD(TOLD):
	"""TOLD but with a decoder."""
	def __init__(self, cfg):
		super().__init__(cfg)
		self._decoder = h.dec(cfg)
		self._decoder.apply(h.orthogonal_init)
	
	def g(self, z):
		"""Decodes a latent representation (z) into an observation (g)."""
		return self._decoder(z)


class TDMPC():
	"""Implementation of TD-MPC learning + inference."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.std = h.linear_schedule(cfg.std_schedule, 0)
		if self.cfg.dynamics_obj == 'consistency':
			self.model = TOLD(cfg).cuda()
		elif self.cfg.dynamics_obj == 'reconstruction':
			self.model = ReconstructionTOLD(cfg).cuda()
		else:
			raise ValueError('Unknown dynamics objective: {}'.format(self.cfg.dynamics_obj))
		self.model_target = deepcopy(self.model)
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
		self.aug = h.RandomShiftsAug(cfg)
		self.model.eval()
		self.model_target.eval()
		print('Model parameters: {}'.format(sum(p.numel() for p in self.model.parameters())))

	def state_dict(self):
		"""Retrieve state dict of TOLD model, including slow-moving target network."""
		return {'model': self.model.state_dict(),
				'model_target': self.model_target.state_dict(),
				'optim': self.optim.state_dict(),
				'pi_optim': self.pi_optim.state_dict()}

	def save(self, fp, metadata={}):
		"""Save state dict of TOLD model to filepath."""
		state_dict = self.state_dict()
		state_dict['metadata'] = metadata
		torch.save(state_dict, fp)
	
	def load(self, fp):
		"""Load a saved state dict from filepath (or dictionary) into current agent."""
		d = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(d['model'])
		self.model_target.load_state_dict(d['model_target'])
		self.optim.load_state_dict(d['optim'])
		self.pi_optim.load_state_dict(d['pi_optim'])
		return d['metadata']

	@torch.no_grad()
	def estimate_value(self, z, actions, task_vec, horizon):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(horizon):
			z, reward = self.model.next(z, actions[t], task_vec)
			G += discount * reward
			discount *= self.cfg.discount
		G += discount * torch.min(*self.model.Q(z, self.model.pi(z, task_vec, self.cfg.min_std), task_vec))
		return G

	@torch.no_grad()
	def plan(self, obs, task_vec=None, eval_mode=False, step=None, t0=True, open_loop=False):
		"""
		Plan next action using TD-MPC inference.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		open_loop: whether to use open-loop dynamics.
		"""
		# Seed steps
		if step < self.cfg.seed_steps and not eval_mode:
			return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)
		
		# Sample policy trajectories
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		if task_vec is not None:
			task_vec = torch.tensor(task_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
		num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
		if num_pi_trajs > 0:
			pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
			z = self.model.h(obs).repeat(num_pi_trajs, 1)
			for t in range(horizon):
				pi_actions[t] = self.model.pi(z, task_vec, self.cfg.min_std)
				z, _ = self.model.next(z, pi_actions[t], task_vec)

		# Initialize state and parameters
		z = self.model.h(obs).repeat(self.cfg.num_samples+num_pi_trajs, 1)
		mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
		std = 2*torch.ones(horizon, self.cfg.action_dim, device=self.device)
		if not t0 and hasattr(self, '_prev_mean'):
			mean[:-1] = self._prev_mean[1:]

		# Iterate CEM
		for i in range(self.cfg.iterations):
			actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device), -1, 1)
			if num_pi_trajs > 0:
				actions = torch.cat([actions, pi_actions], dim=1)

			# Compute elite actions
			value = self.estimate_value(z, actions, task_vec, horizon).nan_to_num_(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score /= score.sum(0)
			_mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			_std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
			_std = _std.clamp_(self.std, 2)
			mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std
		
		# Outputs
		score = score.squeeze(1).cpu().numpy()
		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
		if open_loop:
			return actions
		self._prev_mean = mean
		mean, std = actions[0], _std[0]
		a = mean
		if not eval_mode:
			a += std * torch.randn(self.cfg.action_dim, device=std.device)
		return a.clamp_(-1, 1)

	def update_pi(self, zs, task_vec):
		"""Update policy using a sequence of latent states."""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		# Loss is a weighted sum of Q-values
		pi_loss = 0
		for t,z in enumerate(zs):
			a = self.model.pi(z, task_vec, self.cfg.min_std)
			Q = torch.min(*self.model.Q(z, a, task_vec))
			pi_loss += -Q.mean() * (self.cfg.rho ** t)

		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.pi_optim.step()
		self.model.track_q_grad(True)
		return pi_loss.item()

	@torch.no_grad()
	def _td_target(self, next_obs, reward, task_vec=None):
		"""Compute the TD-target from a reward and the observation at the following time step."""
		next_z = self.model.h(next_obs)
		td_target = reward + self.cfg.discount * \
			torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, task_vec, self.cfg.min_std), task_vec))
		return td_target

	def update(self, replay_buffer, step=int(1e6)):
		"""Main update function. Corresponds to one iteration of the TOLD model learning."""
		obs, next_obses, action, reward, _, _, task_vec, idxs, weights = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		self.std = h.linear_schedule(self.cfg.std_schedule, step)
		self.model.train()

		# Representation
		z = self.model.h(self.aug(obs))
		zs = [z.detach()]

		consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
		for t in range(self.cfg.horizon):

			# Predictions
			Q1, Q2 = self.model.Q(z.detach() if self.cfg.detach_rewval else z, action[t], task_vec)
			z, reward_pred = self.model.next(z, action[t], task_vec)
			with torch.no_grad():
				next_obs = self.aug(next_obses[t])
				next_z = self.model_target.h(next_obs)
				td_target = self._td_target(next_obs, reward[t], task_vec)
			zs.append(z.detach())

			# Losses
			rho = (self.cfg.rho ** t)
			if self.cfg.dynamics_obj == 'consistency':
				consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
			elif self.cfg.dynamics_obj == 'reconstruction':
				obs_pred = self.model.g(z)
				consistency_loss += rho * torch.mean(h.mse(obs_pred, next_obs), dim=1, keepdim=True)
			else:
				raise ValueError('Unknown dynamics objective: {}'.format(self.cfg.dynamics_obj))
			reward_loss += rho * h.mse(reward_pred, reward[t])
			value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
			priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))

		# Optimize model
		total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
					 self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
					 self.cfg.value_coef * value_loss.clamp(max=1e4)
		weighted_loss = (total_loss * weights).mean()
		weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
		weighted_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.optim.step()
		if self.cfg.per:
			replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

		# Update policy + target network
		pi_loss = self.update_pi(zs, task_vec)
		if step % self.cfg.update_freq == 0:
			h.soft_update_params(self.model, self.model_target, self.cfg.tau)

		self.model.eval()
		return {'consistency_loss': float(consistency_loss.mean().item()),
				'reward_loss': float(reward_loss.mean().item()),
				'value_loss': float(value_loss.mean().item()),
				'pi_loss': pi_loss,
				'total_loss': float(total_loss.mean().item()),
				'weighted_loss': float(weighted_loss.mean().item()),
				'grad_norm': float(grad_norm)}
