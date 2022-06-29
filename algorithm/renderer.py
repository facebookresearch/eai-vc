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
		# self._encoder = h.enc(cfg)
		self._pred = h.mlp(cfg.latent_dim, 1024, 24)
		self.apply(h.orthogonal_init)
	
	def forward(self, latent):
		# return self._pred(self._encoder(latent))
		return self._pred(latent)


class Renderer():
	"""Rendering states from latent representations."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.latent2state = LatentToState(cfg).cuda()
		self.state2pixels = StateToPixels().cuda()
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
		self.latent2state.load_state_dict(d['latent2state'])

	def set_tdmpc_agent(self, agent):
		self.agent = agent
		self.agent.model.eval()
		self.agent.model_target.eval()
	
	@torch.no_grad()
	def render_sim(self, input, from_state=False):
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

	@torch.no_grad()
	def render(self, input, from_state=False):
		input = input.cuda()
		if not from_state:
			input = self.latent2state(self.agent.model.h(input))
		return self.state2pixels(input).cpu(), input.cpu()

	@torch.no_grad()
	def imagine(self, input, actions):
		input, actions = input.cuda(), actions.cuda()
		if input.ndim == 1:
			input = input.unsqueeze(0)
		assert actions.ndim > 1
		if actions.ndim == 2:
			actions = actions.unsqueeze(0)
		z = self.agent.model.h(input)
		b, T, _ = actions.shape
		states = torch.empty(b, T, 24, dtype=z.dtype, device=z.device)
		pixels = torch.empty(b, T, 3, 64, 64, dtype=z.dtype, device=z.device)
		for t in range(T):
			states[:, t] = self.latent2state(z)
			pixels[:, t] = self.state2pixels(states[:, t])
			z, _ = self.agent.model.next(z, actions[:, t])
		return pixels.cpu(), states.cpu()

	def update(self, replay_buffer, step=int(1e6)):
		features, _, action, _, state_target, next_state_target, _, _, _ = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		self.latent2state.train()

		with torch.no_grad():
			z = self.agent.model.h(features)
		
		# Compute loss
		state_target = torch.cat((state_target.unsqueeze(0), next_state_target), dim=0)
		total_loss = 0
		for t in range(self.cfg.horizon):
			state_pred = self.latent2state(z)
			total_loss += (self.cfg.rho ** t) * h.mse(state_pred, state_target[t]).mean(dim=1)
			z, _ = self.agent.model.next(z, action[t])

		# Optimize model
		total_loss = total_loss.clamp(max=1e4).mean()
		total_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.latent2state.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.optim.step()

		self.latent2state.eval()
		return {'total_loss': float(total_loss.mean().item()),
				'grad_norm': float(grad_norm)}
