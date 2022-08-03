import numpy as np
import torch
import torch.nn as nn
import algorithm.helper as h
import torchvision.transforms as transforms


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
		self._pred = h.mlp(cfg.latent_dim, 1024, 24)
		self.apply(h.orthogonal_init)
	
	def forward(self, latent):
		return self._pred(latent)


class Renderer():
	"""Rendering states from latent representations."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.decoder = h.dec(cfg).cuda()
		self.optim = torch.optim.Adam(self.decoder.parameters(), lr=self.cfg.lr)
		self.decoder.eval()
		self._resize = transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC)

	def state_dict(self):
		return {'decoder': self.decoder.state_dict(), 'optim': self.optim.state_dict()}

	def save(self, fp):
		torch.save(self.state_dict(), fp)
	
	def load(self, fp):
		d = torch.load(fp)
		self.decoder.load_state_dict(d['decoder'])
		self.optim.load_state_dict(d['optim'])

	def set_tdmpc_agent(self, agent):
		self.agent = agent
		self.agent.model.eval()
		self.agent.model_target.eval()

	@torch.no_grad()
	def render(self, input, from_state=False):
		input = input.cuda()
		if not from_state:
			input = self.latent2state(self.agent.model.h(input))
		return self.state2pixels(input).cpu(), input.cpu()
	
	@torch.no_grad()
	def encode_decode(self, input):
		return self.decoder(self.agent.model.h(input.cuda())).cpu()

	def preprocess_target(self, target):
		if self.cfg.target_modality == 'pixels':
			return self._resize(target[:, -3:]/255.).clip(0, 1)
		return target

	@torch.no_grad()
	def imagine(self, input, actions):
		input, actions = input.cuda(), actions.cuda()
		if input.ndim in {1, 3}:
			input = input.unsqueeze(0)
		assert actions.ndim > 1
		if actions.ndim == 2:
			actions = actions.unsqueeze(1)
		z = self.agent.model.h(input)
		output = []
		for action in actions:
			output.append(self.decoder(z).cpu())
			z, _ = self.agent.model.next(z, action)
		return torch.cat(output, dim=0)

	def update(self, buffer):
		dictionary = buffer.sample({self.cfg.modality, self.cfg.target_modality})
		input = dictionary[self.cfg.modality]
		target = torch.cat((dictionary[self.cfg.target_modality].unsqueeze(0), dictionary['next_'+self.cfg.target_modality]), dim=0)
		action = dictionary['action']
		self.optim.zero_grad(set_to_none=True)
		self.decoder.train()

		with torch.no_grad():
			z = self.agent.model.h(input)

		# Compute loss
		total_loss = 0
		for t in range(self.cfg.horizon):
			total_loss += (self.cfg.rho ** t) * h.mse(self.decoder(z), self.preprocess_target(target[t])).mean()
			z, _ = self.agent.model.next(z, action[t])

		# Optimize model
		total_loss = total_loss.clamp(max=1e4).mean()
		total_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
		total_loss.backward()
		self.optim.step()
		
		self.decoder.eval()
		return {'total_loss': float(total_loss.mean().item())}
