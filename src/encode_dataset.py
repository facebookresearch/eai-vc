import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from cfg import parse_cfg
from dataloader import DMControlDataset
from termcolor import colored
from logger import make_dir
from tqdm import tqdm
import torch.nn as nn
import torchvision
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__, __DATA__ = 'cfgs', 'logs', 'data'


__ENCODER__ = None
__PREPROCESS__ = None


def encode_clip(obs):
	"""Encode one or more observations using CLIP."""
	global __ENCODER__, __PREPROCESS__
	assert torch.cuda.is_available(), 'CUDA is not available'
	if __ENCODER__ is None:
		print('Loading CLIP model...')
		import clip
		__ENCODER__, __PREPROCESS__ = clip.load('ViT-B/32', device='cuda') # 151M params
	assert isinstance(obs, (torch.Tensor, np.ndarray)), 'Observation must be a tensor or numpy array'
	if isinstance(obs, torch.Tensor):
		obs = obs.cpu().numpy()
	assert obs.ndim >= 3, 'Observation must be at least 3D'
	
	# Preprocess
	if obs.ndim == 4:
		obses = []
		for _obs in obs:
			_obs = __PREPROCESS__(Image.fromarray(_obs))
			obses.append(_obs)
		obs = torch.stack(obses).cuda()
	else:
		obs = __PREPROCESS__(Image.fromarray(obs)).cuda().unsqueeze(0)

	# Encode
	with torch.no_grad():
		features = __ENCODER__.encode_image(obs)
	return features


def encode_rn50(obs):
	"""Encode one or more observations using RN50."""
	global __ENCODER__, __PREPROCESS__
	assert torch.cuda.is_available(), 'CUDA is not available'
	if __ENCODER__ is None:
		__ENCODER__ = torchvision.models.resnet50(pretrained=True).cuda() # 24M params
		__ENCODER__.fc = nn.Identity()
		__ENCODER__.eval()
		__PREPROCESS__ = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	assert isinstance(obs, (torch.Tensor, np.ndarray)), 'Observation must be a tensor or numpy array'
	if isinstance(obs, np.ndarray):
		obs = torch.from_numpy(obs)
	assert obs.ndim >= 3, 'Observation must be at least 3D'

	# Preprocess
	if obs.ndim == 3:
		obs = obs.unsqueeze(0)
	obs = obs.permute(0, 3, 1, 2).cuda() / 255.
	obs = __PREPROCESS__(obs)

	# Encode
	with torch.no_grad():
		features = __ENCODER__(obs)
	return features


def encode(cfg):
	"""Encoding an image dataset using a pretrained model."""
	assert torch.cuda.is_available()
	assert cfg.modality == 'pixels'
	from env import make_env
	_env = make_env(cfg)
	print(colored(f'\nTask: {cfg.task}', 'blue', attrs=['bold']))

	# Load dataset
	partitions = ['iterations=0', 'iterations=1', 'iterations=2',
				  'iterations=3', 'iterations=4', 'iterations=5',
				  'iterations=6', 'variable_std=0.3', 'variable_std=0.5']
	dataset = DMControlDataset(cfg, Path().cwd() / __DATA__, tasks=[cfg.task], partitions=partitions)
	features_to_fn = {
		'clip': encode_clip,
		'rn50': encode_rn50
	}
	fn = features_to_fn[cfg.features]
	
	# Encode dataset
	for episode in tqdm(dataset.episodes):
		obs = episode.obs[:, -3:].permute(0, 2, 3, 1)
		features = fn(obs).cpu().numpy()
		data = torch.load(episode.filepath)
		if 'features' not in data:
			data['features'] = {}
		data['features'][cfg.features] = features
		torch.save(data, episode.filepath)


if __name__ == '__main__':
	cfg = parse_cfg(Path().cwd() / __CONFIG__)
	encode(cfg)
