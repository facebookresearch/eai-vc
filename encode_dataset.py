from email.policy import default
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
from cfg_parse import parse_cfg
from dataloader import DMControlDataset
from termcolor import colored
from logger import make_dir
from tqdm import tqdm
import torch.nn as nn
import torchvision
import hydra
import kornia.augmentation as K
from kornia.constants import Resample
torch.backends.cudnn.benchmark = True


__ENCODER__ = None
__PREPROCESS__ = None


def make_encoder(cfg):
	"""Make an encoder."""
	assert torch.cuda.is_available(), 'CUDA is not available'
	assert cfg.get('features', None) is not None, 'Features must be specified'
	if cfg.features == 'clip':
		import clip
		return clip.load('ViT-B/32', device='cuda') # 151M params
	# resnet50: 24M params
	# resnet18: 11M params
	encoder = torchvision.models.__dict__['resnet' + str(18 if '18' in cfg.features else 50)](pretrained=False).cuda()
	if cfg.features == 'moco':
		state_dict = torch.load(os.path.join(cfg.encoder_dir, 'moco_v2_800ep_pretrain.pth.tar'))['state_dict']
		state_dict = {k.replace('module.encoder_q.', ''): v for k,v in state_dict.items() if not k.startswith('fc.')}
		encoder.load_state_dict(state_dict, strict=False)
	encoder.fc = nn.Identity()
	encoder.eval()
	preprocess = nn.Sequential(
		K.Resize((252, 252), resample=Resample.BILINEAR), K.RandomCrop((224, 224)),
		K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).cuda()

	return encoder, preprocess


def encode_clip(obs, cfg):
	"""Encode one or more observations using CLIP."""
	global __ENCODER__, __PREPROCESS__
	if __ENCODER__ is None:
		__ENCODER__, __PREPROCESS__ = make_encoder(cfg)
	assert isinstance(obs, (torch.Tensor, np.ndarray)), 'Observation must be a tensor or numpy array'
	if isinstance(obs, torch.Tensor):
		obs = obs.cpu().numpy()
	assert obs.ndim >= 3, 'Observation must be at least 3D'
	
	# Preprocess
	if obs.ndim == 4:
		obses = []
		for _obs in obs:
			_obs = __PREPROCESS__(Image.fromarray(_obs.permute(0, 2, 3, 1)))
			obses.append(_obs)
		obs = torch.stack(obses).cuda()
	else:
		obs = __PREPROCESS__(Image.fromarray(obs.permute(0, 2, 3, 1))).cuda().unsqueeze(0)

	# Encode
	with torch.no_grad():
		features = __ENCODER__.encode_image(obs)
	return features


def encode_resnet(obs, cfg):
	"""Encode one or more observations using a ResNet model."""
	global __ENCODER__, __PREPROCESS__
	if __ENCODER__ is None:
		__ENCODER__, __PREPROCESS__ = make_encoder(cfg)
	assert isinstance(obs, (torch.Tensor, np.ndarray)), 'Observation must be a tensor or numpy array'
	if isinstance(obs, np.ndarray):
		obs = torch.from_numpy(obs)
	assert obs.ndim >= 3, 'Observation must be at least 3D'

	# Prepare
	if obs.ndim == 3:
		obs = obs.unsqueeze(0)
	obs = obs.cuda()

	# Encode
	with torch.no_grad():
		features = __ENCODER__(__PREPROCESS__(obs / 255.))
	return features


@hydra.main(config_name='default', config_path='config')
def encode(cfg: dict):
	"""Encoding an image dataset using a pretrained model."""
	assert cfg.get('features', None), 'Features must be specified'
	cfg.modality = 'pixels'
	cfg = parse_cfg(cfg)
	from env import make_env
	_env = make_env(cfg)
	print(colored(f'\nTask: {cfg.task}', 'blue', attrs=['bold']))

	# Load dataset
	tasks = _env.unwrapped.tasks if cfg.get('multitask', False) else [cfg.task]
	dataset = DMControlDataset(cfg, Path(cfg.data_dir) / 'dmcontrol', tasks=tasks, fraction=cfg.fraction)
	features_to_fn = defaultdict(lambda: encode_resnet)
	features_to_fn.update({'clip': encode_clip})
	fn = features_to_fn[cfg.features]

	# Encode dataset
	for episode in tqdm(dataset.episodes):
		
		# Compute features
		features = fn(episode.obs, cfg).cpu().numpy()

		# Save features
		feature_dir = make_dir(Path(os.path.dirname(episode.filepath)) / 'features' / cfg.features)
		torch.save(features, feature_dir / os.path.basename(episode.filepath))


if __name__ == '__main__':
	encode()
