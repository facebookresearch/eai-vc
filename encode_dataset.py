import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
from pathlib import Path
from cfg_parse import parse_cfg
from dataloader import make_dataset
from termcolor import colored
from logger import make_dir
from tqdm import tqdm
from collections import deque
import torch.nn as nn
import torchvision
import hydra
import kornia.augmentation as K
from kornia.constants import Resample
torch.backends.cudnn.benchmark = True


__ENCODER__ = None
__PREPROCESS__ = None


def stack_frames(source, target, num_frames):
    frames = deque([], maxlen=num_frames)
    for _ in range(num_frames):
        frames.append(source[0])
    target[0] = torch.cat(list(frames), dim=0)
    for i in range(1, target.shape[0]):
        frames.append(source[i])
        target[i] = torch.cat(list(frames), dim=0)
    return target


def make_encoder(cfg):
	"""Make an encoder."""
	assert torch.cuda.is_available(), 'CUDA is not available'
	assert cfg.get('features', None) is not None, 'Features must be specified'
	if 'mae' in cfg.features:
		import mvp
		encoder = mvp.load("vits-mae-hoi").cuda()
		encoder.freeze()
	else:
		encoder = torchvision.models.__dict__['resnet50'](pretrained=False).cuda()
		if 'moco' in cfg.features:
			if cfg.features == 'moco':
				fn = 'moco_v2_800ep_pretrain.pth.tar'
			elif cfg.features == 'mocodmcontrol':
				fn = 'moco_v2_100ep_dmcontrol.pt'
			elif cfg.features == 'mocometaworld':
				fn = 'moco_v2_33ep_metaworld.pt'
			elif cfg.features == 'mocoego':
				fn = 'moco_v2_15ep_pretrain_ego4d.pth.tar'
			elif cfg.features == 'mocoegodmcontrol':
				fn = 'moco_v2_15ep_pretrain_ego_dmcontrol_finetune.pth.tar'
			else:
				raise ValueError('Unknown MOCO model')
			print(colored('Loading MOCO pretrained model: {}'.format(fn), 'green'))
			state_dict = torch.load(os.path.join(cfg.encoder_dir, fn))['state_dict']
			state_dict = {k.replace('module.encoder_q.', ''): v for k,v in state_dict.items() if not k.startswith('fc.')}
			encoder.load_state_dict(state_dict, strict=False)
		encoder.fc = nn.Identity()
		encoder.eval()
	preprocess = nn.Sequential(
		K.Resize((224, 224), resample=Resample.BICUBIC),
		K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).cuda()
	return encoder, preprocess


def encode_mae(obs, cfg):
	"""Encode one or more observations using a MAE ViT-S model."""
	global __ENCODER__, __PREPROCESS__
	if __ENCODER__ is None:
		__ENCODER__, __PREPROCESS__ = make_encoder(cfg)
	assert isinstance(obs, (torch.Tensor, np.ndarray)), 'Observation must be a tensor or numpy array'
	if isinstance(obs, np.ndarray):
		obs = torch.from_numpy(obs)
	assert obs.ndim >= 3, 'Observation must be at least 3D'
	if obs.ndim == 3:
		obs = obs.unsqueeze(0)
	with torch.no_grad():
		features = __ENCODER__(__PREPROCESS__(obs.cuda() / 255.))
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
	if obs.ndim == 3:
		obs = obs.unsqueeze(0)
	with torch.no_grad():
		features = __ENCODER__(__PREPROCESS__(obs.cuda() / 255.))
	return features


def encode(obs, cfg):
	"""Encode one or more observations using a pretrained model."""
	features2encoder = {'maehoi': encode_mae}
	fn = features2encoder.get(cfg.features, encode_resnet)
	return fn(obs, cfg)


@hydra.main(config_name='default', config_path='config')
def main(cfg: dict):
	"""Encoding an image dataset using a pretrained model."""
	assert cfg.get('features', None), 'Features must be specified'
	cfg.modality = 'pixels'
	cfg = parse_cfg(cfg)
	from env import make_env
	_env = make_env(cfg)
	print(colored(f'\nTask: {cfg.task}', 'blue', attrs=['bold']))

	# Load dataset
	dataset = make_dataset(cfg)

	# Encode dataset
	for episode in tqdm(dataset.episodes):
		
		# Compute features
		features = encode(episode.obs, cfg).cpu().numpy()

		# Save features
		feature_dir = make_dir(Path(os.path.dirname(episode.filepath)) / 'features' / cfg.features)
		torch.save(features, feature_dir / os.path.basename(episode.filepath))


if __name__ == '__main__':
	main()
