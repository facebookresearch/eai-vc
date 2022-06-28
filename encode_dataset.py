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
from collections import deque
import torch.nn as nn
import torchvision
import hydra
import kornia.augmentation as K
from kornia.constants import Resample
torch.backends.cudnn.benchmark = True


__ENCODER__ = None
__PREPROCESS__ = None
__PREPROCESS_EVAL__ = None


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
	if cfg.features == 'clip':
		import clip
		return clip.load('ViT-B/32', device='cuda') # 151M params
	# resnet50: 24M params
	# resnet18: 11M params
	encoder = torchvision.models.__dict__['resnet' + str(18 if '18' in cfg.features else 50)](pretrained=False).cuda()
	if 'moco' in cfg.features:
		if cfg.features == 'moco':
			fn = 'moco_v2_800ep_pretrain.pth.tar'
		elif cfg.features == 'mocodmcontrol':
			fn = 'moco_v2_100ep_pretrain_dmcontrol.pth.tar'
		elif cfg.features == 'mocoego15':
			fn = 'moco_v2_15ep_pretrain_ego4d.pth.tar'
		elif cfg.features == 'mocoego15center':
			fn = 'moco_v2_15ep_pretrain_ego4d.pth.tar'
		elif cfg.features == 'mocoego50':
			fn = 'moco_v2_50ep_pretrain_ego4d.pth.tar'
		elif cfg.features == 'mocoego190':
			fn = 'moco_v2_190ep_pretrain_ego4d.pth.tar'
		elif cfg.features == 'mocodmcontrolmini':
			fn = 'moco_v2_80ep_pretrain_dmcontrolmini.pth.tar'
			encoder.conv1.weight.data = encoder.conv1.weight.data.repeat(1, 3, 1, 1)
			encoder.conv1.in_channels = encoder.conv1.in_channels * 3
		elif cfg.features == 'mocodmcontrol5m':
			fn = 'moco_v2_20ep_pretrain_dmcontrol_5m.pth.tar'
			encoder.conv1.weight.data = encoder.conv1.weight.data.repeat(1, 3, 1, 1)
			encoder.conv1.in_channels = encoder.conv1.in_channels * 3
		else:
			raise ValueError('Unknown MOCO model')
		print(colored('Loading MOCO pretrained model: {}'.format(fn), 'green'))
		state_dict = torch.load(os.path.join(cfg.encoder_dir, fn))['state_dict']
		state_dict = {k.replace('module.encoder_q.', ''): v for k,v in state_dict.items() if not k.startswith('fc.')}
		encoder.load_state_dict(state_dict, strict=False)
	encoder.fc = nn.Identity()
	encoder.eval()
	if cfg.features in {'mocodmcontrol5m', 'mocodmcontrolmini'}:
		preprocess = K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		preprocess_eval = preprocess
	elif cfg.features == 'mocoego15center':
		preprocess_eval = nn.Sequential(
			K.Resize((252, 252), resample=Resample.BILINEAR), K.CenterCrop((224, 224)),
			K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).cuda()
		preprocess = preprocess_eval
	else:
		preprocess = nn.Sequential(
			K.Resize((252, 252), resample=Resample.BILINEAR), K.RandomCrop((224, 224)),
			K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).cuda()
		preprocess_eval = nn.Sequential(
			K.Resize((252, 252), resample=Resample.BILINEAR), K.CenterCrop((224, 224)),
			K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).cuda()

	return encoder, preprocess, preprocess_eval


def encode_clip(obs, cfg, eval=False):
	"""Encode one or more observations using CLIP."""
	global __ENCODER__, __PREPROCESS__, __PREPROCESS_EVAL__
	if __ENCODER__ is None:
		__ENCODER__, __PREPROCESS__, __PREPROCESS_EVAL__ = make_encoder(cfg)
	assert isinstance(obs, (torch.Tensor, np.ndarray)), 'Observation must be a tensor or numpy array'
	if isinstance(obs, torch.Tensor):
		obs = obs.cpu().numpy()
	assert obs.ndim >= 3, 'Observation must be at least 3D'

	# Preprocess
	prep_fn = __PREPROCESS_EVAL__ if eval else __PREPROCESS__
	if obs.ndim == 4:
		obses = []
		for _obs in obs:
			_obs = prep_fn(Image.fromarray(_obs.permute(0, 2, 3, 1)))
			obses.append(_obs)
		obs = torch.stack(obses).cuda()
	else:
		obs = prep_fn(Image.fromarray(obs.permute(0, 2, 3, 1))).cuda().unsqueeze(0)

	# Encode
	with torch.no_grad():
		features = __ENCODER__.encode_image(obs)
	return features


def encode_resnet(obs, cfg, eval=False):
	"""Encode one or more observations using a ResNet model."""
	global __ENCODER__, __PREPROCESS__, __PREPROCESS_EVAL__
	if __ENCODER__ is None:
		__ENCODER__, __PREPROCESS__, __PREPROCESS_EVAL__ = make_encoder(cfg)
	assert isinstance(obs, (torch.Tensor, np.ndarray)), 'Observation must be a tensor or numpy array'
	if isinstance(obs, np.ndarray):
		obs = torch.from_numpy(obs)
	assert obs.ndim >= 3, 'Observation must be at least 3D'
	if obs.ndim == 3:
		obs = obs.unsqueeze(0)
	prep_fn = __PREPROCESS_EVAL__ if eval else __PREPROCESS__
	
	# Encode frame-stacked input
	if __ENCODER__.conv1.in_channels == 9:
		obs = prep_fn(obs / 255.)
		if eval:
			obs = obs.view(1, 3*obs.shape[1], *obs.shape[-2:])
		else:
			_obs = torch.empty((obs.shape[0], __ENCODER__.conv1.in_channels, *obs.shape[-2:]), dtype=torch.float32)
			obs = stack_frames(obs, _obs, cfg.frame_stack)
		obs = obs.cuda()
		with torch.no_grad():
			features = __ENCODER__(obs)
		return features
	
	# Encode single frame input
	with torch.no_grad():
		features = __ENCODER__(prep_fn(obs.cuda() / 255.))
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
