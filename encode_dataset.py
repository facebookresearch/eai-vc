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
	if cfg.features == 'clip':
		import clip
		return clip.load('ViT-B/32', device='cuda') # 151M params
	# resnet50: 24M params
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


def encode_clip(obs, cfg, eval=False):
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


def encode_resnet(obs, cfg, eval=False):
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
	
	# Encode single frame input
	with torch.no_grad():
		features = __ENCODER__(__PREPROCESS__(obs.cuda() / 255.))
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
	dataset = make_dataset(cfg)
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
