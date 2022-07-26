import os
import glob
import numpy as np
import torch
from pathlib import Path
from collections import deque
from PIL import Image
from torch.utils.data import Dataset
from algorithm.helper import Episode
from logger import make_dir
from tqdm import tqdm
import matplotlib.pyplot as plt


def stack_frames(source, target, num_frames):
	frames = deque([], maxlen=num_frames)
	for _ in range(num_frames):
		frames.append(source[0])
	target[0] = np.concatenate(list(frames), axis=0)
	for i in range(1, target.shape[0]):
		frames.append(source[i])
		target[i] = np.concatenate(list(frames), axis=0)
	return target


def summary_stats(rewards):
	stats = {'mean': rewards.mean(), 'std': rewards.std(), 'min': rewards.min(), 'max': rewards.max(), 'median': np.median(rewards), 'n': len(rewards)}
	for k, v in stats.items():
		stats[k] = round(float(v), 2) if isinstance(v, (torch.FloatTensor, float)) else v
	return stats


class OfflineDataset(Dataset):
	def __init__(self, cfg, tasks='*', buffer=None):
		self._cfg = cfg
		self._tasks = tasks if tasks != '*' else sorted(os.listdir(self._data_dir))
		
		# Locate and filter episodes
		self._fps = self._locate_episodes()
		self._filter_episodes()

		# Optionally dump filelist
		self._dump_filelist()

		# Optionally use a subset of episodes
		idxs = np.random.choice(len(self._fps), int(self._cfg.fraction*len(self._fps)), replace=False)
		self._fps = [self._fps[i] for i in idxs]

		# Load episodes
		self._buffer = buffer
		if cfg.get('lazy_load', False):
			self._buffer.init(self._fps)
		else:
			self._load_episodes()
			self._load_into_buffer()
	
	def _locate_episodes(self):
		raise NotImplementedError()

	def _filter_episodes(self):
		raise NotImplementedError()

	def _load_episodes(self):
		raise NotImplementedError()

	def _partition_episodes(self, datas, cumrews):
		if self._cfg.multitask or self._cfg.get('use_all', False):
			return datas, cumrews, range(len(datas)), None, None, None
		assert len(datas) in {int(1650*self._cfg.fraction), int(3300*self._cfg.fraction)}, 'Unexpected number of episodes: {}'.format(len(datas))
		train_episodes = int((3280 if self._cfg.task.startswith('mw-') else 1500)*self._cfg.fraction)
		train_idxs = torch.topk(cumrews, k=train_episodes, dim=0, largest=False).indices
		print('Training on bottom {} episodes'.format(train_episodes))
		print(f'Training returns: [{cumrews[train_idxs].min():.2f}, {cumrews[train_idxs].max():.2f}]')
		assert len(datas)-train_episodes > 0, 'Not enough validation episodes'
		val_idxs = torch.topk(cumrews, k=len(datas)-train_episodes, dim=0, largest=True).indices
		if self._cfg.get('use_val', False):
			print('Validation on top {} episodes'.format(len(datas)-train_episodes))
			print(f'Validation returns: [{cumrews[val_idxs].min():.2f}, {cumrews[val_idxs].max():.2f}]')
		return [datas[i] for i in train_idxs], cumrews[train_idxs], train_idxs, \
			   [datas[i] for i in val_idxs], cumrews[val_idxs], val_idxs

	def _dump_filelist(self):
		raise NotImplementedError()

	def _dump_histogram(self, cumulative_rewards):
		raise NotImplementedError()
		 
	def _load_into_buffer(self):
		if self._buffer is None:
			return
		for episode in self._episodes:
			self._buffer += episode

	@property
	def tasks(self):
		return self._tasks

	@property
	def buffer(self):
		return self._buffer

	@property
	def episodes(self):
		return self._episodes
	
	@property
	def cumrew(self):
		return self._cumulative_rewards

	@property
	def summary(self):
		return summary_stats(self.cumrew)

	def __getitem__(self, idx):
		return self._episodes[idx]

	def __len__(self):
		return len(self._episodes)


class DMControlDataset(OfflineDataset):
	def __init__(self, cfg, buffer=None):
		self._data_dir = Path(cfg.data_dir) / 'dmcontrol'
		tasks = cfg.task_list if cfg.get('multitask', False) else [cfg.task]
		if len(tasks) > 1:
			print('Tasks: {}'.format(tasks))
		super().__init__(cfg, tasks, buffer)
	
	def _locate_episodes(self):
		return sorted(glob.glob(str(self._data_dir / '*/*/*.pt')))

	def _filter_episodes(self):
		print('Found {} episodes before filtering'.format(len(self._fps)))
		if self._tasks != '*':
			self._fps = [fp for fp in self._fps if np.any([f'/{t}/' in fp for t in self._tasks])]
		print('Found {} episodes after filtering'.format(len(self._fps)))

	def _dump_filelist(self):
		filelist_fp = self._cfg.get('dump_filelist', False)
		if not filelist_fp:
			return
		filelist = []
		for fp in tqdm(self._fps, desc='Dumping filelist'):
			data = torch.load(fp)
			frames_dir = Path(os.path.dirname(fp)) / 'frames'
			assert frames_dir.exists(), 'No frames directory found for {}'.format(fp)
			filelist.extend([frames_dir / fn for fn in data['frames']])
		keep_num = 1_000_000
		idxs = np.random.choice(len(filelist), keep_num, replace=False)
		filelist = [filelist[i] for i in idxs]
		with open(filelist_fp, 'w') as f:
			for fp in filelist:
				f.write(str(fp) + '\n')
		print('Dumped {} frames to {}'.format(len(filelist), filelist_fp))
		exit(0)

	def _dump_histogram(self, cumulative_rewards):
		if not self._cfg.get('dump_histogram', False):
			return
		hist_fp = make_dir(Path(self._cfg.logging_dir) / 'histograms') / f'{self._cfg.task}.png'
		plt.figure(figsize=(8, 4))
		plt.hist(cumulative_rewards, bins=400 if self._cfg.task.startswith('mw-') else 200)
		plt.xlabel('Episode return')
		plt.ylabel('Count')
		plt.xlim(0, 5000 if self._cfg.task.startswith('mw-') else 1000)
		plt.title(self._cfg.task_title.split(' ', 1)[1] + f' ({len(cumulative_rewards)} episodes)')
		plt.tight_layout()
		plt.savefig(hist_fp)
		plt.close()
		print(f'Dumped histogram to {hist_fp}')
		exit(0)

	def _load_episodes(self):
		datas = []
		cumrews = []
		for fp in tqdm(self._fps, 'Loading metadata'):
			data = torch.load(fp)
			datas.append(data)
			cumrew = np.array(data['rewards']).sum()
			cumrews.append(cumrew)
		print('Loaded {} episodes'.format(len(datas)))
		cumrews = np.array(cumrews)
		self._dump_histogram(cumrews)
		datas, self._cumulative_rewards, idxs, val_datas, self._val_cumulative_rewards, val_idxs = \
			self._partition_episodes(datas, torch.tensor(cumrews, dtype=torch.float32))
		self._episodes = []

		def load_episode(data, idx):
			fp = self._fps[idx]
			if self._cfg.get('lazy_load', False):
				assert self._cfg.modality in {'features', 'pixels'} and self._cfg.get('multitask', False), \
					f'Unexpected lazy load for modality {self._cfg.modality} and multitask={self._cfg.get("multitask", False)}'
				obs = None
			else:
				if self._cfg.modality == 'features' or self._cfg.get('all_modalities', False):
					assert self._cfg.get('features', None) is not None, 'Features must be specified'
					features_dir = Path(os.path.dirname(fp)) / 'features' / self._cfg.features
					assert features_dir.exists(), 'No features directory found for {}'.format(fp)
					obs = torch.load(features_dir / os.path.basename(fp))
					_obs = np.empty((obs.shape[0], self._cfg.frame_stack*obs.shape[1]), dtype=np.float32)
					obs = stack_frames(obs, _obs, self._cfg.frame_stack)
					data['metadata']['states'] = data['states']
					if 'phys_states' in data:
						data['metadata']['phys_states'] = data['phys_states']
					if self._cfg.get('all_modalities', False):
						data['metadata']['features'] = obs
				if self._cfg.modality == 'pixels' or self._cfg.get('all_modalities', False):
					frames_dir = Path(os.path.dirname(fp)) / 'frames'
					assert frames_dir.exists(), 'No frames directory found for {}'.format(fp)
					frame_fps = [frames_dir / fn for fn in data['frames']]
					obs = np.stack([np.array(Image.open(fp)) for fp in frame_fps]).transpose(0, 3, 1, 2)
					data['metadata']['states'] = data['states']
					if 'phys_states' in data:
						data['metadata']['phys_states'] = data['phys_states']
					if self._cfg.get('all_modalities', False):
						data['metadata']['pixels'] = obs
						if self._cfg.modality == 'features':
							obs = data['metadata']['features']
				elif self._cfg.modality == 'state':
					obs = data['states']
			actions = np.array(data['actions'], dtype=np.float32).clip(-1, 1)
			if self._cfg.get('multitask', False):
				task = fp.split('/')[-2]
				data['metadata']['task'] = task
				if self._cfg.modality == 'state' and obs is not None and obs[0].shape[0] < self._cfg.obs_shape[0]:
					obs = [np.concatenate([_obs, np.zeros((self._cfg.obs_shape[0] - _obs.shape[0],))]) for _obs in obs]
				if actions.shape[-1] < self._cfg.action_dim:
					actions = np.concatenate([actions, np.zeros((actions.shape[0], self._cfg.action_dim - actions.shape[-1]))], axis=-1)
			episode = Episode.from_trajectory(self._cfg, obs, actions, data['rewards'])
			episode.info = data['infos']
			episode.metadata = data['metadata']
			episode.task_vec = torch.tensor([float(data['metadata']['cfg']['task'] == self._tasks[i]) for i in range(len(self._tasks))], dtype=torch.float32, device=episode.device)
			episode.task_id = episode.task_vec.argmax()
			episode.filepath = fp
			return episode

		for data, idx in tqdm(zip(datas, idxs), desc='Loading episodes'):
			try:
				fp = self._fps[idx]
				episode = load_episode(data, idx)
				self._episodes.append(episode)
			except Exception as e:
				print(f'Failed to load episode {fp}, error: {e}')

		print('Training with {} episodes'.format(len(self._episodes)))

		if self._cfg.get('use_val', False):
			self._val_episodes = []
			for data, idx in tqdm(zip(val_datas, val_idxs), desc='Loading validation episodes'):
				episode = load_episode(data, idx)
				self._val_episodes.append(episode)


def make_dataset(cfg, buffer=None):
	return DMControlDataset(cfg, buffer)
