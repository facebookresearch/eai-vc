from email.policy import default
import os
import glob
import numpy as np
import torch
import pickle as pkl
from pathlib import Path
from collections import deque, defaultdict
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
			return datas, cumrews, range(len(datas))
		assert len(datas) in {int(1650*self._cfg.fraction), int(3300*self._cfg.fraction)}, 'Unexpected number of episodes: {}'.format(len(datas))
		train_episodes = int((3150 if self._cfg.task.startswith('mw-') else 1500)*self._cfg.fraction)
		train_idxs = torch.topk(cumrews, k=train_episodes, dim=0, largest=False).indices
		val_idxs = torch.topk(cumrews, k=len(datas)-train_episodes, dim=0, largest=True).indices
		print('Training on bottom {} episodes'.format(train_episodes))
		print(f'Training returns: [{cumrews[train_idxs].min():.2f}, {cumrews[train_idxs].max():.2f}]')
		print(f'Validation returns: [{cumrews[val_idxs].min():.2f}, {cumrews[val_idxs].max():.2f}]')
		return [datas[i] for i in train_idxs], cumrews[train_idxs], train_idxs

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
		plt.figure(figsize=(8, 5))
		plt.hist(cumulative_rewards, bins=100)
		plt.xlabel('Episode return')
		plt.ylabel('Count')
		plt.xlim(0, 5000 if self._cfg.task.startswith('mw-') else 1000)
		plt.title(f'{self._cfg.task}')
		plt.savefig(hist_fp)
		plt.close()
		print(f'Dumped histogram to {hist_fp}')
		exit(0)

	def _load_episodes(self):
		datas = []
		cumrews = []
		for fp in self._fps:
			data = torch.load(fp)
			datas.append(data)
			cumrew = np.array(data['rewards']).sum()
			cumrews.append(cumrew)
		cumrews = np.array(cumrews)
		self._dump_histogram(cumrews)
		datas, self._cumulative_rewards, idxs = self._partition_episodes(datas, torch.tensor(cumrews, dtype=torch.float32))
		self._episodes = []
		for data, idx in tqdm(zip(datas, idxs), desc='Loading episodes'):
			fp = self._fps[idx]
			if self._cfg.modality == 'features':
				assert self._cfg.get('features', None) is not None, 'Features must be specified'
				features_dir = Path(os.path.dirname(fp)) / 'features' / self._cfg.features
				assert features_dir.exists(), 'No features directory found for {}'.format(fp)
				obs = torch.load(features_dir / os.path.basename(fp))
				_obs = np.empty((obs.shape[0], self._cfg.frame_stack*obs.shape[1]), dtype=np.float32)
				obs = stack_frames(obs, _obs, self._cfg.frame_stack)
				data['metadata']['states'] = data['states']
				if 'phys_states' in data:
					data['metadata']['phys_states'] = data['phys_states']
			elif self._cfg.modality == 'pixels':
				frames_dir = Path(os.path.dirname(fp)) / 'frames'
				assert frames_dir.exists(), 'No frames directory found for {}'.format(fp)
				frame_fps = [frames_dir / fn for fn in data['frames']]
				obs = np.stack([np.array(Image.open(fp)) for fp in frame_fps]).transpose(0, 3, 1, 2)
				data['metadata']['states'] = data['states']
				if 'phys_states' in data:
					data['metadata']['phys_states'] = data['phys_states']
			else:
				obs = data['states']
			actions = np.array(data['actions'], dtype=np.float32).clip(-1, 1)
			if self._cfg.get('multitask', False):
				task = fp.split('/')[-2]
				data['metadata']['task'] = task
				if self._cfg.modality == 'state' and obs[0].shape[0] < self._cfg.obs_shape[0]:
					obs = [np.concatenate([_obs, np.zeros((self._cfg.obs_shape[0] - _obs.shape[0],))]) for _obs in obs]
				if actions.shape[-1] < self._cfg.action_dim:
					actions = np.concatenate([actions, np.zeros((actions.shape[0], self._cfg.action_dim - actions.shape[-1]))], axis=-1)
			episode = Episode.from_trajectory(self._cfg, obs, actions, data['rewards'])
			episode.info = data['infos']
			episode.metadata = data['metadata']
			episode.task_vec = torch.tensor([float(data['metadata']['cfg']['task'] == self._tasks[i]) for i in range(len(self._tasks))], dtype=torch.float32, device=episode.device)
			episode.task_id = episode.task_vec.argmax()
			episode.filepath = fp
			self._episodes.append(episode)


class RLBenchDataset(OfflineDataset):
	def __init__(self, cfg, buffer=None):
		self._data_dir = Path(cfg.data_dir) / 'rlbench'
		tasks = [cfg.task]
		super().__init__(cfg, tasks, None, buffer)
	
	def _locate_episodes(self):
		return sorted(glob.glob(str(self._data_dir / '*/*/episodes/*/low_dim_obs.pkl')))

	def _filter_episodes(self):
		print('Found {} episodes before filtering'.format(len(self._fps)))
		if self._tasks != '*':
			tasks = [t.replace('rlb-', '').replace('-', '_') for t in self._tasks]
			self._fps = [fp for fp in self._fps if np.any([f'/{t}/' in fp for t in tasks])]
		print('Found {} episodes after filtering'.format(len(self._fps)))

	def _load_episodes(self):
		assert self._cfg.modality == 'pixels'
		assert 'cameras' in self._cfg
		for fp in tqdm(self._fps):
			data = pkl.load(open(fp, 'rb'))

			# Load data
			obs = []
			for eplen in range(self._cfg.episode_length+1):
				frame_fps = [Path(os.path.dirname(fp)) / camera / f'{eplen}.png' for camera in self._cfg.cameras]
				if not all(fp.exists() for fp in frame_fps):
					break
				_obs = np.concatenate([np.array(Image.open(fp)) for fp in frame_fps], axis=-1).transpose(2, 0, 1)
				obs.append(_obs)
			obs = np.stack(obs)
			joint_velocities = np.stack([data._observations[i].joint_velocities for i in range(len(data._observations))], axis=0)[1:]
			gripper_open = np.stack([data._observations[i].gripper_open for i in range(len(data._observations))], axis=0)[1:, None]
			actions = np.concatenate([joint_velocities, gripper_open], axis=-1)
			rewards = np.zeros(actions.shape[0], dtype=np.float32)
			rewards[-2:] = 1.
			states = np.stack([data._observations[i].task_low_dim_state[:24] for i in range(len(data._observations))], axis=0)

			# Repeat the last element until the length is equal to the episode length
			if eplen < self._cfg.episode_length:
				obs = np.concatenate([obs, obs[-1:] * np.ones((self._cfg.episode_length - obs.shape[0] + 1, *obs.shape[1:]), dtype=obs.dtype)], axis=0)
				noop = np.zeros_like(actions[-1])
				noop[-1] = actions[-1,-1]
				actions = np.concatenate([actions, noop * np.ones((self._cfg.episode_length - actions.shape[0], actions.shape[1]), dtype=actions.dtype)], axis=0)
				rewards = np.concatenate([rewards, rewards[-1:] * np.ones((self._cfg.episode_length - rewards.shape[0],), dtype=rewards.dtype)], axis=0)
				states = np.concatenate([states, states[-1:] * np.ones((self._cfg.episode_length - states.shape[0] + 1, *states.shape[1:]), dtype=states.dtype)], axis=0)

			episode = Episode.from_trajectory(self._cfg, obs, actions, rewards)
			episode.metadata = {'states': states}
			episode.task_vec = torch.ones(1, dtype=torch.float32, device=episode.device)
			episode.task_id = 0  # TODO
			episode.filepath = fp
			self._episodes.append(episode)
			self._cumulative_rewards.append(episode.cumulative_reward)


def make_dataset(cfg, buffer=None):
	cls = defaultdict(lambda: DMControlDataset, {'rlb': RLBenchDataset})[cfg.domain]
	return cls(cfg, buffer)
