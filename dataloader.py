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
import torchvision
from algorithm.helper import Episode
from tqdm import tqdm


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
    def __init__(self, cfg, tasks='*', rbounds=None, buffer=None):
        self._cfg = cfg
        self._tasks = tasks if tasks != '*' else sorted(os.listdir(self._data_dir))
        self._rbounds = rbounds
        
        # Locate and filter episodes
        self._fps = self._locate_episodes()
        self._filter_episodes()

        # Optionally use a subset of episodes
        idxs = np.random.choice(len(self._fps), int(self._cfg.fraction*len(self._fps)), replace=False)
        self._fps = [self._fps[i] for i in idxs]

        # dump_filelist = cfg.get('dump_filelist', None)
        # if dump_filelist:
        #     filelist = []

        # Load episodes
        self._buffer = buffer
        self._episodes = []
        self._cumulative_rewards = []
        self._load_episodes()

        # for fp in tqdm(self._fps):
        #     data = torch.load(fp)
        #     cumr = np.array(data['rewards']).sum()
        #     if rbounds is not None:
        #         if not (rbounds[0] <= cumr <= rbounds[1]):
        #             continue
        #     if cfg.modality == 'features':
        #         assert cfg.get('features', None) is not None, 'Features must be specified'
        #         features_dir = Path(os.path.dirname(fp)) / 'features' / cfg.features
        #         assert features_dir.exists(), 'No features directory found for {}'.format(fp)
        #         obs = torch.load(features_dir / os.path.basename(fp))
        #         if not cfg.features in {'mocodmcontrol5m', 'mocodmcontrolmini'}:
        #             _obs = np.empty((obs.shape[0], cfg.frame_stack*obs.shape[1]), dtype=np.float32)
        #             obs = stack_frames(obs, _obs, cfg.frame_stack)
        #         data['metadata']['states'] = data['states']
        #         if 'phys_states' in data:
        #             data['metadata']['phys_states'] = data['phys_states']
        #     elif cfg.modality == 'pixels':
        #         frames_dir = Path(os.path.dirname(fp)) / 'frames'
        #         assert frames_dir.exists(), 'No frames directory found for {}'.format(fp)
        #         frame_fps = [frames_dir / fn for fn in data['frames']]
        #         if dump_filelist:
        #             obs = np.empty((len(frame_fps), 3*cfg.frame_stack, 84, 84), dtype=np.float32)
        #             filelist.extend(frame_fps)
        #         else:
        #             obs = np.stack([np.array(Image.open(fp)) for fp in frame_fps]).transpose(0, 3, 1, 2)
        #             data['metadata']['states'] = data['states']
        #             if 'phys_states' in data:
        #                 data['metadata']['phys_states'] = data['phys_states']
        #     else:
        #         obs = data['states']
        #     actions = np.array([v['expert_action'] for v in data['infos']] if cfg.get('expert_actions', False) else data['actions'], dtype=np.float32).clip(-1, 1)
        #     episode = Episode.from_trajectory(cfg, obs, actions, data['rewards'])
        #     episode.info = data['infos']
        #     episode.metadata = data['metadata']
        #     episode.task_vec = torch.tensor([float(data['metadata']['cfg']['task'] == tasks[i]) for i in range(len(tasks))], dtype=torch.float32, device=episode.device)
        #     episode.task_id = episode.task_vec.argmax()
        #     episode.filepath = fp
        #     if buffer is not None:
        #         self._buffer += episode
        #     else:
        #         self._episodes.append(episode)
        #     self._cumulative_rewards.append(episode.cumulative_reward)

        self._cumulative_rewards = torch.tensor(self._cumulative_rewards, dtype=torch.float32, device=torch.device('cpu'))
        if rbounds is not None:
            print('Found {} episodes within reward range {}'.format(len(self._cumulative_rewards), rbounds))
        
        # if dump_filelist:
        #     # randomly drop some frames to reduce size
        #     keep_num = 5_000_000
        #     idxs = np.random.choice(len(filelist), keep_num, replace=False)
        #     filelist = [filelist[i] for i in idxs]
        #     with open(dump_filelist, 'w') as f:
        #         for fp in filelist:
        #             f.write(str(fp) + '\n')
        #     print('Dumped {} frames to {}'.format(len(filelist), dump_filelist))
        #     exit(0)
    
    def _locate_episodes(self):
        raise NotImplementedError()

    def _filter_episodes(self):
        raise NotImplementedError()

    def _load_episodes(self):
        raise NotImplementedError()

    @property
    def tasks(self):
        return self._tasks

    @property
    def partitions(self):
        return self._partitions

    @property
    def episodes(self):
        return self._episodes
    
    @property
    def buffer(self):
        return self._buffer
    
    @property
    def cumrew(self):
        return self._cumulative_rewards

    @property
    def summary(self):
        return summary_stats(self.cumrew)

    def __len__(self):
        return len(self._episodes)


class DMControlDataset(OfflineDataset):
    def __init__(self, cfg, buffer=None):
        self._data_dir = Path(cfg.data_dir) / 'dmcontrol'
        tasks = cfg.task_list if cfg.get('multitask', False) else [cfg.task]
        max_bound = {
            'walker-walk': 900,
            'walker-stand': 925,
            'walker-run': 375,
            'walker-arabesque': 800,
            'walker-walk-backwards': 750,
            'walker-run-backwards': 225
        }
        rbounds = [0, max_bound[cfg.task]] if cfg.task in max_bound else None
        super().__init__(cfg, tasks, rbounds, buffer)
    
    def _locate_episodes(self):
        return sorted(glob.glob(str(self._data_dir / '*/*/*.pt')))

    def _filter_episodes(self):
        print('Found {} episodes before filtering'.format(len(self._fps)))
        if self._tasks != '*':
            self._fps = [fp for fp in self._fps if np.any([f'/{t}/' in fp for t in self._tasks])]
        print('Found {} episodes after filtering'.format(len(self._fps)))

    def _load_episodes(self):
        for fp in tqdm(self._fps):
            data = torch.load(fp)
            cumr = np.array(data['rewards']).sum()
            if self._rbounds is not None:
                if not (self._rbounds[0] <= cumr <= self._rbounds[1]):
                    continue
            if self._cfg.modality == 'features':
                assert self._cfg.get('features', None) is not None, 'Features must be specified'
                features_dir = Path(os.path.dirname(fp)) / 'features' / cfg.features
                assert features_dir.exists(), 'No features directory found for {}'.format(fp)
                obs = torch.load(features_dir / os.path.basename(fp))
                if not self._cfg.features in {'mocodmcontrol5m', 'mocodmcontrolmini'}:
                    _obs = np.empty((obs.shape[0], self._cfg.frame_stack*obs.shape[1]), dtype=np.float32)
                    obs = stack_frames(obs, _obs, self._cfg.frame_stack)
                data['metadata']['states'] = data['states']
                if 'phys_states' in data:
                    data['metadata']['phys_states'] = data['phys_states']
            elif self._cfg.modality == 'pixels':
                frames_dir = Path(os.path.dirname(fp)) / 'frames'
                assert frames_dir.exists(), 'No frames directory found for {}'.format(fp)
                frame_fps = [frames_dir / fn for fn in data['frames']]
                # if dump_filelist:
                #     obs = np.empty((len(frame_fps), 3*self._cfg.frame_stack, 84, 84), dtype=np.float32)
                #     filelist.extend(frame_fps)
                # else:
                obs = np.stack([np.array(Image.open(fp)) for fp in frame_fps]).transpose(0, 3, 1, 2)
                data['metadata']['states'] = data['states']
                if 'phys_states' in data:
                    data['metadata']['phys_states'] = data['phys_states']
            else:
                obs = data['states']
            actions = np.array([v['expert_action'] for v in data['infos']] if self._cfg.get('expert_actions', False) else data['actions'], dtype=np.float32).clip(-1, 1)
            episode = Episode.from_trajectory(self._cfg, obs, actions, data['rewards'])
            episode.info = data['infos']
            episode.metadata = data['metadata']
            episode.task_vec = torch.tensor([float(data['metadata']['cfg']['task'] == self._tasks[i]) for i in range(len(self._tasks))], dtype=torch.float32, device=episode.device)
            episode.task_id = episode.task_vec.argmax()
            episode.filepath = fp
            if self._buffer is not None:
                self._buffer += episode
            else:
                self._episodes.append(episode)
            self._cumulative_rewards.append(episode.cumulative_reward)

    def __getitem__(self, idx):
        return self._episodes[idx]

    def __len__(self):
        return len(self._episodes)


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
            if self._buffer is not None:
                self._buffer += episode
            else:
                self._episodes.append(episode)
            self._cumulative_rewards.append(episode.cumulative_reward)
        

    def __getitem__(self, idx):
        return self._episodes[idx]

    def __len__(self):
        return len(self._episodes)


def make_dataset(cfg, buffer=None):
    cls = defaultdict(lambda: DMControlDataset, {'rlb': RLBenchDataset})[cfg.domain]
    return cls(cfg, buffer)


def _test(tasks, partitions='*', fraction=.5):
    dataset = OfflineDataset('/home/nh/code/dmcontrol-data/data', tasks=tasks, partitions=partitions, fraction=fraction)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    _tasks = '_'.join(tasks).replace('*', 'all')
    for i, imgs in enumerate(dataloader):
        torchvision.utils.save_image(imgs, f'samples/imgs_{_tasks}.png', nrow=8, padding=2)
        break


if __name__ == '__main__':
    if not os.path.exists('samples'):
        os.makedirs('samples')
    _test('*', fraction=.1)
    _test(['humanoid-stand', 'humanoid-walk', 'humanoid-run'])
    _test(['walker-stand', 'walker-walk', 'walker-run'])
