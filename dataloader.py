import os
import glob
import numpy as np
import torch
from pathlib import Path
from collections import deque
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


class DMControlDataset(Dataset):
    def __init__(self, cfg, data_dir, tasks='*', fraction=1., transform=None, buffer=None):
        self._cfg = cfg
        self._data_dir = data_dir
        self._tasks = tasks if tasks != '*' else sorted(os.listdir(data_dir))
        self._transform = transform
        self._fps = sorted(glob.glob(str(data_dir / '*/*/*.pt')))
        print('Found {} episodes before filtering'.format(len(self._fps)))
        if tasks != '*':
            self._fps = [fp for fp in self._fps if np.any([f'/{t}/' in fp for t in tasks])]
        print('Found {} episodes after filtering'.format(len(self._fps)))
        idxs = np.random.choice(len(self._fps), int(fraction*len(self._fps)), replace=False)
        self._fps = [self._fps[i] for i in idxs]
        if buffer is not None:
            self._buffer = buffer
        else:
            self._episodes = []
        self._cumulative_rewards = []

        dump_filelist = cfg.get('dump_filelist', None)
        if dump_filelist:
            filelist = []

        for fp in tqdm(self._fps):
            data = torch.load(fp)
            if cfg.modality == 'features':
                assert cfg.get('features', None) is not None, 'Features must be specified'
                features_dir = Path(os.path.dirname(fp)) / 'features' / cfg.features
                assert features_dir.exists(), 'No features directory found for {}'.format(fp)
                obs = torch.load(features_dir / os.path.basename(fp))
                if not cfg.features in {'mocodmcontrol5m', 'mocodmcontrolmini'}:
                    _obs = np.empty((obs.shape[0], cfg.frame_stack*obs.shape[1]), dtype=np.float32)
                    obs = stack_frames(obs, _obs, cfg.frame_stack)
                data['metadata']['states'] = data['states']
                data['metadata']['phys_states'] = data['phys_states']
            elif cfg.modality == 'pixels':
                frames_dir = Path(os.path.dirname(fp)) / 'frames'
                assert frames_dir.exists(), 'No frames directory found for {}'.format(fp)
                frame_fps = [frames_dir / fn for fn in data['frames']]
                if dump_filelist:
                    obs = np.empty((len(frame_fps), 3*cfg.frame_stack, 84, 84), dtype=np.float32)
                    filelist.extend(frame_fps)
                else:
                    obs = np.stack([np.array(Image.open(fp)) for fp in frame_fps]).transpose(0, 3, 1, 2)
                    data['metadata']['states'] = data['states']
                    data['metadata']['phys_states'] = data['phys_states']
            else:
                obs = data['states']
            actions = np.array([v['expert_action'] for v in data['infos']] if cfg.get('expert_actions', False) else data['actions'], dtype=np.float32).clip(-1, 1)
            episode = Episode.from_trajectory(cfg, obs, actions, data['rewards'])
            episode.info = data['infos']
            episode.metadata = data['metadata']
            episode.task_vec = torch.tensor([float(data['metadata']['cfg']['task'] == tasks[i]) for i in range(len(tasks))], dtype=torch.float32, device=episode.device)
            episode.task_id = episode.task_vec.argmax()
            episode.filepath = fp
            if buffer is not None:
                self._buffer += episode
            else:
                self._episodes.append(episode)
            self._cumulative_rewards.append(episode.cumulative_reward)

        self._cumulative_rewards = torch.tensor(self._cumulative_rewards, dtype=torch.float32, device=torch.device('cpu'))
        
        if dump_filelist:
            # randomly drop some frames to reduce size
            keep_num = 5_000_000
            idxs = np.random.choice(len(filelist), keep_num, replace=False)
            filelist = [filelist[i] for i in idxs]
            with open(dump_filelist, 'w') as f:
                for fp in filelist:
                    f.write(str(fp) + '\n')
            print('Dumped {} frames to {}'.format(len(filelist), dump_filelist))
            exit(0)

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


def _test(tasks, partitions='*', fraction=.5):
    dataset = DMControlDataset('/home/nh/code/dmcontrol-data/data', tasks=tasks, partitions=partitions, fraction=fraction)
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
