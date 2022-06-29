import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
from pathlib import Path
from cfg_parse import parse_cfg
from env import make_env, set_seed
from algorithm.tdmpc import TDMPC
from algorithm.bc import BC
from algorithm.renderer import Renderer
from algorithm.helper import ReplayBuffer
from dataloader import DMControlDataset
from termcolor import colored
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import logger
import imageio
from logger import make_dir
import pandas as pd
import hydra
torch.backends.cudnn.benchmark = True
__LOGS__ = 'logs'


def evaluate(env, agent, cfg):
	"""Evaluate a trained agent."""
	episode_rewards = []
	for i in range(cfg.eval_episodes):
		if cfg.get('multitask', False):
			env.unwrapped.task_id = i % len(env.unwrapped.tasks)
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		while not done:
			action = agent.plan(obs, env.unwrapped.task_vec if cfg.get('multitask', False) else None, eval_mode=True, step=int(1e6), t0=t==0)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			ep_reward += reward
			t += 1
		episode_rewards.append(ep_reward)
	episode_rewards = np.array(episode_rewards)
	return np.nanmean(episode_rewards), episode_rewards


def make_agent(cfg):
	algorithm2class = {'bc': BC, 'tdmpc': TDMPC, 'renderer': Renderer}
	return algorithm2class[cfg.algorithm](cfg)


@hydra.main(config_name='default', config_path='config')
def render(cfg: dict):
	"""Rendering script for evaluating learned representations."""
	assert torch.cuda.is_available()
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	work_dir = Path().cwd() / __LOGS__ / cfg.task / (cfg.get('features', cfg.modality)) / cfg.algorithm / cfg.exp_name / str(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), work_dir)
	env, agent, buffer = make_env(cfg), make_agent(cfg), ReplayBuffer(cfg)
	print(agent.latent2state)

	# Load dataset
	tasks = env.unwrapped.tasks if cfg.get('multitask', False) else [cfg.task]
	rbounds = [0, 900 if cfg.task == 'walker-walk' else 1000]
	dataset = DMControlDataset(cfg, Path(cfg.data_dir) / 'dmcontrol', tasks=tasks, fraction=cfg.fraction, rbounds=rbounds, buffer=buffer)
	dataset_eval = DMControlDataset(cfg, Path(cfg.data_dir) / 'dmcontrol', tasks=tasks, fraction=cfg.fraction, rbounds=[rbounds[1], 1000])
	print(f'Buffer contains {buffer.capacity if buffer.full else buffer.idx} transitions, capacity is {buffer.capacity}')
	dataset_summary = dataset.summary
	print(f'\n{colored("Dataset statistics:", "yellow")}\n{dataset_summary}')
	print(f'\n{colored("Evaluation statistics:", "yellow")}\n{dataset_eval.summary}')
	save_dir = make_dir(f'/private/home/nihansen/code/tdmpc2/reconstruction/{cfg.task}/{cfg.get("features")}/{cfg.modality}/latent2state')

	if cfg.get('load_agent', False):
		print('Loading from', save_dir)
		agent.load(f'{save_dir}/model.pt')
	else:
		# Run training (latent2state)
		num_images = 8
		print('Saving to', save_dir)
		metrics = []
		for i in tqdm(range(500_000+1)):
			common_metrics = agent.update(buffer)
			if i % 50000 == 0:

				# Evaluate (training set)
				idx = np.random.randint(0, len(dataset.cumrew))
				idxs = np.arange(idx*500, (idx+1)*500)
				obs_pred, state_pred = agent.render(buffer._obs[idxs])
				obs_target, state_target = agent.render(buffer._state[idxs], from_state=True)
				train_mse = torch.mean((state_pred - state_target)**2).item()
				image_idxs = np.random.randint(0, 500, size=num_images)
				save_image(make_grid(torch.cat([obs_pred[image_idxs], obs_target[image_idxs]], dim=0), nrow=8), f'{save_dir}/train_{i}.png')
				imageio.mimsave(f'{save_dir}/train_{i}.mp4', (torch.cat([obs_pred, obs_target], dim=-1).permute(0,2,3,1)*255).byte().cpu().numpy(), fps=12)
				print(colored('Training MSE:', 'yellow'), train_mse)

				# Evaluate (evaluation set)
				episode = dataset_eval.episodes[dataset_eval.cumrew.argmax()]
				obs_pred, state_pred = agent.render(episode.obs[:500])
				obs_target, state_target = agent.render(torch.FloatTensor(episode.metadata['states'][:500]), from_state=True)
				eval_mse = torch.mean((state_pred - state_target)**2).item()
				image_idxs = np.random.randint(0, 500, size=num_images)
				save_image(make_grid(torch.cat([obs_pred[image_idxs], obs_target[image_idxs]], dim=0), nrow=8), f'{save_dir}/eval_{i}.png')
				imageio.mimsave(f'{save_dir}/eval_{i}.mp4', (torch.cat([obs_pred, obs_target], dim=-1).permute(0,2,3,1)*255).byte().cpu().numpy(), fps=12)
				print(colored('Eval MSE:', 'yellow'), eval_mse)

				# Logging
				metrics.append(np.array([i, train_mse, eval_mse, common_metrics['total_loss'], common_metrics['grad_norm']]))
				pd.DataFrame(np.array(metrics)).to_csv(f'{save_dir}/metrics.csv', header=['iteration', 'train_mse', 'eval_mse', 'batch_mse', 'grad_norm'], index=None)
		
		agent.save(f'{save_dir}/model.pt')


if __name__ == '__main__':
	render()
