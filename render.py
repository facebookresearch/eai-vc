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
	dataset = DMControlDataset(cfg, Path(cfg.data_dir) / 'dmcontrol', tasks=tasks, fraction=cfg.fraction, buffer=buffer)
	print(f'Buffer contains {buffer.capacity if buffer.full else buffer.idx} transitions, capacity is {buffer.capacity}')
	dataset_summary = dataset.summary
	print(f'\n{colored("Dataset statistics:", "yellow")}\n{dataset_summary}')

	# Run training (state2pixels)
	# num_images = 8
	# for i in tqdm(range(50_000)):
	# 	metrics = agent.update(buffer)
	# 	if i % 500 == 0:
	# 		print(colored('Iteration:', 'yellow'), f'{i:6d}', ' '.join([f'{k}: {v:.3f}' for k, v in metrics.items()]))
	# 		obs, _, _, _, state, _, _, _ = buffer.sample()
	# 		obs_pred = agent.render(state[:num_images])
	# 		obs_target = agent._resize(obs[:num_images, -3:]/255.).cpu()
	# 		save_image(make_grid(torch.cat([obs_pred, obs_target], dim=0), nrow=8), f'/private/home/nihansen/code/tdmpc2/recon/recon_{i}.png')
	# 		if i % 1000 == 0:
	# 			agent.save(f'/private/home/nihansen/code/tdmpc2/recon/renderer_{i}.pt')

	save_dir = make_dir(f'/private/home/nihansen/code/tdmpc2/reconstruction/{cfg.task}/{cfg.get("features")}/{cfg.modality}/latent2state')

	if cfg.get('load_agent', False):
		print('Loading from', save_dir)
		agent.load(f'{save_dir}/model.pt')
	else:
		# Run training (latent2state)
		num_images = 8
		print('Saving to', save_dir)
		for i in tqdm(range(50_000+1)):
			metrics = agent.update(buffer)
			if i % 10000 == 0:
				print(colored('Iteration:', 'yellow'), f'{i:6d}', ' '.join([f'{k}: {v:.3f}' for k, v in metrics.items()]))
				latent, _, _, _, state, _, _, _ = buffer.sample()
				obs_pred = agent.render(latent[:num_images])
				obs_target = agent.render(state[:num_images], from_state=True)
				save_image(make_grid(torch.cat([obs_pred, obs_target], dim=0), nrow=8), f'{save_dir}/{i}.png')
		agent.save(f'{save_dir}/model.pt')
	
	# Evaluate
	idx = dataset.cumrew.argmax()
	idxs = np.arange(idx*500, (idx+1)*500)
	latent = buffer._obs[idxs]
	state = buffer._state[idxs]

	obs_pred = agent.render(latent)
	obs_target = agent.render(state, from_state=True)

	frames = torch.cat([obs_pred, obs_target], dim=-1)
	imageio.mimsave(f'{save_dir}/optimal.mp4', (frames.permute(0,2,3,1)*255).byte().cpu().numpy(), fps=12)


if __name__ == '__main__':
	render()
