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
import random
from copy import deepcopy
import multiprocessing
from multiprocessing import set_start_method
from pathlib import Path
from cfg_parse import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.bc import BC
from algorithm.helper import ReplayBuffer
from dataloader import DMControlDataset, summary_stats
from termcolor import colored
from logger import make_dir
import hydra
import wandb
torch.backends.cudnn.benchmark = True
__LOGS__, __DATA__ = 'logs', 'data'


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, cfg):
	"""Evaluate a trained agent."""
	episode_rewards = []
	for i in range(cfg.eval_episodes):
		if cfg.get('multitask', False):
			env.unwrapped.task_id = i % len(env.unwrapped.tasks)
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		while not done:
			action = agent.plan(obs, env.unwrapped.task_vec, eval_mode=True, step=int(1e6), t0=t==0)
			obs, reward, done, info = env.step(action.cpu().numpy())
			ep_reward += reward
			t += 1
		episode_rewards.append(ep_reward)
	return np.nanmean(episode_rewards), episode_rewards


def make_agent(cfg):
	algorithm2class = {'bc': BC, 'tdmpc': TDMPC}
	return algorithm2class[cfg.algorithm](cfg)


@hydra.main(config_name='default', config_path='config')
def train(cfg: dict):
	"""Training script for offline TD-MPC/BC."""
	assert torch.cuda.is_available()
	print(f'Configuration:\n{cfg}')
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)

	# Prepare objects
	work_dir = make_dir(Path().cwd() / __LOGS__ / cfg.task / (cfg.get('features', cfg.modality)) / cfg.algorithm / cfg.exp_name / str(cfg.seed))
	print(colored('Work dir:', 'yellow', attrs=['bold']), work_dir)
	env, agent, buffer = make_env(cfg), make_agent(cfg), ReplayBuffer(cfg)
	print(agent.model)

	# Prepare buffer
	tasks = env.unwrapped.tasks if cfg.get('multitask', False) else [cfg.task]
	dataset = DMControlDataset(cfg, Path(cfg.data_dir) / 'dmcontrol', tasks=tasks, fraction=cfg.fraction, buffer=buffer)
	print(f'Buffer contains {buffer.capacity if buffer.full else buffer.idx} transitions, capacity is {buffer.capacity}')
	dataset_summary = dataset.summary
	print(f'\n{colored("Dataset statistics:", "yellow")}\n{dataset_summary}')

	# Run training
	print(colored(f'Training: {work_dir}', 'blue', attrs=['bold']))
	t = time.time()
	best = None
	all_rewards = []
	iterations = 500_000
	for iteration in range(iterations+1):
		if iteration % cfg.eval_freq == 0:
			mean_reward, rewards = evaluate(env, agent, cfg)
			print(f'I: {iteration:<6d}   R: {mean_reward:<6.1f}   D: {(time.time()-t)/60:<6.1f}')
			last = {'iteration': iteration,
					'mean_reward': mean_reward,
					'rewards': rewards,
					'time': time.time()-t,
					'cfg': cfg.__dict__,
					'work_dir': work_dir,
					'num_transitions': buffer.idx,
					'dataset_summary': dataset_summary,
					'agent': agent.state_dict()}
			if best is None or mean_reward > best.get('mean_reward', 0):
				best = last
			all_rewards.append(rewards)
			torch.save({'best': best, 'last': last, 'all_rewards': all_rewards}, work_dir / f'{cfg.fraction}.pt')
			t = time.time()
		agent.update(buffer, int(1e6))
	
	# Finish
	best_summary = summary_stats(torch.tensor(best['rewards'], dtype=torch.float))
	last_summary = summary_stats(torch.tensor(last['rewards'], dtype=torch.float))
	print(f'{colored("Best:", "yellow")}\n{best_summary}\n')
	print(f'{colored("Last:", "yellow")}\n{last_summary}\n')


if __name__ == '__main__':
	train()
