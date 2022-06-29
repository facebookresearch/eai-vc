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
from algorithm.helper import ReplayBuffer
from dataloader import DMControlDataset
from termcolor import colored
import logger
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
	algorithm2class = {'bc': BC, 'tdmpc': TDMPC}
	return algorithm2class[cfg.algorithm](cfg)


@hydra.main(config_name='default', config_path='config')
def train_offline(cfg: dict):
	"""Training script for offline TD-MPC/BC."""
	assert torch.cuda.is_available()
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	work_dir = Path().cwd() / __LOGS__ / cfg.task / (cfg.get('features', cfg.modality)) / cfg.algorithm / cfg.exp_name / str(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), work_dir)
	env, agent, buffer = make_env(cfg), make_agent(cfg), ReplayBuffer(cfg)
	print(agent.model)

	# Load dataset
	tasks = env.unwrapped.tasks if cfg.get('multitask', False) else [cfg.task]
	rbounds = [0, 900 if cfg.task == 'walker-walk' else 1000]
	dataset = DMControlDataset(cfg, Path(cfg.data_dir) / 'dmcontrol', tasks=tasks, fraction=cfg.fraction, rbounds=rbounds, buffer=buffer)
	print(f'Buffer contains {buffer.capacity if buffer.full else buffer.idx} transitions, capacity is {buffer.capacity}')
	dataset_summary = dataset.summary
	print(f'\n{colored("Dataset statistics:", "yellow")}\n{dataset_summary}')

	# Run training
	print(colored(f'Training: {work_dir}', 'blue', attrs=['bold']))
	L = logger.Logger(work_dir, cfg)
	train_metrics, start_time, t = {}, time.time(), time.time()
	for iteration in range(cfg.train_iter+1):

		# Update model
		train_metrics = agent.update(buffer, int(1e6))

		if iteration % cfg.eval_freq == 0:

			# Evaluate agent
			mean_reward, rewards = evaluate(env, agent, cfg)

			# Log results
			common_metrics = {
				'iteration': iteration,
				'total_time': time.time() - start_time,
				'duration': time.time() - t,
				'reward': mean_reward,
			}
			common_metrics.update(train_metrics)
			if cfg.get('multitask', False):
				task_idxs = np.array([i % len(tasks) for i in range(cfg.eval_episodes)])
				task_rewards = np.empty((len(tasks), cfg.eval_episodes//len(tasks)))
				for i in range(len(tasks)):
					task_rewards[i] = rewards[task_idxs==i]
				task_rewards = task_rewards.mean(axis=1)
				common_metrics.update({f'task_reward/{task}': task_rewards[i] for i, task in enumerate(tasks)})
			else:
				common_metrics.update({f'task_reward/{cfg.task}': mean_reward})
			L.log(common_metrics, category='offline')
			t = time.time()
	
	L.finish()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train_offline()
