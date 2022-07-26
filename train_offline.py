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
from algorithm.helper import make_buffer
from dataloader import make_dataset
from termcolor import colored
import logger
import hydra
torch.backends.cudnn.benchmark = True
__LOGS__ = 'logs'


def evaluate(env, agent, cfg, iteration, video):
	"""Evaluate a trained agent."""
	episode_rewards = []
	for i in range(cfg.eval_episodes):
		if cfg.get('multitask', False):
			env.task_id = i % len(env.tasks)
			enable_video = i < len(env.tasks)
		else:
			enable_video = i == 0
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		if video: video.init(env, enabled=enable_video)
		while not done:
			action = agent.plan(obs, env.task_vec if cfg.get('multitask', False) else None, eval_mode=True, step=int(1e6), t0=t==0)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			ep_reward += reward
			if video: video.record(env)
			t += 1
		episode_rewards.append(ep_reward)
		if video: video.save(iteration, f'videos/{env.task}') if cfg.get('multitask', False) else video.save(iteration)
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
	work_dir = Path(cfg.logging_dir) / __LOGS__ / cfg.task / (cfg.get('features', cfg.modality)) / cfg.algorithm / cfg.exp_name / str(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), work_dir)
	env, agent, buffer = make_env(cfg), make_agent(cfg), make_buffer(cfg)
	print(agent.model)

	# Load dataset
	dataset = make_dataset(cfg, buffer)
	try:
		print(f'Buffer contains {buffer.capacity if buffer.full else buffer.idx} transitions, capacity is {buffer.capacity-1}')
		print(f'\n{colored("Dataset statistics:", "yellow")}\n{dataset.summary}\n')
	except:
		print('Using lazy replay buffer')
	print(colored(f'Training: {work_dir}', 'blue', attrs=['bold']))

	# Resume training (if applicable)
	L = logger.Logger(work_dir, cfg)
	try:
		assert cfg.resume and os.path.exists(L.model_dir / 'chkpt.pt')
		common_metrics = agent.load(L.model_dir / 'chkpt.pt')
		print(colored('Resuming from checkpoint', 'blue', attrs=['bold']))
	except:
		common_metrics = {'iteration': 0, 'start_time': time.time(), 't': time.time()}

	# Run training
	for iteration in range(common_metrics['iteration'], cfg.train_iter+1):

		# Update model
		train_metrics = agent.update(buffer, int(1e6))

		# if cfg.get('lazy_load', False):
		# 	# time the update
		# 	t0 = time.time()
		# 	for _ in range(100):
		# 		agent.update(buffer, int(1e6))
		# 	t1 = time.time()
		# 	print(f'Update time: {t1-t0}')
		# 	exit(0)
			### Timings (state, per 100) ###
			# Regular buffer: 4s
			# 8 workers: 43s
			# 16 workers: 25s
			# 32 workers: 15s
			# 8 workers, amortized: 12s
			# 16 workers, amortized: 11s
			# 32 workers, amortized: 5s
			# 32 workers, full mem: 3s

			### Timings (features, per 100) ###
			# Regular buffer: 8s
			# 8 workers: 81s
			# 16 workers: 47s
			# 32 workers: 29s
			# 8 workers, amortized: 23s
			# 16 workers, amortized: 9s
			# 32 workers, amortized: 9s
			# 32 workers, full mem: 4s

		if iteration % cfg.eval_freq == 0: # and not cfg.get('lazy_load', False):

			# Evaluate agent
			mean_reward, rewards = evaluate(env, agent, cfg, iteration, L.video)

			# Log results
			t = time.time()
			common_metrics = {
				'iteration': iteration,
				'start_time': common_metrics['start_time'],
				'total_time': t - common_metrics['start_time'],
				'duration': t - common_metrics['t'],
				't': t,
				'reward': mean_reward,
			}
			common_metrics.update(train_metrics)
			if cfg.get('multitask', False):
				task_idxs = np.array([i % cfg.num_tasks for i in range(cfg.eval_episodes)])
				task_rewards = np.empty((cfg.num_tasks, cfg.eval_episodes//cfg.num_tasks))
				for i in range(cfg.num_tasks):
					task_rewards[i] = rewards[task_idxs==i]
				task_rewards = task_rewards.mean(axis=1)
				common_metrics.update({f'task_reward/{task}': task_rewards[i] for i, task in enumerate(env.tasks)})
			else:
				common_metrics.update({f'task_reward/{cfg.task}': mean_reward})
			L.log(common_metrics, category='offline')
			if iteration % cfg.save_freq == 0 and iteration > 0:
				L.save_model(agent, 'chkpt', common_metrics)
			t = time.time()
	
	L.finish()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train_offline()
