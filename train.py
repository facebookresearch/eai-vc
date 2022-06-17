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
from multiprocessing import Process
from pathlib import Path
from cfg_parse import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import logger
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__, __MODELS__ = 'cfgs', 'logs', 'models'


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def parallel(fn, cfg, wait=5, verbose=False):
	assert cfg.seed is not None, 'No seed(s) given'
	seeds = cfg.seed
	if isinstance(seeds, int):
		return fn(cfg)
	seeds = [int(seed) for seed in seeds.split(',')]
	proc = []
	for seed in seeds:
		_cfg = deepcopy(cfg)
		_cfg.seed = seed
		p = Process(target=fn, args=(_cfg,))
		p.start()
		proc.append(p)
		if verbose:
			print(f'Started process {p.pid} with seed {seed}')
		time.sleep(wait)
	for p in proc:
		p.join()
	while len(proc) > 0:
		time.sleep(wait)
		for p in proc:
			if not p.is_alive():
				if verbose:
					print(f'Process {p.pid} has finished')
				p.terminate()
				proc.remove(p)
	exit(0)


def evaluate(env, agent, num_episodes, step, env_step, video):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	episode_successes = []
	for i in range(num_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		if video: video.init(env, enabled=(i==0))
		while not done:
			action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
			obs, reward, done, info = env.step(action.cpu().numpy())
			ep_reward += reward
			if video: video.record(env)
			t += 1
		episode_rewards.append(ep_reward)
		episode_successes.append(info.get('success', 0))
		if video: video.save(env_step)
	return np.nanmean(episode_rewards), np.nanmean(episode_successes)


def train(cfg):
	"""Training script for TD-MPC."""
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	work_dir = Path().cwd() / __MODELS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)

	# Run training
	L = logger.Logger(work_dir, cfg)
	episode_idx, start_time = 0, time.time()
	for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):

		# Collect trajectory
		obs = env.reset()
		episode = Episode(cfg, obs)
		while not episode.done:
			action = agent.plan(obs, step=step, t0=episode.first)
			obs, reward, done, info = env.step(action.cpu().numpy())
			episode += (obs, action, reward, done)
		if len(episode) < cfg.episode_length:
			print(len(episode), done, episode.reward[:len(episode)].sum().cpu().item())
		assert len(episode) == cfg.episode_length
		buffer += episode

		# Update model
		train_metrics = {}
		if step >= cfg.seed_steps:
			num_updates = (cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length) // cfg.steps_per_update
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+int(i*cfg.steps_per_update)))

		# Log training episode
		episode_idx += 1
		env_step = int(step*cfg.action_repeat)
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward,
			'episode_success': info.get('success', 0)}
		train_metrics.update(common_metrics)
		L.log(train_metrics, category='train')

		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:
			eval_rew, eval_succ = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
			common_metrics.update({
				'episode_reward': eval_rew,
				'episode_success': eval_succ})
			L.log(common_metrics, category='eval')
			if cfg.save_model:
				L.save_model(agent, env_step)

	L.finish(agent)
	print('\nTraining completed successfully')


if __name__ == '__main__':
	parallel(train, parse_cfg(Path().cwd() / __CONFIG__), verbose=True)
