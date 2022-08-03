import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
from pathlib import Path
from cfg_parse import parse_cfg
from env import make_env, set_seed
from algorithm.tdmpc import TDMPC
from algorithm.mtdmpc import MultiTDMPC
from algorithm.bc import BC
from termcolor import colored
import imageio
from logger import make_dir
import pandas as pd
import hydra
os.environ["WANDB_SILENT"] = "true"
import wandb
torch.backends.cudnn.benchmark = True
__LOGS__ = 'logs'


def evaluate(env, agent, cfg):
	"""Evaluate a trained agent."""
	episode_rewards = []
	for i in range(cfg.eval_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		while not done:
			action = agent.plan(obs, None, None, eval_mode=True, step=int(1e6), t0=t==0)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			ep_reward += reward
			t += 1
		print(f'Episode {i} reward: {ep_reward}')
		episode_rewards.append(ep_reward)
	episode_rewards = np.array(episode_rewards)
	return np.nanmean(episode_rewards), episode_rewards


def make_agent(cfg):
	algorithm2class = {'bc': BC, 'tdmpc': TDMPC, 'mtdmpc': MultiTDMPC}
	return algorithm2class[cfg.algorithm](cfg)


@hydra.main(config_name='default', config_path='config')
def main(cfg: dict):
	"""Rendering script for evaluating learned representations."""
	assert torch.cuda.is_available()
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	work_dir = Path(cfg.logging_dir) / __LOGS__ / cfg.task / (cfg.get('features', cfg.modality)) / cfg.algorithm / cfg.exp_name / str(cfg.seed)
	env, agent = make_env(cfg), make_agent(cfg)
	print(agent.model)

	# Load agent
	print(colored('Work dir:', 'yellow', attrs=['bold']), work_dir)
	assert os.path.exists(work_dir / 'models' / 'chkpt.pt')
	common_metrics = agent.load(work_dir / 'models' / 'chkpt.pt')
	print('Training reward:', common_metrics['reward'])

	# Evaluate
	print(colored('Evaluating generalization...', 'yellow'))
	mean_reward, _ = evaluate(env, agent, cfg)
	print('Mean reward:', mean_reward)
	


if __name__ == '__main__':
	main()
