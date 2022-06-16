import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import random
from multiprocessing import Process
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from dataloader import DMControlDataset, summary_stats
from termcolor import colored
from logger import make_dir
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__, __DATA__ = 'cfgs', 'logs', 'data'


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

__DMC3__ = [
	'walker-run',
	'walker-walk',
	'walker-stand',
]
__DMC5__ = [
	'finger-spin',
	'reacher-hard',
	'cheetah-run',
	'walker-stand',
	'walker-walk',
]
__DMC20__ = [
	'acrobot-swingup',
	'cartpole-balance',
	'cartpole-swingup',
	'cartpole-two-poles',
	'cheetah-run',
	'cup-catch',
	'finger-spin',
	'finger-turn-easy',
	'finger-turn-hard',
	'fish-swim',
	'hopper-hop',
	'hopper-stand',
	'pendulum-swingup',
	'quadruped-run',
	'quadruped-walk',
	'reacher-easy',
	'reacher-hard',
	'walker-run',
	'walker-stand',
	'walker-walk',
]
key2tasks = {'dmc3': __DMC3__, 'dmc5': __DMC5__, 'dmc20': __DMC20__}


def summary(cfg):
	set_seed(cfg.seed)
	results_dir = make_dir(Path().cwd() / 'results')
	tasks = key2tasks[cfg.get('tasks', 'dmc20')]
	checkpoint = cfg.get('checkpoint', 'last')
	normalization = cfg.get('normalization', 'expert')

	print('Modality:', cfg.get('features', cfg.modality))
	print('Fraction:', cfg.get('fraction', 1.0))
	print('Tasks:', cfg.get('tasks', 'dmc20'))
	print('Checkpoint:', checkpoint)
	print('Normalization:', normalization)

	# Load dataset statistics
	results_fn = results_dir / 'single_state_10.pt'
	if os.path.exists(results_fn):
		results = torch.load(results_fn)
	else:
		results = {}

	def load_results(exp_name):
		fp = root_dir / exp_name / str(cfg.seed) / f'{cfg.fraction}.pt'
		if os.path.exists(fp):
			summary = summary_stats(torch.tensor(torch.load(fp)[checkpoint]['rewards'], dtype=torch.float))
			print(f'{colored(f"{exp_name}:", "yellow")}\n{summary}')
			results[task][exp_name] = summary
		else:
			results[task][exp_name] = {'mean': np.nan}

	for task in tasks:

		print(colored(f'\nTask: {task}', 'blue', attrs=['bold']))
		root_dir = make_dir(Path().cwd() / __LOGS__ / task / (cfg.get('features', cfg.modality)))
		env = make_env(cfg)
		if task not in results:
			results[task] = {}
		if 'dataset' not in results[task] or 'expert' not in results[task]:
			partitions = ['iterations=0', 'iterations=1', 'iterations=2',
						'iterations=3', 'iterations=4', 'iterations=5',
						'iterations=6', 'variable_std=0.3', 'variable_std=0.5']
			dataset = DMControlDataset(cfg, Path().cwd() / __DATA__, tasks=[task], partitions=partitions, fraction=cfg.fraction)
			results[task]['dataset'], results[task]['expert'] = dataset.summary

		# Display dataset statistics
		print(f'{colored("dataset:", "yellow")}\n{results[task]["dataset"]}')
		print(f'{colored("expert:", "yellow")}\n{results[task]["expert"]}')

		# Load results
		load_results('offline')
		load_results('offline-cat')

		# Load BC results
		load_results('bc')

		# Load BC-expert results
		load_results('bc-expert')
		load_results('bc-expert-again')

		# Load CLIP results
		load_results('bc-expert-clip-cat')
		load_results('bc-expert-bn')
		load_results('bc-expert-flare')
		load_results('bc-expert-again-cat')
		load_results('bc-expert-again-bn')
	
	# Summarize results
	denominator = np.nanmean([results[task]['expert']['mean'] for task in tasks]) if normalization == 'expert' else 10.
	for method in results[task].keys():
		mean = np.nanmean([results[task][method]['mean'] for task in tasks])
		if not np.isnan(mean):
			print(f'{colored(method, "yellow")} {round(float(mean/denominator), 4)}')

	# Save results
	if cfg.modality in {'state', 'pixels'} and cfg.get('tasks', 'dmc20') == 'dmc20':
		torch.save(results, results_dir / f'single_{cfg.modality}_{str(cfg.fraction).replace(".", "")}.pt')


if __name__ == '__main__':
	summary(parse_cfg(Path().cwd() / __CONFIG__))
