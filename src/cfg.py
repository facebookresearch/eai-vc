import os
import re
from omegaconf import OmegaConf


def parse_cfg(cfg_path: str) -> OmegaConf:
	"""Parses a config file and returns an OmegaConf object."""
	base = OmegaConf.load(cfg_path / 'default.yaml')
	cli = OmegaConf.from_cli()
	for k,v in cli.items():
		if v == None:
			cli[k] = True
	base.merge_with(cli)

	# Modality config
	if cli.get('modality', base.modality) not in {'state', 'pixels', 'features'}:
		raise ValueError('Invalid modality: {}'.format(cli.get('modality', base.modality)))
	modality = cli.get('modality', base.modality)
	if modality != 'state':
		mode = OmegaConf.load(cfg_path / f'{modality}.yaml')
		base.merge_with(mode, cli)

	# Task config
	try:
		domain, task = base.task.split('-', 1)
	except:
		raise ValueError(f'Invalid task name: {base.task}')
	domain_path = cfg_path / 'tasks' / f'{domain}.yaml'
	if not os.path.exists(domain_path):
		domain_path = cfg_path / 'tasks' / 'default.yaml'
	domain_cfg = OmegaConf.load(domain_path)
	base.merge_with(domain_cfg, cli)

	# Algebraic expressions
	for k,v in base.items():
		if isinstance(v, str):
			match = re.match(r'(\d+)([+\-*/])(\d+)', v)
			if match:
				base[k] = eval(match.group(1) + match.group(2) + match.group(3))
				if isinstance(base[k], float) and base[k].is_integer():
					base[k] = int(base[k])

	# Convenience
	base.task_title = base.task.replace('-', ' ').title()
	base.device = 'cpu' if base.modality == 'pixels' else 'cuda'
	base.exp_name = str(base.get('exp_name', 'default'))

	# Multi-task
	base.multitask = bool(re.search(r'(mt\d+)', base.task))
	if base.multitask: # equal number (and >=2) of episodes per task
		base.num_tasks = int(re.search(r'(mt\d+)', base.task).group(1)[2:])
		base.eval_episodes = max(base.eval_episodes, base.num_tasks * 2)
		base.eval_episodes = base.eval_episodes - (base.eval_episodes % base.num_tasks)
	else:
		base.num_tasks = 1

	return base
