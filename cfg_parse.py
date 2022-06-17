import os
import re
from omegaconf import OmegaConf


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
	"""Parses a Hydra config file."""

	# Logic
	for k in cfg.keys():
		try:
			v = cfg[k]
			if v == None:
				v = True
		except:
			pass

	# Algebraic expressions
	for k in cfg.keys():
		try:
			v = cfg[k]
			if isinstance(v, str):
				match = re.match(r'(\d+)([+\-*/])(\d+)', v)
				if match:
					cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
					if isinstance(cfg[k], float) and cfg[k].is_integer():
						cfg[k] = int(cfg[k])
		except:
			pass

	# Convenience
	cfg.task_title = cfg.task.replace('-', ' ').title()
	cfg.device = 'cpu' if cfg.modality == 'pixels' else 'cuda'
	cfg.exp_name = str(cfg.get('exp_name', 'default'))

	# Multi-task
	cfg.multitask = bool(re.search(r'(mt\d+)', cfg.task))
	if cfg.multitask: # equal number (and >=2) of episodes per task
		cfg.num_tasks = int(re.search(r'(mt\d+)', cfg.task).group(1)[2:])
		cfg.eval_episodes = max(cfg.eval_episodes, cfg.num_tasks * 2)
		cfg.eval_episodes = cfg.eval_episodes - (cfg.eval_episodes % cfg.num_tasks)
	else:
		cfg.num_tasks = 1

	return cfg
