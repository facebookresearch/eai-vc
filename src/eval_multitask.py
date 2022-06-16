import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ["WANDB_SILENT"] = 'true'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
from pathlib import Path
from PIL import Image
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.bc import BC
from logger import make_dir
from train import parallel, set_seed
import imageio
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def make_agent(cfg):
	algorithm2class = {'bc': BC, 'tdmpc': TDMPC}
	return algorithm2class[cfg.algorithm](cfg)


def evaluate(env, agent, step):
	"""Evaluate a trained agent on tasks sequentially."""
	state = env.reset()
	frames = []
	print('Tasks:', [(task, i) for i, task in enumerate(env.unwrapped.tasks)])
	for task_id in range(len(env.unwrapped.tasks)):
		task_vec = np.zeros(len(env.unwrapped.tasks), dtype=np.float32)
		task_vec[-task_id] = 1.
		print(f'Task {task_id}: {env.unwrapped.tasks[task_id]}')
		for t in range(200 if 'cheetah' in env.unwrapped.tasks[0] else 150):
			action = agent.plan(state, task_vec, eval_mode=True, step=step, t0=t==0)
			state, _, _, _ = env.step(action.cpu().numpy())
			frame = env.render(mode='rgb_array', width=384, height=384)
			frames.append(frame)
	frames = np.stack(frames, axis=0)
	imageio.mimsave('frames.mp4', frames, fps=20)


# def evaluate(env, agent, step):
# 	"""Evaluate a trained agent on two tasks sequentially."""
# 	state = env.reset()
# 	frames = []
# 	task_vec = np.zeros(len(env.unwrapped.tasks), dtype=np.float32)
# 	print('Tasks:', [(task, i) for i, task in enumerate(env.unwrapped.tasks)])
# 	print('Select two indices:')
# 	task_id1, task_id2 = map(int, input().split(','))
# 	print('Task 1:', env.unwrapped.tasks[task_id1])
# 	print('Task 2:', env.unwrapped.tasks[task_id2])
# 	task_vec[task_id1] = 1.
# 	for t in range(300):
# 		if t == 150:
# 			task_vec[task_id1] = 0.
# 			task_vec[task_id2] = 1.
# 		action = agent.plan(state, task_vec, eval_mode=True, step=step, t0=t==0)
# 		state, _, _, _ = env.step(action.cpu().numpy())
# 		frame = env.render(mode='rgb_array', width=384, height=384)
# 		frames.append(frame)
# 	frames = np.stack(frames, axis=0)
# 	imageio.mimsave('twotask.mp4', frames, fps=20)


def main(cfg):
	"""Script for evaluating multi-task TD-MPC agents."""
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	env, agent = make_env(cfg), make_agent(cfg)

	# Load agent
	model_dir = Path(__LOGS__) / cfg.task / cfg.modality / cfg.algorithm / cfg.exp_name / str(cfg.seed)
	agent.load(torch.load(model_dir / '1.0.pt')['last']['agent'])

	# Evaluate
	evaluate(env, agent, int(1e6))


if __name__ == '__main__':
	cfg = parse_cfg(Path().cwd() / __CONFIG__)
	cfg.episode_length = 10_000
	cfg.infinite_horizon = True
	main(cfg)
