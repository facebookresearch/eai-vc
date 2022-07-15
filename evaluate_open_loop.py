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
from algorithm.renderer import Renderer
import algorithm.helper as h
from algorithm.helper import RendererBuffer
from dataloader import make_dataset
from encode_dataset import stack_frames
from termcolor import colored
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
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
	


@hydra.main(config_name='default', config_path='config')
def open_loop(cfg: dict):
	"""Rendering script for evaluating learned representations."""
	assert torch.cuda.is_available()
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	save_dir = make_dir(Path(cfg.logging_dir) / 'renderer' / cfg.task / (cfg.features if cfg.modality=='features' else cfg.modality) / cfg.target_modality / cfg.exp_name / str(cfg.seed))
	if not os.path.exists(save_dir / 'model_10000.pt'):
		print('Failed to find renderer model. Please train the renderer first.')
		return
	env, renderer, buffer, val_buffer = make_env(cfg), Renderer(cfg), RendererBuffer(cfg), RendererBuffer(cfg)

	# Load agent from wandb
	run_name = 'renderer' + str(np.random.randint(0, int(1e6)))
	run = wandb.init(job_type='renderer', entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_name, tags='renderer')
	agent = TDMPC(cfg)
	tdmpc_artifact = f'{cfg.task}-{cfg.modality}-{cfg.exp_name}-{cfg.seed}-chkpt:v1'
	print(f'Loading TDMPC artifact {tdmpc_artifact}')
	artifact = run.use_artifact(tdmpc_artifact, type='model')
	artifact_dir = Path(artifact.download())
	agent.load(artifact_dir / os.listdir(artifact_dir)[0])
	renderer.set_tdmpc_agent(agent)
	renderer.load(save_dir / 'model_10000.pt')
	print(renderer.decoder)

	# Evaluate
	print(colored('Evaluating open loop...', 'yellow'))
	episode_rewards = []
	for i in range(cfg.eval_episodes):
		obs, done, ep_reward = env.reset(), False, 0
		actions = agent.plan(obs, None, eval_mode=True, step=int(1e6), t0=True, open_loop=True)
		pred_frames = renderer.imagine(torch.from_numpy(obs), actions).permute(0,2,3,1)
		gt_frames = []
		for action in actions.cpu().numpy():
			gt_frames.append(env.render(height=64, width=64))
			obs, reward, done, _ = env.step(action)
			ep_reward += reward
			if done:
				break
		print('Reward:', ep_reward)
		gt_frames = np.array(gt_frames)
		imageio.mimsave(Path(cfg.logging_dir) / f'openloop_{i}.gif', torch.cat(((pred_frames*255).byte(), torch.from_numpy(gt_frames)), dim=2))
		episode_rewards.append(ep_reward)
	episode_rewards = np.array(episode_rewards)
	print('Mean reward:', episode_rewards.mean())


	def eval_rollout(buffer, num_episodes, fp=None, num_images=num_images, rollout_length=rollout_length):
		start_idx = np.random.randint(cfg.episode_length//4-rollout_length-1) + np.random.randint(num_episodes) * cfg.episode_length
		idxs = np.arange(start_idx, start_idx+rollout_length+1)
		input = buffer.__dict__['_'+cfg.modality][idxs]
		target = buffer.__dict__['_'+cfg.target_modality][idxs]
		action = buffer._action[idxs]
		if cfg.modality == 'pixels':
			_input = torch.empty((input.shape[0], input.shape[1]*cfg.frame_stack, *input.shape[2:]), dtype=torch.float32)
			input = stack_frames(input, _input, cfg.frame_stack).cuda()
		pred = renderer.imagine(input[0], action)
		target = renderer.preprocess_target(target).cpu()
		mse_rollout = 0
		for t in range(rollout_length):
			mse_rollout += (0.95**t) * h.mse(pred[t], target[t], reduce=True)
		mse_rollout = mse_rollout.item()
		if fp is not None and cfg.target_modality == 'pixels':
			image_idxs = np.arange(0, rollout_length+1, rollout_length//(num_images-1))
			save_image(make_grid(torch.cat([pred[image_idxs], target[image_idxs]], dim=0), nrow=num_images), fp)
			imageio.mimsave(str(fp).replace('.png', '.mp4'), torch.cat([pred, target], dim=-1).mul(255).byte().numpy().transpose(0, 2, 3, 1), fps=6)
		return mse_rollout


if __name__ == '__main__':
	open_loop()
