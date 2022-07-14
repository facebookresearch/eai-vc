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
from algorithm.helper import ReplayBuffer
from dataloader import make_dataset
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


@hydra.main(config_name='default', config_path='config')
def render(cfg: dict):
	"""Rendering script for evaluating learned representations."""
	assert torch.cuda.is_available()
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	save_dir = make_dir(Path(cfg.logging_dir) / 'renderer' / cfg.task / (cfg.get('features', cfg.modality)) / cfg.exp_name / str(cfg.seed))
	env, renderer, buffer, val_buffer = make_env(cfg), Renderer(cfg), ReplayBuffer(cfg), ReplayBuffer(cfg)

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
	print(renderer.decoder)

	# Load dataset
	assert cfg.get('use_val', False), 'Validation dataset is required for simulation experiments'
	dataset = make_dataset(cfg, buffer)
	for episode in dataset._val_episodes:
		val_buffer += episode
	print(f'Buffer contains {buffer.capacity if buffer.full else buffer.idx} transitions, capacity is {buffer.capacity-1}')
	print(f'Validation buffer contains {val_buffer.capacity if val_buffer.full else val_buffer.idx} transitions, capacity is {val_buffer.capacity-1}')
	dataset_summary = dataset.summary
	print(f'\n{colored("Dataset statistics:", "yellow")}\n{dataset_summary}\n')

	# Config
	num_images = 9
	rollout_length = 40

	def eval_encode_decode(buffer, fp=None, num_images=num_images):
		"""Evaluate single-step reconstruction error"""
		pixels = buffer.sample()[0]
		pixels_pred = renderer.encode_decode(pixels)
		pixels_target = renderer.preprocess_target(pixels).cpu()
		if fp is not None:
			image_idxs = np.random.randint(len(pixels), size=num_images)
			save_image(make_grid(torch.cat([pixels_pred[image_idxs], pixels_target[image_idxs]], dim=0), nrow=num_images), fp)
		return h.mse(pixels_pred, pixels_target, reduce=True)
	
	def eval_rollout(buffer, num_episodes, fp=None, num_images=num_images, rollout_length=rollout_length):
		start_idx = np.random.randint(cfg.episode_length//4-rollout_length-1) + np.random.randint(num_episodes) * cfg.episode_length
		idxs = np.arange(start_idx, start_idx+rollout_length+1)
		pixels, actions = buffer._obs[idxs], buffer._action[idxs]
		pixels_pred = renderer.imagine(pixels[0], actions)
		pixels_target = renderer.preprocess_target(pixels).cpu()
		mse_rollout = 0
		for t in range(rollout_length):
			mse_rollout += (0.95**t) * h.mse(pixels_pred[t], pixels_target[t], reduce=True)
		mse_rollout = mse_rollout.item()
		if fp is not None:
			image_idxs = np.arange(0, rollout_length+1, rollout_length//(num_images-1))
			save_image(make_grid(torch.cat([pixels_pred[image_idxs], pixels_target[image_idxs]], dim=0), nrow=num_images), fp)
		return mse_rollout

	# Run training
	metrics = []
	print(colored('Saving to dir:', 'yellow'), save_dir)
	for iteration in tqdm(range(cfg.train_iter+1)):

		# Update model
		common_metrics = renderer.update(buffer)

		if iteration % cfg.eval_freq == 0:

			# Evaluate (training set)
			train_mse = eval_encode_decode(buffer, fp=save_dir / f'train_{iteration}.png')
			train_mse_rollout = eval_rollout(buffer, len(dataset._episodes), fp=save_dir / f'train_rollout_{iteration}.png')

			# Evaluate (validation set)
			eval_mse = eval_encode_decode(val_buffer, fp=save_dir / f'val_{iteration}.png')
			eval_mse_rollout = eval_rollout(val_buffer, len(dataset._val_episodes), fp=save_dir / f'val_rollout_{iteration}.png')

			# Logging
			metrics.append(np.array([iteration, train_mse, eval_mse, train_mse_rollout, eval_mse_rollout, *common_metrics.values()]))
			pd.DataFrame(np.array(metrics)).to_csv(f'{save_dir}/metrics.csv', header=['iteration', 'train_mse', 'eval_mse', 'train_mse_rollout', 'eval_mse_rollout', *common_metrics.keys()], index=None)
			print(f'Iteration {iteration}, train mse: {train_mse:.4f}, eval mse: {eval_mse:.4f}, train mse rollout: {train_mse_rollout:.4f}, eval mse rollout: {eval_mse_rollout:.4f}')
			
			if iteration % cfg.save_freq == 0 and iteration > 0:
				renderer.save(f'{save_dir}/model_{iteration}.pt')

	# if cfg.get('renderer_path', None):
	# 	print('Loading renderer from', cfg.renderer_path)
	# 	renderer.load(cfg.renderer_path)
	# else:
	# 	num_images = 8
	# 	print('Saving to', save_dir)
	# 	metrics = []
	# 	for i in tqdm(range(50_000+1)):
	# 		common_metrics = renderer.update(buffer)
	# 		if i % cfg.eval_freq == 0:

	# 			# Evaluate (training set)
	# 			idx = np.random.randint(0, len(dataset.cumrew))
	# 			idxs = np.arange(idx*500, (idx+1)*500)
	# 			obs_pred, state_pred = renderer.render(buffer._obs[idxs])
	# 			obs_target, state_target = renderer.render(buffer._state[idxs], from_state=True)
	# 			train_mse = torch.mean((state_pred - state_target)**2).item()
	# 			image_idxs = np.random.randint(0, 500, size=num_images)
	# 			save_image(make_grid(torch.cat([obs_pred[image_idxs], obs_target[image_idxs]], dim=0), nrow=8), f'{save_dir}/train_{i}.png')
	# 			imageio.mimsave(f'{save_dir}/train_{i}.mp4', (torch.cat([obs_pred, obs_target], dim=-1).permute(0,2,3,1)*255).byte().cpu().numpy(), fps=12)
	# 			print(colored('Training MSE:', 'green'), train_mse)

	# 			# Evaluate (training set; imagined rollout)
	# 			start_idx, length = 100, 20
	# 			obs_pred, state_pred = renderer.imagine(buffer._obs[idxs[start_idx]], buffer._action[idxs[start_idx:start_idx+length]])
	# 			obs_target, state_target = renderer.render(buffer._state[idxs[start_idx:start_idx+length]], from_state=True)
	# 			save_image(make_grid(torch.cat([obs_pred[0,::2], obs_target[::2]], dim=0), nrow=10), f'{save_dir}/train_imagine_{i}.png')
	# 			train_imagine_mse = np.around(torch.mean((state_pred - state_target)**2, dim=-1)[0].numpy(), 2)
	# 			print(colored('Training imagine MSE:', 'green'), train_imagine_mse)

	# 			# Evaluate (evaluation set)
	# 			episode = dataset_eval.episodes[dataset_eval.cumrew.argmax()]
	# 			obs_pred, state_pred = renderer.render(episode.obs[:500])
	# 			obs_target, state_target = renderer.render(torch.FloatTensor(episode.metadata['states'][:500]), from_state=True)
	# 			eval_mse = torch.mean((state_pred - state_target)**2).item()
	# 			image_idxs = np.random.randint(0, 500, size=num_images)
	# 			save_image(make_grid(torch.cat([obs_pred[image_idxs], obs_target[image_idxs]], dim=0), nrow=8), f'{save_dir}/eval_{i}.png')
	# 			imageio.mimsave(f'{save_dir}/eval_{i}.mp4', (torch.cat([obs_pred, obs_target], dim=-1).permute(0,2,3,1)*255).byte().cpu().numpy(), fps=12)
	# 			print(colored('Eval MSE:', 'yellow'), eval_mse)

	# 			# Evaluate (evaluation set; imagined rollout)
	# 			obs_pred, state_pred = renderer.imagine(episode.obs[start_idx], episode.action[start_idx:start_idx+length])
	# 			obs_target, state_target = renderer.render(torch.FloatTensor(episode.metadata['states'][start_idx:start_idx+length]), from_state=True)
	# 			save_image(make_grid(torch.cat([obs_pred[0,::2], obs_target[::2]], dim=0), nrow=10), f'{save_dir}/eval_imagine_{i}.png')
	# 			eval_imagine_mse = np.around(torch.mean((state_pred - state_target)**2, dim=-1)[0].numpy(), 2)
	# 			print(colored('Eval imagine MSE:', 'yellow'), eval_imagine_mse)

	# 			# Logging
	# 			metrics.append(np.array([i, train_mse, eval_mse, train_imagine_mse[-1], eval_imagine_mse[-1], common_metrics['total_loss'], common_metrics['grad_norm']]))
	# 			pd.DataFrame(np.array(metrics)).to_csv(f'{save_dir}/metrics.csv', header=['iteration', 'train_mse', 'eval_mse', 'train_imagine_mse', 'eval_imagine_mse', 'batch_mse', 'grad_norm'], index=None)
	# 			renderer.save(f'{save_dir}/model_{i}.pt')


if __name__ == '__main__':
	render()
