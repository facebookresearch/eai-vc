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
from PIL import Image
from pathlib import Path
from cfg_parse import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from logger import make_dir
from train import set_seed
import hydra
import wandb
torch.backends.cudnn.benchmark = True


def get_state(env):
	try:
		return env.env.env.env._env._env._env._env.physics.get_state()
	except:
		return np.zeros(6, dtype=np.float32)


def evaluate(env, agent, num_episodes, step):
	"""Evaluate a trained agent."""
	episode_rewards = []
	episodes = []
	for i in range(num_episodes):
		state, done, ep_reward, t = env.reset(), False, 0, 0
		states, actions, rewards, infos, phys_states = [state], [], [], [], [get_state(env)]
		while not done:
			action = agent.plan(state, eval_mode=False, step=step, t0=t==0)
			state, reward, done, info = env.step(action.cpu().numpy())
			ep_reward += reward
			states.append(state)
			actions.append(action.cpu().numpy())
			rewards.append(reward)
			infos.append(info)
			phys_states.append(get_state(env))
			t += 1
		episode_rewards.append(ep_reward)
		frames = np.stack(env.frames, axis=0)
		episodes.append({
			'frames': frames,
			'states': states,
			'actions': actions,
			'rewards': rewards, 
			'infos': infos,
			'phys_states': phys_states})
	return np.nanmean(episode_rewards), episodes


@hydra.main(config_name='default', config_path='config')
def generate(cfg: dict):
	"""Script for generating data using pretrained TD-MPC."""
	assert torch.cuda.is_available()	
	print(f'Configuration:\n{cfg}')
	cfg = parse_cfg(cfg)
	cfg.demo = True
	cfg.exp_name = 'v1'
	cfg.eval_freq = 50_000
	cfg.eval_episodes = 100 if cfg.task.startswith('mw-') else 50
	set_seed(cfg.seed)
	env, agent = make_env(cfg), TDMPC(cfg)

	# Load from wandb
	run_name = 'demo' + str(np.random.randint(0, int(1e6)))
	run = wandb.init(job_type='demo', entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_name, tags='demo')
	
	for identifier in range(0, int(cfg.train_steps*cfg.action_repeat)+1, cfg.eval_freq):
		artifact_dir = None
		for version in range(0, 2):
			name = f'{cfg.wandb_entity}/{cfg.wandb_project}/{cfg.task}-state-{cfg.exp_name}-{cfg.seed}-{identifier}:v{version}'
			try:
				artifact = run.use_artifact(name, type='model')
				artifact_dir = Path(artifact.download())
				break
			except Exception as e:
				print(e)
		run.join()
		if artifact_dir is None:
			print(f'Warning: no artifact found for {identifier}, using random initialization')
		else:
			agent.load(artifact_dir / f'{identifier}.pt')

		# Evaluate
		print(f'Evaluating model at step {identifier}')
		reward, episodes = evaluate(env, agent, cfg.eval_episodes, identifier)
		print(f'Name: {name}, Reward:', reward)

		# Save transitions to disk
		data_dir = make_dir(Path(cfg.data_dir) / 'dmcontrol' / cfg.task / str(identifier))
		frames_dir = make_dir(data_dir / 'frames')
		for episode in range(cfg.eval_episodes):
			data = episodes[episode]
			frames, fps = data['frames'], []
			for i in range(len(frames)):
				fp = f'{cfg.seed:03d}_{episode:03d}_{i:03d}.png'
				Image.fromarray(frames[i]).save(frames_dir / fp)
				fps.append(fp)
			data['frames'] = fps
			data.update({'metadata': {
				'cfg': cfg,
				'name': name,
				'run_name': run_name,
				'episode': episode,
				'reward': reward,
			}})
			# _data = torch.load(data_dir / f'{cfg.seed:03d}_{episode:03d}.pt') # for comparison to previous version
			torch.save(data, data_dir / f'{cfg.seed:03d}_{episode:03d}.pt')

if __name__ == '__main__':
	generate()
