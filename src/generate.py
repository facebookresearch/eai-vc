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
from cfg import parse_hydra
from env import make_env
from algorithm.tdmpc import TDMPC
from logger import make_dir
from train import set_seed
import hydra
import wandb
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'
__ENTITY__, __PROJECT__ = 'nicklashansen', 'tdmpc'


def evaluate(env, agent, num_episodes, step):
	"""Evaluate a trained agent."""
	episode_rewards = []
	episodes = []
	for i in range(num_episodes):
		state, done, ep_reward, t = env.reset(), False, 0, 0
		states, actions, rewards, infos = [state], [], [], []
		while not done:
			action = agent.plan(state, eval_mode=False, step=step, t0=t==0)
			state, reward, done, info = env.step(action.cpu().numpy())
			ep_reward += reward
			states.append(state)
			actions.append(action.cpu().numpy())
			rewards.append(reward)
			infos.append(info)
			t += 1
		episode_rewards.append(ep_reward)
		assert len(env.frames) == 501, f'{len(env.frames)} != 501'
		frames = np.stack(env.frames, axis=0)
		episodes.append({
			'frames': frames,
			'states': states,
			'actions': actions,
			'rewards': rewards, 
			'infos': infos})
	return np.nanmean(episode_rewards), episodes


@hydra.main(config_path="../cfgs", config_name="default")
def generate(cfg):
	"""Script for generating data using pretrained TD-MPC."""
	assert torch.cuda.is_available()
	cfg = parse_hydra(cfg)
	cfg.demo = True
	set_seed(cfg.seed)
	env, agent = make_env(cfg), TDMPC(cfg)

	# Load from wandb
	run_name = 'demo' + str(np.random.randint(0, int(1e6)))
	run = wandb.init(job_type='demo', entity=__ENTITY__, project=__PROJECT__, name=run_name, tags='demo')
	
	for identifier in range(0, int(cfg.train_steps*cfg.action_repeat)+1, cfg.eval_freq):
		# try:
			# Load model
		artifact_dir = None
		for version in range(0, 2):
			name = f'{__ENTITY__}/{__PROJECT__}/{cfg.task}-state-v1-{cfg.seed}-{identifier}:v{version}'
			try:
				artifact = run.use_artifact(name, type='model')
				artifact_dir = artifact.download()
				break
			except Exception as e:
				print(e)
		run.join()
		if artifact_dir is None and identifier > 0:
			raise Exception('No model found')
		elif artifact_dir is None and identifier == 0:
			print('No model found for identifier 0, using random initialization')
		else:
			agent.load(os.path.join(artifact_dir, f'{identifier}.pt'))

		# Evaluate
		reward, episodes = evaluate(env, agent, cfg.eval_episodes, int(1e6))
		print(f'Name: {name}, Reward:', reward)

		# Save transitions to disk
		data_dir = make_dir(Path().cwd() / 'data' / cfg.task / str(identifier))
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
			torch.save(data, data_dir / f'{cfg.seed:03d}_{episode:03d}.pt')
		# except Exception as e:
		# 	print('Failed to run {}'.format(name))
		# 	print(e)

if __name__ == '__main__':
	generate()
