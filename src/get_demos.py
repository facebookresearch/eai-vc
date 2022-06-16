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
from logger import make_dir
from train import parallel, set_seed
import wandb
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'
__ENTITY__, __PROJECT__ = 'nicklashansen', 'tdmpc'


def evaluate(env, agent, num_episodes, step):
	"""Evaluate a trained agent."""
	episode_rewards = []
	episodes = []
	demo_iterations = agent.cfg.get('demo_iterations', agent.cfg.iterations)
	demo_std_max = agent.cfg.get('demo_variable_std', agent.cfg.min_std) / agent.cfg.min_std
	demo_std_fractions = np.linspace(0, demo_std_max, num_episodes)
	for i in range(num_episodes):
		state, done, ep_reward, t = env.reset(), False, 0, 0
		states, actions, rewards, infos = [state], [], [], []
		while not done:
			expert_action, std, action_list = agent.plan(state, eval_mode=True, step=step, t0=t==0, demo_std_fraction=demo_std_fractions[i])
			if demo_iterations == agent.cfg.iterations and 'demo_variable_std' not in agent.cfg.keys(): # expert action
				action = expert_action
			elif demo_iterations == 0: # random action
				action = torch.empty_like(expert_action).uniform_(-1, 1)
			else: # noisy action
				action = action_list[demo_iterations-1]
			state, reward, done, info = env.step(action.cpu().numpy())
			ep_reward += reward
			states.append(state)
			actions.append(action.cpu().numpy())
			rewards.append(reward)
			info.update({'expert_action': expert_action.cpu().numpy(), 'std': std.cpu().numpy(), 'action_list': action_list.cpu().numpy()})
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


def get_demos(cfg):
	"""Script for generating demos using pretrained TD-MPC."""
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	env, agent = make_env(cfg), TDMPC(cfg)

	# Load from wandb
	run_name = 'demo' + str(np.random.randint(0, int(1e5)))
	run = wandb.init(job_type='demo', entity=__ENTITY__, project=__PROJECT__, name=run_name, tags='demo')
	name = f'{__ENTITY__}/{__PROJECT__}/{cfg.task}-state-demo-{cfg.seed}:v0'
	
	try:
		# Load model
		artifact = run.use_artifact(name, type='model')
		artifact_dir = artifact.download()
		run.join()
		agent.load(os.path.join(artifact_dir, 'model.pt'))

		# Evaluate
		reward, episodes = evaluate(env, agent, cfg.eval_episodes, int(1e6))
		print(f'Name: {name}, Iterations: {cfg.get("demo_iterations", cfg.iterations)}, Reward:', reward)

		# Save transitions to disk
		save_dir = make_dir(Path().cwd() / 'data' / cfg.task / cfg.partition)
		for episode in range(cfg.eval_episodes):
			data = episodes[episode]
			frames = data['frames']
			fps = []
			for i in range(len(frames)):
				frame = Image.fromarray(frames[i])
				fp = save_dir / f'{cfg.seed:03d}_{episode:03d}_{i:03d}.png'
				frame.save(fp)
				fps.append(fp)
			data['frames'] = fps
			data.update({'metadata': {
				'cfg': cfg,
				'name': name,
				'run_name': run_name,
				'episode': episode,
				'reward': reward,
			}})
			torch.save(data, save_dir / f'{cfg.seed:03d}_{episode:03d}.pt')
	except Exception as e:
		print('Failed to run {}'.format(name))
		print(e)


if __name__ == '__main__':
	cfg = parse_cfg(Path().cwd() / __CONFIG__)
	cfg.demo = True
	if cfg.get('demo_iterations', None) is not None:
		cfg.partition = f'iterations={cfg.demo_iterations}'
	elif cfg.get('demo_variable_std', None) is not None:
		cfg.partition = f'variable_std={cfg.demo_variable_std}'
	parallel(get_demos, cfg, wait=1)
