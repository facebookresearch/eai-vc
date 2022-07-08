import os
import numpy as np
import wandb
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# DEFAULT_PLOT_RC = {
# 	'grid.linestyle': ':',
# 	'grid.linewidth': 2.5,
# 	'axes.labelsize': 14,
# 	'axes.titlesize': 16,
# 	'axes.facecolor': '#F6F6F6',
# 	'axes.edgecolor': '#333',
# 	'legend.fontsize': 14,
# 	'xtick.labelsize': 14,
# 	'xtick.bottom': True,
# 	'xtick.color': '#333',
# 	'ytick.labelsize': 24,
# 	'ytick.left': True,
# 	'ytick.color': '#333',
# }
# sns.set(style='whitegrid', rc=DEFAULT_PLOT_RC)
sns.set(style='white', rc={'figure.figsize':(18,10)})
sns.set_context('paper')
ENTITY = 'nihansen'
PROJECT = 'tdmpc2'


def main():
	tasks = {
		'dmcontrol': ['cup-catch', 'finger-spin', 'cheetah-run', 'walker-run', 'quadruped-run'],
		'metaworld': ['mw-drawer-close', 'mw-hammer', 'mw-box-close', 'mw-pick-place', 'mw-shelf-place']
	}
	exp_names = ['v1', 'offline-v1', 'mocoego-v1', 'mocoego-offline-v1', 'random-v1', 'random-offline-v1']
	num_seeds = 3
	seeds = set(range(1, num_seeds+1))

	api = wandb.Api(timeout=60)
	runs = api.runs(
		os.path.join(ENTITY, PROJECT),
		filters = {
			'$or': [{'tags': task} for task in (tasks['dmcontrol'] + tasks['metaworld'])],
			'$or': [{'tags': f'seed:{s}'} for s in seeds],
			'$or': [{'tags': exp_name} for exp_name in exp_names]}
	)
	print(f'Found {len(runs)} runs after filtering')

	entries = []
	for run in runs:
		cfg = {k: v for k,v in run.config.items()}
		try:
			seed = int(run.name)
		except:
			continue
		task = cfg.get('task', None)
		exp_name = cfg.get('exp_name', None)
		if task not in (tasks['dmcontrol'] + tasks['metaworld']) or \
		   exp_name not in exp_names or \
		   seed not in seeds:
			continue
		key = 'offline/reward' if 'offline' in exp_name else 'eval/episode_reward'
		hist = run.history(keys=[key], x_axis='_step')
		if len(hist) < 3:
			continue
		experiment = cfg['modality'] + '-' + exp_name
		reward = np.array(hist[key])[-1]
		print(f'Appending experiment {experiment} with reward {reward}')
		entries.append(
			(cfg['task'], experiment, seed, reward)
		)
	
	df = pd.DataFrame(entries, columns=['task', 'experiment', 'seed', 'reward'])
	df_dmcontrol = df[df['task'].isin(tasks['dmcontrol'])]
	df_metaworld = df[df['task'].isin(tasks['metaworld'])]

	g = sns.catplot(data=df_dmcontrol, kind='bar', x='task', y='reward', hue='experiment', ci=95)
	g.set_axis_labels('', 'Episode return')
	g.set(ylim=(0, 1000))
	g.savefig(f'singletask-dmcontrol.png')

	g = sns.catplot(data=df_metaworld, kind='bar', x='task', y='reward', hue='experiment', ci=95)
	g.set_axis_labels('', 'Episode return')
	g.set(ylim=(0, 5000))
	g.savefig(f'singletask-metaworld.png')



if __name__ == '__main__':
	main()
