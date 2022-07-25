import os
import numpy as np
import wandb
import pandas as pd
from pathlib import Path
from logger import make_dir
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
DEFAULT_PLOT_RC = {
	'axes.labelsize': 14,
	'axes.titlesize': 20,
	'axes.facecolor': '#F6F6F6',
	'axes.edgecolor': '#333',
	'legend.fontsize': 16,
	'xtick.labelsize': 14,
	'xtick.bottom': True,
	'xtick.color': '#333',
	'ytick.labelsize': 16,
	'ytick.left': True,
	'ytick.color': '#333',
}
sns.set(style='whitegrid', rc=DEFAULT_PLOT_RC)
ENTITY = 'nihansen'
PROJECT = 'tdmpc2'


def main():
	tasks = {
		'dmcontrol': ['cup-catch', 'finger-spin', 'cheetah-run', 'walker-run', 'quadruped-run'],
		'metaworld': ['mw-drawer-close', 'mw-drawer-open', 'mw-hammer', 'mw-box-close', 'mw-pick-place']
	}
	exp_names = ['v1', 'mocodmcontrol-v1', 'mocometaworld-v1', 'mocoego-v1', 'random-v1']
	experiment2label = {
		'state-v1': 'State',
		'state-offline-v1': '✻ State',
		'pixels-v1': 'Pixels',
		'pixels-offline-v1': '✻ Pixels',
		'features-mocodmcontrol-v1': 'In-domain',
		'features-mocodmcontrol-offline-v1': '✻ In-domain',
		'features-mocometaworld-v1': 'In-domain',
		'features-mocometaworld-offline-v1': '✻ In-domain',
		'features-mocoego-v1': 'Ego4D',
		'features-mocoego-offline-v1': '✻ Ego4D',
		'features-random-v1': 'Random',
		'features-random-offline-v1': '✻ Random',
	}
	num_seeds = 3
	seeds = set(range(1, num_seeds+1))

	api = wandb.Api(timeout=100)
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
		key = 'eval/episode_reward'
		hist = run.history(keys=[key], x_axis='_step')
		if len(hist) < 4:
			continue
		experiment = cfg['modality'] + '-' + exp_name
		label = experiment2label[experiment]
		idx = list(experiment2label.values()).index(label)
		step = np.array(hist['_step'])
		step = step[step <= 500_000]
		reward = np.array(hist[key])
		reward = reward[:min(len(step), len(reward))]
		print(f'Appending experiment {label} with {len(step)} steps')
		for i in range(len(step)):
			entries.append(
				(idx, cfg['task'], label, seed, step[i], reward[i])
			)
	
	df = pd.DataFrame(entries, columns=['idx', 'task', 'experiment', 'seed', 'step', 'reward'])
	df_dmcontrol = df[df['task'].isin(tasks['dmcontrol'])]
	df_metaworld = df[df['task'].isin(tasks['metaworld'])]

	# print unique experiments
	print(df_dmcontrol['experiment'].unique())
	print(df_metaworld['experiment'].unique())

	# average across tasks
	df_dmcontrol = df_dmcontrol.groupby(['idx', 'experiment', 'seed', 'step']).mean().reset_index()
	df_metaworld = df_metaworld.groupby(['idx', 'experiment', 'seed', 'step']).mean().reset_index()

	# average across seeds
	# df_dmcontrol = df_dmcontrol.groupby(['idx', 'experiment', 'step']).mean().reset_index()
	# df_metaworld = df_metaworld.groupby(['idx', 'experiment', 'step']).mean().reset_index()

	# rescale reward
	df_dmcontrol['reward'] = df_dmcontrol['reward'] / 10
	df_metaworld['reward'] = df_metaworld['reward'] / 45

	# rescale step
	df_dmcontrol['step'] = df_dmcontrol['step'] / 1e6
	df_metaworld['step'] = df_metaworld['step'] / 1e6

	f, axs = plt.subplots(1, 2, figsize=(16,6))

	experiment2kwargs = {
		'State': {'color': 'tab:blue'},
		'Pixels': {'color': 'tab:green'},
		'In-domain': {'color': 'tab:purple'},
		'Ego4D': {'color': 'tab:pink'},
		'Random': {'color': 'tab:olive'},
	}

	# dmcontrol
	ax = axs[0]
	for experiment in experiment2kwargs.keys():
		df_dmcontrol_exp = df_dmcontrol[df_dmcontrol['experiment'] == experiment]
		sns.lineplot(
				data=df_dmcontrol_exp,
				x='step',
				y=f'reward',
				ci=95,
				label=experiment,
				color=experiment2kwargs[experiment]['color'],
				legend=False,
				linewidth=3,
				linestyle='-',
				alpha=0.8,
				err_kws={'alpha': .095},
				ax=ax
			)
	ax.set_title('DMControl', fontweight='bold')
	ax.set_xlim(0, 0.5)
	ax.set_ylim(0, 100)
	ax.set_xlabel('Environment steps (M)')
	ax.set_ylabel('Normalized return')

	# metaworld
	ax = axs[1]
	for experiment in experiment2kwargs.keys():
		df_metaworld_exp = df_metaworld[df_metaworld['experiment'] == experiment]
		sns.lineplot(
				data=df_metaworld_exp,
				x='step',
				y=f'reward',
				ci=95,
				label=experiment,
				color=experiment2kwargs[experiment]['color'],
				legend=False,
				linewidth=4,
				linestyle='-',
				alpha=0.8,
				err_kws={'alpha': .08},
				ax=ax
			)
	ax.set_title('Meta-World', fontweight='bold')
	ax.set_xlim(0, 0.5)
	ax.set_ylim(0, 100)
	ax.set_xlabel('Environment steps (M)')
	ax.set_ylabel('')

	h, l = axs[0].get_legend_handles_labels()
	f.legend(h, l, loc='lower center', ncol=5, frameon=False)
	plt.tight_layout()
	f.subplots_adjust(bottom=0.2, wspace=0.15)
	plt.savefig(Path(make_dir('plots')) / 'online.png', bbox_inches='tight')




if __name__ == '__main__':
	main()
