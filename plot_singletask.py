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
	'legend.fontsize': 14,
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
		'dmcontrol': ['cup-catch', 'finger-spin', 'cheetah-run', 'walker-walk', 'walker-run'],
		'metaworld': ['mw-drawer-close', 'mw-drawer-open', 'mw-hammer', 'mw-box-close', 'mw-push']
	}
	exp_names = ['v1', 'offline-v2', 'mocodmcontrol-v1', 'mocodmcontrol-offline-v2', 'mocometaworld-v1', 'mocometaworld-offline-v2', 'mocoego-v1', 'mocoego-offline-v2', 'random-v1', 'random-offline-v2', 'bc-offline-v2']
	experiment2label = {
		'state-v1': 'State',
		'state-offline-v2': '✻ State',
		'pixels-v1': 'Pixels',
		'pixels-offline-v2': '✻ Pixels',
		'features-mocodmcontrol-v1': 'In-domain',
		'features-mocodmcontrol-offline-v2': '✻ In-domain',
		'features-mocometaworld-v1': 'In-domain',
		'features-mocometaworld-offline-v2': '✻ In-domain',
		'features-mocoego-v1': 'Ego4D',
		'features-mocoego-offline-v2': '✻ Ego4D',
		'features-random-v1': 'Random',
		'features-random-offline-v2': '✻ Random',
		'state-bc-offline-v2': '✻ BC (State)',
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
		key = 'offline/reward' if 'offline' in exp_name else 'eval/episode_reward'
		hist = run.history(keys=[key], x_axis='_step')
		if len(hist) < 4:
			continue
		experiment = cfg['modality'] + '-' + exp_name
		label = experiment2label[experiment]
		idx = list(experiment2label.values()).index(label)
		step = np.array(hist['_step'])
		step = step[step <= 500_000]
		if step[-1] < 70_000:
			continue
		reward = np.array(hist[key])
		reward = reward[:min(len(step), len(reward))][-1]
		print(f'Appending experiment {label} for task {task} with reward {reward} at step {step[-1]}')
		entries.append(
			(idx, cfg['task'], label, seed, reward)
		)
	
	df = pd.DataFrame(entries, columns=['idx', 'task', 'experiment', 'seed', 'reward'])
	df_dmcontrol = df[df['task'].isin(tasks['dmcontrol'])]
	df_metaworld = df[df['task'].isin(tasks['metaworld'])]

	# average across tasks
	df_dmcontrol = df_dmcontrol.groupby(['idx', 'experiment', 'seed']).mean().reset_index()
	df_metaworld = df_metaworld.groupby(['idx', 'experiment', 'seed']).mean().reset_index()

	# print unique experiments
	print(df_dmcontrol['experiment'].unique())
	print(df_metaworld['experiment'].unique())
	
	# average across seeds
	df_dmcontrol = df_dmcontrol.groupby(['idx', 'experiment']).mean().reset_index()
	df_metaworld = df_metaworld.groupby(['idx', 'experiment']).mean().reset_index()

	# normalize
	df_dmcontrol['reward'] = (100 * df_dmcontrol['reward'] / df_dmcontrol[df_dmcontrol['experiment'] == 'State']['reward'].values[0]).round()
	df_metaworld['reward'] = (100 * df_metaworld['reward'] / df_metaworld[df_metaworld['experiment'] == 'State']['reward'].values[0]).round()

	f, axs = plt.subplots(1, 2, figsize=(18,6))

	experiment2kwargs = {
		'State': {'color': 'tab:blue'},
		'✻ State': {'color': 'tab:blue'},
		'Pixels': {'color': 'tab:green'},
		'✻ Pixels': {'color': 'tab:green'},
		'In-domain': {'color': 'tab:purple'},
		'✻ In-domain': {'color': 'tab:purple'},
		'Ego4D': {'color': 'tab:pink'},
		'✻ Ego4D': {'color': 'tab:pink'},
		'Random': {'color': 'tab:olive'},
		'✻ Random': {'color': 'tab:olive'},
		'✻ BC (State)': {'color': 'tab:grey'},
	}

	# dmcontrol
	ax = axs[0]
	sns.barplot(data=df_dmcontrol, x='experiment', y='reward', ax=ax, ci=None)
	ax.set_title('DMControl', fontweight='bold')
	ax.set_ylim(0, 115)
	ax.set_xlabel('')
	ax.set_ylabel('Normalized return')
	ax.bar_label(ax.containers[0], fontsize=18)
	ax.tick_params(labelrotation=35)
	for i in range(1, len(df_dmcontrol), 2):
		ax.containers[0].patches[i]._hatch = '//'
		ax.containers[0].patches[i].set_facecolor(ax.containers[0].patches[i-1]._facecolor)
	# for i in range(1, len(df_dmcontrol)):
	# 	if '✻' in df_dmcontrol.iloc[i]['experiment']:
	# 		ax.containers[0].patches[i]._hatch = '//'
	# 	ax.containers[0].patches[i].set_facecolor(experiment2kwargs[df_dmcontrol.iloc[i]['experiment']]['color'])
	
	# metaworld
	ax = axs[1]
	sns.barplot(data=df_metaworld, x='experiment', y='reward', ax=ax, ci=None)
	ax.set_title('Meta-World', fontweight='bold')
	ax.set_ylim(0, 115)
	ax.set_xlabel('')
	ax.set_ylabel('')
	ax.bar_label(ax.containers[0], fontsize=18)
	ax.tick_params(labelrotation=35)
	for i in range(1, len(df_metaworld), 2):
		ax.containers[0].patches[i]._hatch = '//'
		ax.containers[0].patches[i].set_facecolor(ax.containers[0].patches[i-1]._facecolor)

	h, l = axs[0].get_legend_handles_labels()
	f.legend(h, l, loc='lower center', ncol=4, frameon=False)
	plt.tight_layout()
	f.subplots_adjust(bottom=0.16, wspace=0.15)
	plt.savefig(Path(make_dir('plots')) / 'singletask.png', bbox_inches='tight')


if __name__ == '__main__':
	main()
