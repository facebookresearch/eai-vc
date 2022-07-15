import os
import numpy as np
import pandas as pd
from pathlib import Path
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


def get_eval_mse(logging_dir, task, source, target, exp_name, seed):
	fp = logging_dir / task / source / target / exp_name / str(seed) / 'metrics.csv'
	df = pd.read_csv(fp)
	try:
		eval_mse = df['eval_mse'][50]
	except:
		# return last value instead + throw warning
		eval_mse = df['eval_mse'].iloc[-1]
		print('Experiment', task, source, target, exp_name, seed, 'only has', len(df['eval_mse']), 'entries')
	return eval_mse


def main():
	tasks = {
		'dmcontrol': ['cup-catch', 'finger-spin', 'cheetah-run', 'walker-run', 'quadruped-run'],
		'metaworld': ['mw-drawer-close', 'mw-drawer-open', 'mw-hammer', 'mw-box-close', 'mw-pick-place']
	}
	logging_dir = Path('/private/home/nihansen/code/tdmpc2/renderer')

	source2exp = {
		'pixels': 'offline-v1',
		'mocoego': 'mocoego-offline-v1',
		'mocodmcontrol': 'mocodmcontrol-offline-v1',
		'random': 'random-offline-v1',
	}
	source2label = {
		'pixels': 'Pixels',
		'mocodmcontrol': 'In-domain',
		'mocoego': 'Ego4D',
		'random': 'Random',
	}

	entries = []
	for task in tasks['dmcontrol']:
		for source in ['pixels', 'mocoego', 'mocodmcontrol', 'random']:
			for target in ['pixels', 'state']:
				for seed in range(1, 3+1):
					try:
						idx = list(source2label.keys()).index(source)
						eval_mse = get_eval_mse(logging_dir, task, source, target, source2exp[source], seed)
						entries.append(
							(idx, task, source2label[source], target, seed, eval_mse)
						)
					except FileNotFoundError:
						print('File not found:', task, source, target, seed)

	df = pd.DataFrame(entries, columns=['idx', 'task', 'source', 'target', 'seed', 'eval_mse'])

	# min across seeds
	df = df.groupby(['idx', 'task', 'source', 'target']).min().reset_index()

	# median across tasks
	df = df.groupby(['idx', 'source', 'target']).median().reset_index()

	# subgroup by target
	df_pixels = df[df['target'] == 'pixels']
	df_state = df[df['target'] == 'state']

	# rescale mse by max value
	df_state['eval_mse'] = (100 * df_state['eval_mse'] / df_state['eval_mse'].max()).round()
	df_pixels['eval_mse'] = (100 * df_pixels['eval_mse'] / df_pixels['eval_mse'].max()).round()

	f, axs = plt.subplots(1, 2, figsize=(18,6))

	# colors (use tableau palette)
	colors = sns.color_palette()[::2][1:]

	# state
	ax = axs[0]
	sns.barplot(data=df_state, x='source', y='eval_mse', ax=ax, ci=None, palette=colors)
	ax.set_title('State prediction', fontweight='bold')
	ax.set_ylim(0, 110)
	ax.set_xlabel('')
	ax.set_ylabel('Normalized MSE')
	ax.bar_label(ax.containers[0], fontsize=18)
	ax.tick_params(labelrotation=35)
	
	# pixels
	ax = axs[1]
	sns.barplot(data=df_pixels, x='source', y='eval_mse', ax=ax, ci=None, palette=colors)
	ax.set_title('Pixel prediction', fontweight='bold')
	ax.set_ylim(0, 110)
	ax.set_xlabel('')
	ax.set_ylabel('Normalized MSE')
	ax.bar_label(ax.containers[0], fontsize=18)
	ax.tick_params(labelrotation=35)

	h, l = axs[0].get_legend_handles_labels()
	f.legend(h, l, loc='lower center', ncol=4, frameon=False)
	plt.tight_layout()
	f.subplots_adjust(bottom=0.16, wspace=0.15)
	plt.savefig('plot_rendering.png', bbox_inches='tight')


if __name__ == '__main__':
	main()
