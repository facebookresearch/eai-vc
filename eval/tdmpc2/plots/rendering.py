import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
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


def get_metric(logging_dir, task, source, target, exp_name, seed, metric='eval_mse'):
	fp = logging_dir / task / source / target / exp_name / str(seed) / 'metrics.csv'
	return pd.read_csv(fp)[metric].iloc[-1]


def main():
	tasks = {
		'dmcontrol': ['cup-catch', 'finger-spin', 'cheetah-run', 'walker-walk', 'walker-run'],
		'metaworld': ['mw-drawer-close', 'mw-drawer-open', 'mw-hammer', 'mw-box-close', 'mw-push']
	}
	logging_dir = Path('/private/home/nihansen/code/tdmpc2/renderer')

	source2exp = {
		'pixels': 'offline-v2',
		'mocoego': 'mocoego-offline-v2',
		'mocodmcontrol': 'mocodmcontrol-offline-v2',
		# 'mocometaworld': 'mocometaworld-offline-v2',
		'random': 'random-offline-v2',
	}
	source2label = {
		'pixels': 'Pixels',
		'mocodmcontrol': 'In-domain',
		# 'mocometaworld': 'In-domain',
		'mocoego': 'Ego4D',
		'random': 'Random',
	}
	metric = 'total_loss'

	entries = []
	for task in ['cheetah-run']: #tasks['dmcontrol'] + tasks['metaworld']:
		for source in ['pixels', 'mocoego', 'mocodmcontrol', 'random']: # 'mocometaworld'
			for target in ['pixels', 'state']:
				for seed in range(1, 3+1):
					try:
						idx = list(source2label.keys()).index(source)
						val = get_metric(logging_dir, task, source, target, source2exp[source], seed, metric=metric)
						entries.append(
							(idx, task, source2label[source], target, seed, val)
						)
					except FileNotFoundError:
						print('File not found:', task, source, target, seed)

	df = pd.DataFrame(entries, columns=['idx', 'task', 'source', 'target', 'seed', metric])

	# mean across seeds
	df = df.groupby(['idx', 'task', 'source', 'target']).min().reset_index()

	# mean across tasks
	# df = df.groupby(['idx', 'source', 'target']).median().reset_index()

	# select a single task
	df = df[df['task'] == 'cheetah-run']

	# subgroup by target
	df_pixels = df[df['target'] == 'pixels']
	df_state = df[df['target'] == 'state']

	# rescale mse by max value
	df_state[metric] = (100 * df_state[metric] / df_state[metric].max()).round()
	df_pixels[metric] = (100 * df_pixels[metric] / df_pixels[metric].max()).round()

	f, axs = plt.subplots(1, 2, figsize=(18,6))

	# color palette
	colors = []
	for color in sns.color_palette('colorblind')[1:]:
		colors.extend([color])

	# state
	ax = axs[0]
	sns.barplot(data=df_state, x='source', y=metric, ax=ax, ci=None, palette=colors)
	ax.set_title('State prediction', fontweight='bold')
	ax.set_ylim(0, 110)
	ax.set_xlabel('')
	ax.set_ylabel('Normalized MSE')
	ax.bar_label(ax.containers[0], fontsize=18)
	ax.tick_params(labelrotation=35)
	
	# pixels
	ax = axs[1]
	sns.barplot(data=df_pixels, x='source', y=metric, ax=ax, ci=None, palette=colors)
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
	plt.savefig(Path(make_dir('plots')) / 'rendering.png', bbox_inches='tight')


if __name__ == '__main__':
	main()
