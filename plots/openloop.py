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


def main():
	tasks = {
		'dmcontrol': ['cup-catch', 'finger-spin', 'cheetah-run', 'walker-walk', 'walker-run'],
		'metaworld': ['mw-drawer-close', 'mw-drawer-open', 'mw-hammer', 'mw-box-close', 'mw-push']
	}
	# openloop_dir = make_dir(Path(cfg.logging_dir) / 'openloop' / cfg.task / (cfg.features if cfg.modality=='features' else cfg.modality) / cfg.exp_name / str(cfg.horizon) / str(cfg.seed))
	logging_dir = Path('/private/home/nihansen/code/tdmpc2/openloop')

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
	horizon2label = {
		5: '5 steps (1%)',
		50: '50 steps (10%)',
		500: '500 steps (100%)',
	}
	metric = 'episode_reward'

	entries = []
	for task in tasks['dmcontrol']: # + tasks['metaworld']:
		for source in ['pixels', 'mocoego', 'mocodmcontrol', 'random']: # 'mocometaworld' # 'pixels'
			for horizon in [5, 50, 500]:
				for seed in range(1, 3+1):
					fp = logging_dir / task / source / source2exp[source] / str(horizon) / str(seed) / 'metrics.csv'
					try:
						df = pd.read_csv(fp)
						df = df.groupby(['exp_name']).mean().reset_index()
						idx = list(source2label.keys()).index(source)
						entries.append(
							(idx, task, source2label[source], horizon2label[horizon], seed, int(df[metric]))
						)
					except FileNotFoundError:
						print('File not found:', task, source, seed)

	df = pd.DataFrame(entries, columns=['idx', 'task', 'experiment', 'horizon', 'seed', metric])

	# mean across seeds
	df = df.groupby(['idx', 'task', 'experiment', 'horizon']).mean().reset_index()

	# mean across tasks
	df = df.groupby(['idx', 'experiment', 'horizon']).mean().reset_index()

	f, ax = plt.subplots(1, 1, figsize=(8,4.5))

	# color palette
	colors = sns.color_palette('colorblind')

	# plot
	g = sns.barplot(data=df, x='experiment', y=metric, hue='horizon', ax=ax, ci=None, palette=colors)
	g.legend_.remove()
	ax.set_title('Open-loop planning', fontweight='bold')
	ax.set_ylim(0, 600)
	ax.set_xlabel('')
	ax.set_ylabel('Episode return')

	h, l = ax.get_legend_handles_labels()
	f.legend(h, l, loc='lower center', ncol=4, frameon=False)
	plt.tight_layout()
	f.subplots_adjust(bottom=0.185, wspace=0.15)
	plt.savefig(Path(make_dir('plots')) / 'openloop.png', bbox_inches='tight')


if __name__ == '__main__':
	main()
