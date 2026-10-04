"""Render the fixed calibration outcomes and their Monte Carlo uncertainty."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT/'results/detection_summary.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
xs = np.arange(4)
p = np.array([r['detection_fraction'] for r in data])
ci = np.array([r['monte_carlo_wilson_ci95'] for r in data])
err = np.maximum(np.array([p-ci[:, 0], ci[:, 1]-p]), 0)
ax.errorbar(xs, p, yerr=err, fmt='o', color='#156b73', markersize=10,
            capsize=7, elinewidth=2, zorder=3)
for x, row in zip(xs, data):
    y = row['detection_fraction']
    ax.annotate(f"{row['detected_panels']}/24", (x, y), xytext=(0, 14 if y == 1 else -24),
                textcoords='offset points', ha='center', fontweight='bold')
ax.set_xticks(xs, ['0\nNull', '0.01\nSmall signal', '0.05', '0.10'])
ax.set_xlim(-.45, 3.45)
ax.set_ylim(-.17, 1.17)
ax.set_yticks(np.arange(0, 1.01, .25))
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xlabel('Known population improvement in R² from history', labelpad=10)
ax.set_ylabel('Panels passing the complete screen')
ax.grid(axis='y', color='#dce1e5', alpha=.8)
ax.set_title('T2-C1 | Synthetic sensitivity calibration', loc='left', fontsize=16, fontweight='bold', pad=24)
fig.text(.5, -.055, '24 independent panels × 6 datasets • Bars: 95% Wilson Monte Carlo intervals\n'
         'Gaussian linear signals at step-2 dimensions; this does not estimate power for natural Transformer signals.',
         ha='center', fontsize=9, color='#48545d')
fig.savefig(ROOT/'T2_C1_Sensitivity.png', dpi=180, bbox_inches='tight', facecolor='white')
fig.savefig(ROOT/'T2_C1_Sensitivity.svg', bbox_inches='tight', facecolor='white')
plt.close(fig)
