"""Scientific figure from closed main results; no fitting or outcome selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'results/seed_comparisons.json').read_text())
panels=json.loads((ROOT/'results/panel_comparisons.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,5.7),sharey=True,layout='constrained')
all_bounds=[]
for ax,baseline,title,color in zip(axes,['current_rff_mlp','current_reference'],
        ['Matched current-state MLP','Selected current-only reference'],['#146f7a','#a65034']):
    cells=[r for r in rows if r['baseline']==baseline and r['relative_mse_gain'] is not None]
    panel=next(r for r in panels if r['baseline']==baseline)
    points=np.array([r['relative_mse_gain'] for r in cells]+[panel['mean_relative_mse_gain']])
    intervals=np.array([r['conditional_gain_ci95'] for r in cells]+[panel['conditional_gain_ci95']])
    # The percentile interval need not contain the point estimate; draw it independently.
    ys=np.arange(len(points))
    ax.hlines(ys,intervals[:,0],intervals[:,1],color=color,lw=1.8)
    ax.plot(points[:-1],ys[:-1],'o',color=color,ms=6)
    ax.plot(points[-1],ys[-1],'D',color=color,ms=8)
    ax.axvline(0,color='#59636b',lw=1)
    ax.axvline(.05,color='#899298',ls='--',lw=1)
    ax.axhline(len(points)-1.5,color='#d6dde0',lw=1)
    ax.set_title(title,loc='left',fontsize=12,pad=15)
    ax.set_xlabel('Relative MSE reduction with history',labelpad=10)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='x',color='#e3e8ea',alpha=.7)
    all_bounds.extend(intervals.ravel());all_bounds.extend(points)
axes[0].set_yticks(np.arange(7),['503','607','709','811','907','1009','Six-model mean'])
axes[0].invert_yaxis()
lo=min(all_bounds+[0]);hi=max(all_bounds+[.05]);span=max(hi-lo,.1)
for ax in axes:ax.set_xlim(lo-.10*span,hi+.10*span)
fig.suptitle('T2-E2 | Does history help the nonlinear observer?',fontsize=17,fontweight='bold',x=.01,ha='left')
fig.text(.5,-.05,'Positive values favor history • Dashed line: prespecified 5% useful-gain threshold\n'
         'Intervals: 95% paired-family bootstrap, conditional on six fixed models and selected probes.',
         ha='center',fontsize=9,color='#48545d')
fig.savefig(ROOT/'T2_E2_History_Effects.png',dpi=180,bbox_inches='tight',facecolor='white')
fig.savefig(ROOT/'T2_E2_History_Effects.svg',bbox_inches='tight',facecolor='white')
plt.close(fig)
