from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root=Path(__file__).resolve().parent
seeds=json.loads((root/'results/seed_comparisons.json').read_text())
panel=json.loads((root/'results/panel_comparisons.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
fig,axes=plt.subplots(1,2,figsize=(11,5.5),sharex=True,sharey=True)
for ax,obs,title in zip(axes,['cls_current','full_current'],['Current CLS observations','Current all-token observations']):
 rows=[r for r in seeds if r['observer']==obs and r['baseline']=='current_rff']
 total=next(r for r in panel if r['observer']==obs and r['baseline']=='current_rff')
 ax.axvline(0,color='#606875',lw=1)
 ax.axvline(.01,color='#bac0c9',lw=1,ls='--')
 for j,row in enumerate(rows):
  mean=row['delta_r2'];lo,hi=row['conditional_ci95'];color='#28698f' if row['seed']<811 else '#ac6b35'
  ax.plot([lo,hi],[6-j,6-j],color=color,lw=1.6)
  ax.scatter(mean,6-j,color=color,marker='o' if row['seed']<811 else 's',s=34,zorder=3)
 mean=total['mean_delta_r2'];lo,hi=total['conditional_ci95']
 ax.plot([lo,hi],[0,0],color='#212b38',lw=2.5)
 ax.scatter(mean,0,color='#212b38',marker='D',s=45,zorder=3)
 ax.axhline(.55,color='#e5e8ed',lw=1)
 ax.set_title(title,fontsize=12,pad=12,loc='left')
 ax.set_xlim(-.25,.115);ax.set_xticks([-.2,-.1,0,.1])
 ax.set_ylim(-.65,6.65)
 ax.grid(axis='x',color='#e9ecf0',lw=.7)
 ax.set_axisbelow(True)
 ax.set_yticks([6,5,4,3,2,1,0],['503','607','709','811','907','1009','Six-model mean'])
 ax.tick_params(axis='y',length=0,labelleft=True)
 ax.set_xlabel('Difference in held-out R² (history − current control)',labelpad=10)
fig.suptitle('T2-E1 · No robust history advantage in the fixed exploratory panel',x=.075,y=.97,ha='left',fontsize=15,fontweight='semibold')
fig.text(.075,.905,'Ordered CLS history versus a current-only nonlinear control with the same fitted coefficient count',fontsize=10,color='#5b6270')
legend=[Line2D([0],[0],marker='o',linestyle='',color='#28698f',label='Previously passed competence gates'),Line2D([0],[0],marker='s',linestyle='',color='#ac6b35',label='Previously failed competence gates')]
fig.legend(handles=legend,loc='lower left',bbox_to_anchor=(.07,.07),ncol=2,frameon=False,fontsize=9)
fig.text(.075,.035,'Whiskers: 95% paired-family bootstrap intervals, conditional on fixed models and fitted probes. Positive favors history.',fontsize=8.5,color='#5b6270')
fig.text(.075,.008,'Steps weighted equally within each seed; seeds weighted equally. Dashed line: frozen ΔR² screening threshold of 0.01.',fontsize=8.5,color='#5b6270')
fig.subplots_adjust(left=.15,right=.97,bottom=.25,top=.83,wspace=.4)
fig.savefig(root/'T2_E1_History_Effects.png',dpi=220,bbox_inches='tight',facecolor='white')
fig.savefig(root/'T2_E1_History_Effects.svg',bbox_inches='tight',facecolor='white')
print('Scientific figure rendered from verified saved results')
