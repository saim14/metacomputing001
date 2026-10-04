"""Render the audited, frozen F1-A1 result; no fitting or study extension."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
R = json.loads((ROOT / 'run/result.json').read_text())
A = json.loads((ROOT / 'run/audit.json').read_text())
M = json.loads((ROOT / 'run/manifest.json').read_text())
assert A['passed'] and A['source_and_checkpoint_hashes_unchanged']

LABELS = {'c':'C', 'ch':'CH', 'cg':'CG', 'chg':'CHG', 'cr':'CR',
          'chr':'CHR', 'cgr':'CGR', 'chgr':'CHGR', 'ce':'C-expanded',
          'cre':'CR-expanded', 'craw':'C-raw', 'crraw':'CR-raw',
          'cmhg':'C-mismatched-H-G', 'chmg':'C-H-mismatched-G',
          'cshg':'C-shuffled-H-G', 'chgmr':'C-H-G-mismatched-R'}


def condition(row):
    return 'Exact' if row['seed'] is None else f"{row['seed']} / {row['alpha']:g}"


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def ci(values, scale=1, digits=7):
    return f"[{values[0]*scale:+.{digits}f}, {values[1]*scale:+.{digits}f}]"


def draw_figure():
    teal, orange, blue, ink, muted = '#087F8C', '#B9652C', '#4F65A5', '#23333B', '#65757D'
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'text.color':ink, 'axes.labelcolor':ink,
                         'xtick.color':ink, 'ytick.color':ink,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.edgecolor':'#CDD6DA', 'axes.titleweight':'bold'})
    fig, axs = plt.subplots(2, 2, figsize=(13.8, 9.6))
    fig.subplots_adjust(left=.083, right=.97, top=.86, bottom=.16, wspace=.34, hspace=.59)
    fig.suptitle('F1-A1  |  Model error can make extra planning harmful',
                 x=.083, y=.975, ha='left', fontsize=20, fontweight='bold')
    fig.text(.083,.925,'The intervention passed; the combined predictor and controller did not meet their improvement criteria.',
             fontsize=11, color=muted)
    ax = axs[0,0]
    for seed, color, marker, style in [(3109, teal, 'o', '-'),(3119, orange, 's', '--'),(3137, blue, '^', ':')]:
        rows = [r for r in R['per_condition'] if r['seed']==seed]
        ax.plot([0,.5,1], [0]+[r['harmful_fraction']*100 for r in rows],
                color=color, marker=marker, linestyle=style, linewidth=2,
                markersize=6, label=f'Model {seed}')
    ax.set(title='A  Harmful continuation increases with error',
           xlabel='Fraction of frozen neural error retained (α)',
           ylabel='Scenes harmed by > 0.01 cost units (%)',
           xticks=[0,.5,1], xlim=(-.035,1.035), ylim=(-1,36))
    ax.grid(axis='y', alpha=.17)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    h = R['headline']['error_intervention']
    ax.text(.99,.16,f"Full error − exact: +{h['mean']*100:.1f} pp\n98.75% CI: {h['ci9875'][0]*100:.1f} to {h['ci9875'][1]*100:.1f} pp",
            transform=ax.transAxes, ha='right', fontsize=9,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.95})

    ax=axs[0,1]
    for y,key,color in [(1,'joint_history_geometry',teal),(0,'added_reliability',blue)]:
        h=R['headline'][key]; mean=h['mean']*1000
        low,high=np.array(h['ci9875'])*1000
        ax.errorbar(mean,y,xerr=[[mean-low],[high-mean]],fmt='o',color=color,
                    capsize=5,markersize=7,linewidth=2)
        ax.scatter(h['reference_mse']*.05*1000,y,marker='|',s=350,color=orange,zorder=4)
        ax.text(mean,y+.20,f"{h['relative_reduction']*100:+.2f}% relative",ha='center',fontsize=9,color=color)
    ax.axvline(0,color=muted,linewidth=1)
    ax.set(title='B  Predictive improvements were not established',
           xlabel='Reference MSE − candidate MSE (× 10⁻³)',
           yticks=[0,1],yticklabels=['Added R\nCHGR vs CHG','History + geometry\nCHG vs current reference'],
           xlim=(-.25,.97),ylim=(-.6,1.6))
    ax.grid(axis='x',alpha=.17)
    ax.legend(handles=[Line2D([0],[0],color=muted,marker='o',label='98.75% paired interval'),
                       Line2D([0],[0],color=orange,marker='|',markersize=12,linestyle='None',label='5% point-gain requirement')],
              frameon=False,fontsize=8,loc='lower right')

    ax=axs[1,0]
    h=R['headline']['charged_controller']; mean=h['mean']*1000
    low,high=np.array(h['ci9875'])*1000
    ax.errorbar(mean,0,xerr=[[mean-low],[high-mean]],fmt='o',color=teal,
                capsize=7,markersize=8,linewidth=2)
    ax.axvline(0,color=muted,linewidth=1)
    threshold=h['reference_objective']*.01*1000
    ax.scatter(threshold,0,marker='|',s=450,color=orange)
    ax.text(threshold,.28,'1% point-gain\nrequirement',ha='center',fontsize=9,color=orange)
    ax.text(mean,-.31,f"Relative gain: {h['relative_reduction']*100:+.2f}%\n98.75% interval includes zero",ha='center',fontsize=10)
    ax.set(title='C  No gain established in the charged objective',
           xlabel='Current reference cost − CHGR cost (× 10⁻³)',
           yticks=[],xlim=(-3.8,3.4),ylim=(-.8,.8))
    ax.grid(axis='x',alpha=.17)

    ax=axs[1,1]
    rows=R['runtime']['rows']; ypos=np.arange(len(rows))
    candidate=[r['ms_per_case']['chgr'] for r in rows]
    reference=[r['ms_per_case'][R['controller_refs'][i]] for i,r in enumerate(rows)]
    ax.barh(ypos-.17,reference,height=.31,label='Current reference',color='#A4B2BC')
    ax.barh(ypos+.17,candidate,height=.31,label='CHGR',color=teal)
    ax.set(title='D  Added features carried a runtime cost',
           xlabel='Median repetition mean (milliseconds per case)',
           yticks=ypos,yticklabels=[condition(r) for r in R['per_condition']],xlim=(0,5.05))
    ax.invert_yaxis(); ax.grid(axis='x',alpha=.17)
    ax.legend(frameon=False,fontsize=9,loc='lower right')
    fig.text(.083,.038,'A–C: 800 paired evaluation scenes; intervals condition on three fixed neural checkpoints. B–C pool six imperfect variants.\n'
             'D: first 24 evaluation scenes, three timed repetitions, one CPU thread. Exact dynamics are a single shared control.',
             fontsize=9,color=muted,linespacing=1.6)
    fig.savefig(ROOT/'F1_A1_Results.png',dpi=180,facecolor='white')
    plt.close(fig)


def write_report():
    h=R['headline']; per=R['per_condition']; imperfect=per[1:]
    headline_rows=[['Model-error intervention',f"+{h['error_intervention']['mean']*100:.2f} percentage points",
                    ci(h['error_intervention']['ci9875'],100,2)+' pp','PASS: ≥5 pp and positive lower bound']]
    for key,title,units,cutoff in [('joint_history_geometry','History + geometry','MSE',5),
                                  ('added_reliability','Added reliability descriptor','MSE',5),
                                  ('charged_controller','Combined controller','cost',1)]:
        d=h[key]
        headline_rows.append([title,f"{d['mean']:+.7f} {units}; {d['relative_reduction']*100:+.3f}% relative",
                              ci(d['ci9875'])+f' {units}',f'FAIL: required ≥{cutoff}% and positive lower bound'])
    intervention=table(['Model / α','One-step RMSE¹','Ten-step rollout RMSE²','Harmful continuation','Mean true benefit V₃'],
        [[condition(p),f"{d['one_step_rmse']:.6f}",f"{d['ten_step_rmse']:.6f}",
          f"{p['harmful_fraction']*100:.3f}%",f"{p['value_mean']:.6f}"] for p,d in zip(per,R['world_diagnostics'])])
    dims={arm:R['selected_observers'][0][arm]['input_dimensions'] for arm in R['arms']}
    all_arms=table(['Observer input','Input width','Pooled evaluation MSE'],
        [[LABELS[arm],dims[arm],f"{R['pooled_imperfect_arm_mse'][arm]:.8f}"] for arm in R['arms']]+
        [['Fit-mean constant',0,f"{np.mean([p['constant_mse'] for p in imperfect]):.8f}"]])
    predictions=table(['Model / α','No-R reference','Reference MSE','CHG MSE','CHGR MSE','Constant MSE'],
        [[condition(p),LABELS[p['reference_no_r']],f"{p['arm_mse'][p['reference_no_r']]:.8f}",
          f"{p['arm_mse']['chg']:.8f}",f"{p['arm_mse']['chgr']:.8f}",f"{p['constant_mse']:.8f}"] for p in per])
    control_labels={'ch':'CH vs CHG','cg':'CG vs CHG','cmhg':'Mismatched H vs CHG',
                    'chmg':'Mismatched G vs CHG','cshg':'Shuffled H vs CHG',
                    'reliability_mismatch':'Mismatched R vs CHGR',
                    'combined_vs_current_reliability':'Strong R reference vs CHGR'}
    controls=table(['Comparison (reference vs candidate)','MSE reduction','Descriptive 95% interval'],
        [[control_labels[k],f"{v['mean']:+.7f}",ci(v['ci95'])] for k,v in R['descriptive_comparisons'].items()])
    controller=table(['Model / α','Current controller','CHGR charged cost','Reference charged cost','CHGR continue','Reference continue'],
        [[condition(p),LABELS[p['controller_reference']],f"{p['candidate_charged_cost']:.6f}",
          f"{p['reference_charged_cost']:.6f}",f"{p['candidate_continue_fraction']*100:.2f}%",
          f"{p['reference_continue_fraction']*100:.2f}%"] for p in per])
    true_costs=table(['Model / α','CHGR true cost','Reference true cost','CHGR harmful continuation³','Reference harmful continuation³'],
        [[condition(p),f"{p['candidate_true_cost']:.6f}",f"{p['reference_true_cost']:.6f}",
          f"{p['candidate_harmful_continuation_fraction']*100:.3f}%",
          f"{p['reference_harmful_continuation_fraction']*100:.3f}%"] for p in per])
    comparators=table(['Model / α','Always stop','Always continue','Hindsight oracle'],
        [[condition(p),f"{p['stop_cost']:.6f}",f"{p['continue_cost']:.6f}",f"{p['oracle_cost']:.6f}"] for p in per])
    sensitivity=table(['Charge','Reference minus CHGR cost','Descriptive 95% interval','CHGR continue','Reference continue'],
        [[k,f"{v['difference']['mean']:+.7f}",ci(v['difference']['ci95']),
          f"{v['candidate_continue_fraction']*100:.2f}%",f"{v['reference_continue_fraction']*100:.2f}%"]
         for k,v in R['charge_sensitivity'].items()])
    timing=table(['Model / α','CHGR ms','Reference ms','Always-stop ms','Always-continue ms'],
        [[condition(per[i]),f"{r['ms_per_case']['chgr']:.3f}",
          f"{r['ms_per_case'][R['controller_refs'][i]]:.3f}",
          f"{r['ms_per_case']['always_stop']:.3f}",f"{r['ms_per_case']['always_continue']:.3f}"]
         for i,r in enumerate(R['runtime']['rows'])])
    penalties=Counter(v['alpha'] for row in R['selected_observers'] for v in row.values())
    bases=Counter(v['basis_index'] for row in R['selected_observers'] for v in row.values())
    report=f'''# F1-A1 — Reliability, history and geometry in metacomputation

Research 002 · Saim 13.02  
Experiment: 17 September 2026 · Report: 18 September 2026 · Version 1.0  
Status: completed, audited development study; four prespecified headline outcomes retained.

## Finding

Increasing the error of the planner's transition model caused more harmful extra planning in this controlled toy environment. With full frozen neural error, continuing from planning round 3 to round 6 increased true plan cost by more than 0.01 in **20.67% of scenes on average across the three models**, compared with 0% under exact dynamics. The paired risk difference was **+20.67 percentage points**, with a 98.75% bootstrap interval of **[+18.50, +22.92] points**.

This identifies a project-specific failure mechanism; the tested metacomputation features did not solve it. History plus future geometry reduced prediction MSE by only **0.724%** against its tuning-selected current-based reference. Adding the reliability descriptor slightly worsened MSE, and the combined stop/continue controller slightly worsened the charged decision objective. All three improvement criteria failed. Failure to establish an advantage does not prove equivalence or rule out a different representation, decoder or environment.

The exact-model result has a structural explanation: the planner retains its incumbent, so an exact model cannot choose a strictly worse plan under the same cost function. The empirical contribution is the measured harm and its change under the substituted neural error fields, together with the unsuccessful tests of our proposed predictors. This is not a first discovery of model bias.

![Four-panel summary of F1-A1](F1_A1_Results.png)

## Four headline outcomes

Positive differences favor the claimed effect or candidate. Prediction and decision endpoints equally weight the six imperfect conditions. The error-intervention endpoint averages the three full-error conditions against one shared exact control.

{table(['Question','Observed effect','98.75% paired interval','Frozen criterion'],headline_rows)}

Intervals describe **absolute paired differences**, not confidence intervals for the displayed relative percentages. They use 4,000 bootstrap resamples of the same 800 scene IDs jointly across conditions and arms. The 98.75% level allocates a nominal 5% familywise error budget across four headline comparisons by Bonferroni; finite-sample percentile-bootstrap coverage is approximate. All intervals condition on these frozen models, fitted observers and scene distribution. Six imperfect variants are not six independently trained models, and 800 shared scenes are not 5,600 independent evaluation samples.

## What was run

The study reused three previously frozen neural transition models (seeds 3109, 3119, 3137), with **zero neural retraining**. It generated 2,400 fresh scenes from the existing 2D navigation distribution: 1,200 fit, 400 tuning and 800 evaluation scenes. Splits were by complete scene and shared across all conditions. The environment, obstacle penalty, action horizon and planner budget were held fixed.

For true transition f and frozen neural transition fₘ, the intervention was:

`f_(m,α)(x,a) = f(x,a) + α [fₘ(x,a) − f(x,a)]`, with α = 0, 0.5 or 1.

Only one exact condition was run. The half- and full-error conditions retained the same neural error direction at each fixed input; multi-step behavior need not scale linearly because planning changes visited states. Exact and interpolated dynamics are laboratory controls that require simulator access.

The cross-entropy planner used six rounds, 32 candidate ten-action sequences per round, eight elites, and shared candidate noise. The observation checkpoint was round 3. The signed prediction target was:

`V₃ = L_true(U₃) − L_true(U₆)`.

Thus V₃ > 0 means extra planning improves the executed plan, and V₃ < −0.01 defines materially harmful continuation. Negative values were retained. F1-A0 observed round 2; F1-A1's round-3 checkpoint creates two nontrivial history slots, so the two studies' effect sizes are not direct replications.

Sixteen observer inputs were each fitted under seven conditions with a fixed twelve-candidate grid: standardized linear ridge or 128 random Fourier features at two bandwidths, each with four penalties. Tuning MSE selected one pipeline per input/condition: **1,344 candidate fits and 112 selected pipelines**. No additional fitted candidates were added after outcomes. This development design was informed by F1-A0; it is not untouched confirmation. Sealed T1 seeds and P1 reserve families were not used.

## Model error and harmful continuation

{intervention}

¹ One-step coordinate RMSE is measured on the independent 2,048-input reliability bank.  
² Ten-step RMSE aggregates predicted positions over all ten steps and both coordinates on 256 disjoint random action sequences; it is not terminal-only error.  
Each harmful-continuation fraction uses all 800 evaluation scenes under an always-continue counterfactual.

The two less accurate models show substantial harm, increasing from 16.5% to 30.625% for model 3109 and from 18.0% to 31.375% for model 3119 as retained error increases from half to full. Model 3137 has no observed harm above the fixed threshold at either level; this does not prove zero risk in a larger population. Average true benefit remains positive in every condition, so extra planning can help on average while harming a sizeable minority of scenes.

The experiment supports a causal statement about changing these particular transition-error fields while fixing the simulator, scenes, objective and planner randomness. It does not identify a universal relationship between scalar neural-model error and planning harm across architectures or environments.

## The three angles and reliability

| Symbol | Measured quantity | Width |
| --- | --- | --- |
| C | Current scene context; planner means, standard deviations and incumbent actions; sorted candidate costs and incumbent cost | 100 |
| H | C snapshots from rounds 1 and 2 | 200 |
| G | Twelve geometric summaries of the cached imagined paths at round 3 | 12 |
| R | Mean/max local validation error, neighbor-support distance, and global bank RMSE | 4 |
| Raw | Positions from all 32 round-3 candidate trajectories, including their initial positions | 704 |

G describes path length, turns, distance to the goal, obstacle clearance, endpoint dispersion and path efficiency. It is Euclidean geometry of physical rollouts; there is no learned latent manifold or flow-matching component here.

R queries the 16 nearest calibration inputs along the incumbent's predicted path, using scaled positions and actions. It is an error-and-support descriptor, not a calibrated probability of failure. It never observes evaluation-plan true errors when making a decision. Its global RMSE coordinate is constant within each separately fitted condition, so that coordinate cannot explain within-condition differences; the remaining coordinates vary by case.

H and G operationalize past computational history and imagined future structure; C operationalizes a current computational observation. They do not measure subjective awareness. C already carries the planner state needed for its next-round updates. Additional transformations or history might improve a restricted finite-data readout, but they need not contain information beyond a complete current computational state and fixed world model.

### Prediction results

Pooled MSE equally weights the six imperfect variants; lower is better. All inputs receive twelve candidates. C-expanded adds 212 fixed cosine coordinates to C; CR-expanded does the same for CR. Their widths match CHG and CHGR, respectively, without a claim that width exactly matches effective capacity.

{all_arms}

The primary no-R reference is selected **on tuning scenes**, separately per condition, among C, C-expanded and C-raw. Its pooled evaluation MSE is {h['joint_history_geometry']['reference_mse']:.8f}. A selected reference can perform worse on evaluation scenes than an unselected alternative; plain C and CR-raw both have lower pooled evaluation MSE than CHG in this table. We do not choose a new reference using those evaluation results.

{predictions}

### Attribution controls

{controls}

Mismatch mappings were strict within-split derangements: no scene retained its own swapped H, G or R. The H-shuffle independently swapped the two past slots in roughly half of scenes. Some chronology may remain inferable from values, so this is not a general test of time-order information.

The descriptive CHG advantage over CG has a positive interval. The comparisons against CH, mismatched H, mismatched G and the stronger reference do not jointly support the attribution criteria. Accordingly, the weaker CG comparison does not establish that both history and geometry independently contribute useful information. Correctly paired R did not pass its mismatch comparison either.

## Stop/continue decisions and actual computation cost

At round 3, a controller continues to round 6 if and only if its predicted V₃ exceeds 0.01. Its charged objective is true executed-plan cost plus 0.01 for continuation. The common three-round prefix is omitted from this proxy objective. The continuation charge represents three additional planning rounds (960 transition predictions), but excludes feature extraction and readout work; it is not a full runtime cost or an externally supplied economic utility.

The current-based controller was selected by tuning charged objective among C, CR, C-expanded, CR-expanded, C-raw and CR-raw, using each arm's MSE-selected predictor. It may differ from the reference chosen for prediction MSE.

{controller}

Across imperfect conditions, CHGR continues on **54.67%** of cases versus **58.15%** for the reference. Its mean charged cost is **{h['charged_controller']['candidate_objective']:.6f}**, versus **{h['charged_controller']['reference_objective']:.6f}** for the reference. Fewer continuation decisions did not establish a better tradeoff: the observed relative gain is **−0.103%**, and the adjusted interval includes both improvement and deterioration.

{true_costs}

³ Harmful continuation is the fraction of **all** evaluation scenes where the controller actually continues and V₃ < −0.01; it is not the conditional harm rate among continued cases. Pooled uncharged true cost is 0.145068 for CHGR versus 0.144565 for the reference. Pooled executed harmful-continuation fractions are 8.958% and 8.833%, respectively.

The following charged comparators retain the fixed 0.01 charge. The hindsight oracle uses the true counterfactual benefit, is unavailable at decision time, and is only a lower bound for this two-choice task.

{comparators}

### Prespecified charge sensitivities

{sensitivity}

Only the fixed threshold and matching charge vary here. Reference identities remain those selected on tuning data at charge 0.01. These descriptive 95% intervals all include zero and cannot replace the failed primary endpoint.

### Runtime benchmark

{timing}

These are medians of three repetition means over the **first 24 evaluation scenes**, using one CPU thread and rotated method order after warm-up. Timing includes actual online planning, collection of required history, geometry/reliability extraction, feature transforms and readout. It excludes parameter loading, one-time calibration-bank construction, outcome scoring and serialization. Online decisions and executed plans exactly match the saved offline counterfactual choices on this panel.

CHGR was slower than its current-based reference in all seven measured conditions. It was faster than always continuing in two of six imperfect conditions, so the frozen all-condition runtime-saving criterion also failed. The small timing panel is descriptive and hardware-specific; it does not justify a universal latency claim.

The one-time bank uses 2,048 shared true transitions and 2,048 model predictions per condition, plus nearest-neighbor indexing. Recorded per-condition prediction/index setup times ranged from {min(d['calibration_seconds'] for d in R['world_diagnostics'])*1000:.2f} to {max(d['calibration_seconds'] for d in R['world_diagnostics'])*1000:.2f} ms in this simulator. Those timings exclude acquiring the shared observations and are not a real-world data-collection cost.

## Audit and reproducibility

The post-run audit reconstructed **{A['candidate_tuning_scores_reconstructed']:,} candidate tuning scores** and **{A['selected_pipelines_reconstructed']} selected pipelines** from saved coefficients, without any refitting. Tuning scores, selected evaluation predictions, fit-only normalization and prefix feature replay agreed exactly in this environment. Independent true-dynamics/cost replay over every saved plan differed by at most {A['max_absolute_errors']['true_outcome']:.3g}; independent headline interval reconstruction differed by at most {A['headline_interval_max_abs_error']:.3g}.

The first 100 scenes per condition were replayed only through round 3 to verify that C, H, G, R and raw inputs did not require later planning. All split boundaries, mismatch mappings, reference selections, source/checkpoint hashes and recorded online replay checks passed. Audit reconstruction shares some feature/basis functions with the original implementation; it is not a fully independent second implementation of every operation.

Selected penalties: {', '.join(f'{k:g}: {v}' for k,v in sorted(penalties.items()))} pipelines. Thus {penalties[100.0]} of 112 selected pipelines reached the upper penalty boundary. Selected bases were {bases[0]} linear, {bases[1]} Fourier at bandwidth 0.5 and {bases[2]} Fourier at bandwidth 2. A future grid extension would constitute a new study; none was added to this one.

The experiment manifest was written at `{M['started_utc']}` and completion was recorded at `{R['finished_utc']}`. The source, configuration, protocol and prior checkpoint hashes remain unchanged. A residual incomplete `.tmp` write was found during packaging; it is not read by the study or audit and is excluded from the release. The completed numerical archives used for these findings passed reconstruction and archive-integrity checks. No claim is made that the residual temporary file is valid data.

`F1_A1_Study.zip` includes the frozen protocol, source, prior checkpoints with provenance, all observer coefficients and tuning scores, scene splits, counterfactual plans, evaluation predictions, audit, runtime repetitions, this report, the figure, dependency versions and SHA-256 checksums. `README.md` documents audit-only replay and how to reproduce a fresh numerical run without overwriting this record.

## What this changes, and the next bounded question

The project now has a controlled example of **model error changing the value of additional computation**. It does not yet have evidence that combining computational history, current-state evaluation and future geometry produces a better metacomputation controller in this task. F1-A0's negative geometry result remains on record, and this modified development study does not replace it. No incompatible endpoints from the other research branches are pooled into this result.

The next proposed study, **F1-A2**, should separate signal adequacy from decoder limitations before adding another architecture. Its central question would be: *can an estimate of decision-relevant model bias predict when further optimization will worsen the real plan?* General one-step error magnitude may be too weak or misaligned with the cost change; that is a hypothesis arising from this result, not an established explanation.

A useful positive control would expose simulator-derived cost-bias information on a fixed, already available candidate set, clearly labeled as privileged diagnostic information. A deployable counterpart would estimate the same quantity using only fit/calibration data. An oracle that uses U₆ or its true outcome would instead be a hindsight ceiling and could not count as an online feature. Compare those controls with the same strong current-based reference under a newly frozen protocol and fresh scenes. This would test whether a useful signal exists before expanding the model class. F1-A2 is proposed here; it has not been executed or selected after these outcomes.

## Related work and limits

Model exploitation and the tradeoff between model use and model error are established concerns in [Janner et al., *When to Trust Your Model: Model-Based Policy Optimization*](https://arxiv.org/abs/1906.08253). [Yu et al., *MOPO: Model-based Offline Policy Optimization*](https://arxiv.org/abs/2005.13239) studies uncertainty-penalized rewards in offline model-based policy optimization. They motivate the distinction between model accuracy and reliable decisions; F1-A1 is not an implementation or replication of either method.

The evidence is limited to one fully observed deterministic toy world, three fixed neural checkpoints, fresh samples from a development-exposed distribution, one planner and one observation checkpoint. There is no transformer, attention intervention, flow-matching training, partial-observability experiment, consciousness measurement or claim about subjective present-moment awareness. The organizing idea remains testable, while the current tested combination has not demonstrated the intended practical advantage.
'''
    (ROOT/'F1_A1_Report.md').write_text(report,encoding='utf-8')


if __name__ == '__main__':
    draw_figure()
    write_report()
    print('Created F1_A1_Results.png and F1_A1_Report.md from audited run/result.json.')
