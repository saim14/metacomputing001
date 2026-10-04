"""Presentation of the frozen, audited F1-A2 results. No new fits."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
R=json.loads((ROOT/'run/result.json').read_text())
A=json.loads((ROOT/'run/audit.json').read_text())
M=json.loads((ROOT/'run/manifest.json').read_text())
assert A['passed']
LABEL={'c':'C','cr':'CR','ce':'C-expanded','cre':'CR-expanded','craw':'C-raw','crraw':'CR-raw',
       'x':'X = CHGR','cp':'CP (privileged)','cb':'CB (estimated)','xp':'XP','xb':'XB',
       'cmp':'C-mismatched-P','cmb':'C-mismatched-B','cpi':'CP-incumbent','cbi':'CB-incumbent'}


def name(row):
    return 'Exact' if row['seed'] is None else f"{row['seed']} / {row['alpha']:g}"


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def interval(v,digits=7):return f'[{v[0]:+.{digits}f}, {v[1]:+.{digits}f}]'


def figure():
    teal,blue,orange,ink,muted='#087F8C','#4F65A5','#B9652C','#23333B','#65757D'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':ink,'axes.labelcolor':ink,
        'xtick.color':ink,'ytick.color':ink,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#CCD6DA','axes.titleweight':'bold'})
    fig,axes=plt.subplots(2,2,figsize=(13.8,9.6))
    fig.subplots_adjust(left=.12,right=.96,top=.855,bottom=.16,wspace=.36,hspace=.58)
    fig.suptitle('F1-A2  |  Cost-bias information improves prediction',x=.08,y=.974,ha='left',fontsize=20,fontweight='bold')
    fig.text(.08,.925,'The accessible estimate passed the prediction criterion; practical control and runtime gains remain unestablished.',fontsize=10.5,color=muted)
    ax=axes[0,0]
    for y,key,color in [(1,'privileged_signal',blue),(0,'estimated_signal',teal)]:
        h=R['headline'][key];p=h['mean']*1000;lo,hi=np.array(h['ci_adjusted'])*1000
        ax.errorbar(p,y,xerr=[[p-lo],[hi-p]],fmt='o',color=color,linewidth=2,markersize=8,capsize=5)
        ax.text(p,y+.21,f"{h['relative_reduction']*100:+.2f}% relative",ha='center',color=color,fontsize=10)
    threshold=R['headline']['estimated_signal']['reference_mse']*.05*1000
    ax.axvline(0,color=muted,linewidth=1);ax.axvline(threshold,color=orange,linestyle=':',linewidth=1.5)
    ax.set(title='A  Both bias signals passed',xlabel='Reference MSE − candidate MSE (× 10⁻³)',
           yticks=[0,1],yticklabels=['Estimated B','Privileged P'],ylim=(-.6,1.6),xlim=(-.45,10.5))
    ax.text(.98,.06,'98.33% paired intervals\nDotted line: 5% point-gain requirement',transform=ax.transAxes,ha='right',fontsize=8.5,color=muted)
    ax.grid(axis='x',alpha=.17)
    ax=axes[0,1];h=R['headline']['charged_controller'];p=h['mean']*1000;lo,hi=np.array(h['ci_adjusted'])*1000
    ax.errorbar(p,0,xerr=[[p-lo],[hi-p]],fmt='o',color=teal,linewidth=2,markersize=8,capsize=6)
    ax.axvline(0,color=muted,linewidth=1);ax.scatter(h['reference_objective']*.01*1000,0,marker='|',s=400,color=orange)
    ax.text(p,.27,f"{h['relative_reduction']*100:+.2f}% relative",ha='center',fontsize=11,color=teal)
    ax.text(.5,.12,'Adjusted interval includes zero\nExtra 320 model calls are charged',transform=ax.transAxes,ha='center',fontsize=9,color=muted)
    ax.set(title='B  Controller criterion did not pass',xlabel='Reference cost − CB cost (× 10⁻³)',yticks=[],xlim=(-1,8),ylim=(-.65,.65))
    ax.grid(axis='x',alpha=.17)
    ax=axes[1,0]
    for y,key,color in [(1,'integration_privileged',blue),(0,'integration_estimated',teal)]:
        h=R['descriptive_comparisons'][key];p=h['mean']*1000;lo,hi=np.array(h['ci95'])*1000
        ax.errorbar(p,y,xerr=[[p-lo],[hi-p]],fmt='o',color=color,linewidth=2,markersize=7,capsize=5)
        ax.text(p,y+.21,f"{h['relative_reduction']*100:+.2f}% relative",ha='center',fontsize=9,color=color)
    ax.axvline(0,color=muted,linewidth=1)
    ax.set(title='C  Combined features added no established gain',xlabel='Simpler MSE − combined MSE (× 10⁻³)',
           yticks=[0,1],yticklabels=['CB → XB','CP → XP'],xlim=(-2.6,.45),ylim=(-.6,1.6))
    ax.text(.03,.05,'X adds history, geometry and R to C\n95% descriptive intervals',transform=ax.transAxes,fontsize=8.5,color=muted)
    ax.grid(axis='x',alpha=.17)
    ax=axes[1,1];runtime=R['runtime']['rows'][1:]
    ratios=[r['ms_per_case']['cb']/r['ms_per_case'][R['controller_references'][i+1]] for i,r in enumerate(runtime)]
    y=np.arange(6);ax.barh(y,ratios,color=teal,height=.57);ax.axvline(1,color=orange,linestyle=':',linewidth=1.5)
    for yi,value in zip(y,ratios):ax.text(value+.05,yi,f'{value:.2f}×',va='center',fontsize=9)
    ax.set(title='D  Estimation was slower than the reference',xlabel='CB runtime / reference runtime',
           yticks=y,yticklabels=[name(r) for r in R['per_condition'][1:]],xlim=(0,4));ax.invert_yaxis();ax.grid(axis='x',alpha=.17)
    fig.text(.08,.043,'A–C: 800 paired evaluation scenes, six imperfect variants of three fixed neural models. Positive differences favor the candidate.\n'
             'D: first 24 evaluation scenes; medians of three repetition means, one CPU thread. The exact condition is reported separately.',
             fontsize=9,color=muted,linespacing=1.6)
    fig.savefig(ROOT/'F1_A2_Results.png',dpi=180,facecolor='white');plt.close(fig)


def report():
    h=R['headline'];rows=R['per_condition'];desc=R['descriptive_comparisons'];imp=rows[1:]
    headlines=table(['Endpoint','Relative improvement','Absolute improvement','98.33% paired interval','Criterion'],
        [[title,f"{h[key]['relative_reduction']*100:+.3f}%",f"{h[key]['mean']:+.7f}",interval(h[key]['ci_adjusted']),
          'PASS' if R['verdicts'][key] else 'FAIL'] for key,title in [('privileged_signal','Privileged P prediction'),('estimated_signal','Estimated B prediction'),('charged_controller','Estimated B controller')]])
    allarms=table(['Input','Width','Pooled evaluation MSE'],
        [[LABEL[a],R['selected_observers'][0][a]['input_dimensions'],f"{R['pooled_arm_mse'][a]:.8f}"] for a in R['arms']])
    perpred=table(['Model / α','Observed reference','Reference MSE','CP MSE','CB MSE','XB MSE'],
        [[name(r),LABEL[r['reference']],f"{r['mse'][r['reference']]:.8f}",f"{r['mse']['cp']:.8f}",f"{r['mse']['cb']:.8f}",f"{r['mse']['xb']:.8f}"] for r in rows])
    dl={'paired_p':'Correct P vs mismatched P','paired_b':'Correct B vs mismatched B',
        'set_beyond_incumbent_p':'Full P vs incumbent P only','set_beyond_incumbent_b':'Full B vs incumbent B only',
        'estimated_vs_magnitude':'CB vs CR','integration_privileged':'XP vs CP','integration_estimated':'XB vs CB',
        'added_bias_to_x':'XB vs X','privileged_estimated_gap':'CP vs CB'}
    controls=table(['Candidate vs comparator','Relative MSE reduction','Absolute reduction','Descriptive 95% interval'],
        [[dl[k],f"{v['relative_reduction']*100:+.3f}%",f"{v['mean']:+.7f}",interval(v['ci95'])] for k,v in desc.items()])
    quality=table(['Model / α','Uncorrected rollout RMSE','Corrected rollout RMSE','Zero-bias cost MSE','Estimated-bias cost MSE'],
        [[name(d),f"{d['rollout_rmse']:.6f}",f"{d['corrected_rollout_rmse']:.6f}",f"{b['zero_bias_mse']:.7f}",f"{b['all_candidate_bias_mse']:.7f}"]
         for d,b in zip(R['model_diagnostics'],R['bias_estimation'])])
    controller=table(['Model / α','Controller reference','Reference charged cost','CB charged cost','Reference continue','CB continue'],
        [[name(r),LABEL[r['controller_reference']],f"{r['controllers'][r['controller_reference']]['charged_cost']:.6f}",
          f"{r['controllers']['cb']['charged_cost']:.6f}",f"{r['controllers'][r['controller_reference']]['continue_fraction']*100:.2f}%",
          f"{r['controllers']['cb']['continue_fraction']*100:.2f}%"] for r in rows])
    outcomes=table(['Model / α','Reference true cost','CB true cost','Reference harmful continuation','CB harmful continuation','All-continuation harm'],
        [[name(r),f"{r['controllers'][r['controller_reference']]['true_cost']:.6f}",f"{r['controllers']['cb']['true_cost']:.6f}",
          f"{r['controllers'][r['controller_reference']]['harmful_continuation_fraction']*100:.3f}%",f"{r['controllers']['cb']['harmful_continuation_fraction']*100:.3f}%",
          f"{r['harmful_fraction']*100:.3f}%"] for r in rows])
    bounds=table(['Model / α','Always stop','Always continue','Hindsight oracle'],
        [[name(r),f"{r['always_stop']:.6f}",f"{r['always_continue']:.6f}",f"{r['hindsight_oracle']:.6f}"] for r in rows])
    sensitivity=table(['Continuation charge','Reference minus CB cost','Descriptive 95% interval','CB continue','Reference continue'],
        [[k,f"{v['difference']['mean']:+.7f}",interval(v['difference']['ci95']),f"{v['candidate_continue_fraction']*100:.2f}%",f"{v['reference_continue_fraction']*100:.2f}%"] for k,v in R['charge_sensitivity'].items()])
    timing=table(['Model / α','Reference ms','CB ms','XB ms','Always-stop ms','Always-continue ms'],
        [[name(rows[i]),f"{r['ms_per_case'][R['controller_references'][i]]:.3f}",f"{r['ms_per_case']['cb']:.3f}",f"{r['ms_per_case']['xb']:.3f}",
          f"{r['ms_per_case']['always_stop']:.3f}",f"{r['ms_per_case']['always_continue']:.3f}"] for i,r in enumerate(R['runtime']['rows'])])
    penalties=Counter(v['alpha'] for m in R['selected_observers'] for v in m.values())
    bases=Counter(v['basis_index'] for m in R['selected_observers'] for v in m.values())
    text=f'''# F1-A2 — Decision-specific model bias and the value of computation

Research 002 · Saim 13.02 · 18 September 2026 · Version 1.0  
Status: completed and audited development study. Two prediction criteria passed; the practical controller criterion did not.

## Finding

**Cost-bias information on the already available candidate plans helped predict whether further planning would improve the real outcome.** A privileged diagnostic using true candidate costs reduced evaluation MSE by **30.25%**, and an accessible estimate based on a separate calibration bank reduced it by **6.89%**, against a tuning-selected reference with access to the previous current/history/geometry/reliability features. Both cleared the frozen 5% prediction threshold and adjusted interval requirement.

The accessible controller's charged objective improved by **2.34%** at the point estimate, but its **98.33% interval included zero**, so the frozen controller criterion failed. Its runtime was **2.08–3.46 times** its reference in the six imperfect conditions. The result supports a useful predictive signal in this setting; it does not yet establish a more efficient metacomputation controller.

Adding the previous history, geometry and magnitude-reliability features to the new estimated signal did not improve the point estimate: XB MSE was **0.97% higher** than CB. The original three-angle synthesis remains unvalidated as a practical improvement. Prediction from decision-specific model bias is the supported advance in this study.

![F1-A2 scientific summary](F1_A2_Results.png)

## The mathematical connection

For any action sequence U, define model cost bias as

`b(U) = L_true(U) − L_model(U)`.

The true benefit of continuing planning has the exact decomposition

`V3 = [L_model(U3) − L_model(U6)] − [b(U6) − b(U3)]`.

The first bracket is improvement according to the planner's model. The second is the increase in that model's cost bias. If the bias increases more than the predicted improvement, further optimization makes the real plan worse. This algebra is an identity, not a new theorem or an additional experimental finding.

F1-A2's P and B summaries inspect bias on round-3 candidates as a possible clue to the later bias change. They do **not** observe b(U6) when deciding. G describes physical path geometry; P/B describe cost distortions across candidates. These are different observables. The experiment does not establish that a geometric manifold or subjective awareness is needed.

## Frozen endpoints

Positive differences favor the candidate. Prediction effects are reference MSE minus candidate MSE; the controller effect is reference charged cost minus CB charged cost.

{headlines}

The reference MSE was **{h['estimated_signal']['reference_mse']:.8f}**, CP MSE **{h['privileged_signal']['candidate_mse']:.8f}**, and CB MSE **{h['estimated_signal']['candidate_mse']:.8f}**. Both prediction gates required at least 5% relative reduction and a positive adjusted lower bound. The controller gate required at least 1% relative reduction and a positive adjusted lower bound. Its lower bound was **−0.0000692**, so it failed despite a favorable point estimate. The unadjusted 95% controller interval is positive; it cannot replace the prespecified adjusted interval.

The intervals use 4,000 resamples of **800 whole scene IDs**, jointly across all variants and arms. Three headline comparisons use 98.333333% two-sided percentile intervals, a nominal Bonferroni allocation of a 5% familywise budget. Displayed relative percentages are point estimates; the intervals describe absolute paired differences. Bootstrap coverage is approximate. Results condition on three fixed neural checkpoints, fitted readouts and a development-exposed scene distribution. Six imperfect variants are not six independently trained models.

## What was tested

The study used 2,400 fresh scenes, split into 1,200 fit, 400 tuning and 800 evaluation scenes. It retained F1-A1's true 2D dynamics, goal/obstacle/action cost, ten-action horizon, six-round cross-entropy planner, 32 candidates and eight elites. All variants of a scene share the split and planning random draws. The target remains signed V3 = true cost after round 3 minus true cost after round 6.

Seven world-model conditions were retained: one exact simulator and half/full neural error from frozen checkpoints 3109, 3119 and 3137. The primary pool equally weights six imperfect conditions. The exact simulator is diagnostic. There was **zero neural retraining**, and no additional fitted observer after the fixed **1,260 candidate fits and 105 selected pipelines**. F1-A1 informed this development design; F1-A0/F1-A1 results remain intact. No T1 sealed seed or P1 reserve family was used.

### P: privileged diagnostic

At the checkpoint, P uses true-simulator replay of the same 32 candidate action sequences already evaluated by the planner. Twelve summaries capture signed cost bias, its distribution and relation to model cost, elite/rest differences, ranking discordance, and incumbent regret within that candidate set. P uses 320 extra true transition evaluations and is **unavailable to a deployed controller**. It includes true information about the current incumbent cost, which is a component of V3. It never uses later candidates or U6.

### B: accessible estimate

A separate bank of 2,048 state/action observations provides signed one-step transition residuals. At each query, a fixed 16-neighbor weighted affine regression estimates the residual; the model plus that estimated correction rolls out the same 32 cached action sequences. Applying the task cost produces estimated candidate biases, summarized identically to P. No evaluation observation updates the bank. B requires no test-time true transition query beyond the artificial exact/half-error controls embedded in the laboratory model definition. On full neural models, its computation was audited with the true simulator blocked.

B uses signed errors propagated through the cost. R, the earlier reliability descriptor, instead summarizes local error magnitude and calibration support. Both use the same new bank, making the comparison informative about how that data is used. The correction is used only to construct B; it does **not** replace the planner's transition model or choose different U3/U6 plans. This is not value-aware world-model retraining.

## Prediction results and controls

C is the 100-coordinate current planner observation, H contains two past snapshots, G contains twelve physical-path summaries, and R contains four magnitude/support summaries. X = [C,H,G,R]. CP/CB append twelve privileged/estimated bias coordinates to C. XP/XB append them to X. C-expanded adds twelve fixed cosine features and matches CP/CB input width. All fifteen arms receive the same twelve-observer grid, with normalization fit only on fit scenes.

The strong observed reference is selected on tuning MSE separately per condition among C, CR, C-expanded, CR-expanded, C-raw, CR-raw and X. It cannot access P/B. Controller references are selected separately using tuning charged objective. Reference selection is not changed by evaluation results.

{allarms}

Per-condition prediction results retain the exact control and model heterogeneity:

{perpred}

The gains are concentrated in the two less accurate neural checkpoints. The good checkpoint 3137 does not show a CB improvement in either half/full condition. An aggregate predictive gain is not a universal gain for every model.

### Pairing and incumbent-only controls

{controls}

Both correctly paired P and B beat their within-split mismatched counterparts with positive descriptive intervals. Their full candidate-set summaries also beat their incumbent-only controls: **23.91%** relative for P and **5.13%** for B. Thus the gain is not accounted for by merely exposing the incumbent's cost bias within this tested readout family. These controls support useful candidate-set information; they do not identify a unique causal feature or prove that all twelve coordinates matter.

Adding H/G/R to CP increased MSE by 7.15%, with a negative descriptive 95% improvement interval. Adding them to CB increased MSE by 0.97%, with an interval crossing zero. Adding B to the old X representation did help (6.60% descriptive reduction), but the simpler CB still had the better point estimate. These secondary comparisons do not validate a necessary role for history or physical-path geometry and cannot replace any headline endpoint.

## How accurate was the bias estimate?

{quality}

Rollout RMSE aggregates positions across all ten steps and both coordinates on a disjoint 256-sequence diagnostic set. Cost-bias MSE uses all 32 cached candidates in each of the 800 evaluation scenes; its zero-bias baseline simply trusts the original model cost. These descriptive errors are not independent candidate-level hypothesis tests. The local correction improves both diagnostics for every imperfect variant, while substantial error remains for checkpoints 3109 and 3119. The larger CP gain therefore leaves room to improve B, but does not uniquely identify whether estimation, summaries or readouts account for the entire gap.

All twelve descriptor-wise RMSE values, incumbent-only bias errors and ranking-discordance errors are retained in `run/result.json`. Exact-condition cost biases and estimation errors are zero, as expected from its zero residual bank.

## Decision outcomes and computation costs

The per-case policy continues iff predicted V3 > 0.01. The primary objective is

`J = true executed-plan cost + 0.01 × continue + feature model-call charge`.

CB/XB add 320 model transitions to estimate B, versus 960 transitions in three additional planning rounds. Their fixed feature charge is 0.00333333 per case. The reference uses cached observations with no extra transition calls. This proxy excludes local regressions, neighbor searches, geometry and readout overhead; actual runtime below includes them. The proxy differs from F1-A1's convention and must not be compared as an identical economic measure across studies.

{controller}

Pooled charged cost was **{h['charged_controller']['candidate_objective']:.6f} for CB** versus **{h['charged_controller']['reference_objective']:.6f} for the reference**. CB continued on 47.27% of cases versus 62.15%. Its true executed-plan cost averaged 0.134580 versus 0.139849. Those favorable point estimates do not override the failed adjusted controller gate.

{outcomes}

Harmful continuation means the fraction of **all evaluation scenes** where a method continues and V3 < −0.01, not the conditional fraction among continued cases. Pooled rates were 6.958% for CB and 12.146% for the reference. These rates are descriptive; a reduction in harm does not by itself account for foregone beneficial computation or estimation overhead.

Always-stop, always-continue and hindsight bounds at charge 0.01:

{bounds}

The hindsight oracle observes true V3 and is not implementable at the checkpoint. Privileged CP/XP controller rows are saved only as diagnostics and their displayed charges omit unavailable true-simulator acquisition cost; they are never used to support deployable performance.

### Prespecified charge sensitivities

{sensitivity}

Both the continuation threshold and its charge vary together, and B's model-call charge scales proportionally. Reference identities remain fixed from primary-charge tuning. The results depend on the price assigned to computation: lower charges show favorable descriptive intervals, while the 0.05 charge is unfavorable. These sensitivities do not rescue the failed primary endpoint or establish an externally meaningful optimal charge.

### Actual runtime

{timing}

The benchmark uses the first 24 evaluation scene IDs, one warm-up and three timed repetitions with rotated method order on one CPU thread. Entries are medians of repetition means, in milliseconds per case. They include conditional planning, required history and features, corrected rollouts, neighbor search, local solves and readout; they exclude loading, serialization, true outcome scoring and one-time bank acquisition/index setup. Recorded decisions and plans match offline counterfactual choices exactly.

CB was 2.08–3.46× slower than its observed reference across imperfect conditions and slower than always continuing in every condition. The runtime-saving gate failed. This implementation spends substantial computation to decide whether to spend more. The small benchmark is hardware-specific; the result supports a measured limitation, not a universal latency ratio.

Calibration requires 2,048 shared true transitions plus model residual evaluation and indexing for each condition. Its data acquisition cost is not part of online timing and would require a source of trustworthy observations in a real application.

## Audit and reproducibility

The audit reconstructed all **1,260 tuning scores and 105 selected predictors without refitting**. Tuning scores, selected predictions, fit-only normalizers and 100 prefix-feature replays per condition agreed exactly. An independent physics/cost implementation differed from saved true plan costs by at most **{A['max_absolute_errors']['true_outcome']:.3g}** and candidate costs by **{A['max_absolute_errors']['true_candidate_cost']:.3g}**. Independent headline reconstruction differed by at most **{A['max_absolute_errors']['headline']:.3g}**.

All splits, strict mismatch derangements, tuning reference choices, frozen hashes and online replay checks passed. For all three full neural models, the true simulator was replaced with a failing stub while reconstructing B; the reconstruction succeeded. This verifies that the accessible bias path uses the saved calibration bank and model rather than true simulator calls. Feature/readout replay shares some implementation functions with the experiment; it is not a wholly independent implementation.

Selected ridge penalties were {', '.join(f'{k:g}: {v}' for k,v in sorted(penalties.items()))} pipelines. Selected bases were {bases[0]} linear, {bases[1]} Fourier at bandwidth 0.5, and {bases[2]} Fourier at bandwidth 2. There was no post-result grid extension. Source/configuration/protocol/checkpoint hashes were captured at `{M['started_utc']}`; the run completed at `{R['finished_utc']}` and the hashes remained unchanged.

The release includes the frozen source and protocol, inherited neural checkpoints and provenance, calibration bank, split IDs, current candidate actions and true/model/corrected costs, U3/U6 plans, all candidate coefficients, selected predictions, runtime repetitions, audit, report, figure, dependency versions and SHA-256 checksums. `README.md` documents audit-only replay and reproduction in a separate directory. No future F1-A3 result is included.

## Implication for metacomputation and the next question

F1-A1 showed that the model can make extra planning harmful; F1-A2 now shows that **decision-specific model bias can help predict the value of further computation** in the same task family. The accessible estimate improved prediction even against controls that included the earlier H/G/R representation. The result does not show that every form of foresight is useful or that combining all three research angles is better.

The next bounded question is whether we can preserve this predictive gain with a cheaper bias estimate. A proposed **F1-A3** could compare the current estimator with a fixed smaller candidate subset or a compact approximation trained on fit/calibration data, using fresh scenes and an explicit online cost budget. Any learned approximation needs separation between its training and downstream selection to avoid leakage. The estimator should be chosen before outcomes; no favorable subset from F1-A2 may become a retrospective success criterion. This is a proposed direction, not an executed study.

## Related work and limits

Task utility rather than accurate state reconstruction alone is an established motivation in [Grimm et al., *The Value Equivalence Principle for Model-Based Reinforcement Learning*](https://arxiv.org/abs/2011.03506). [Voelcker et al., *Value Gradient weighted Model-Based Reinforcement Learning*](https://arxiv.org/abs/2204.01464) studies the mismatch between dynamics prediction losses and decision utility. F1-A2 does not implement their objectives, establish novelty, or validate their broader claims; it tests cost-bias descriptors for one project's metacomputation question.

The evidence remains limited to one fully observed deterministic toy environment, three frozen neural models, one planner/checkpoint, a development-exposed distribution and the stated observer class. Fresh scene sampling does not test new environments or independently retrained models. P is privileged; exact/half-error models require simulator knowledge. No claim concerns consciousness, subjective present-moment awareness, transformers, recursive attention, flow matching or a learned geometric manifold.
'''
    (ROOT/'F1_A2_Report.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    figure();report();print('Created F1_A2_Results.png and F1_A2_Report.md.')
