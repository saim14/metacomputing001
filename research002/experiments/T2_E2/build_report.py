"""Produce descriptive tables and a report from the closed T2-E2 results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from study_common import ROOT,SEEDS,STEPS,ARMS,dump,sha


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def pct(x):return f'{100*x:+.2f}%'
def interval(values):return '['+', '.join(pct(x) for x in values)+']'


def build():
    spec=json.loads((ROOT/'STUDY_PROTOCOL.json').read_text())
    freeze=json.loads((ROOT/'MAIN_FREEZE.json').read_text())
    dev=json.loads((ROOT/'development/development_result.json').read_text())
    devpanels=json.loads((ROOT/'development/panel_comparisons.json').read_text())
    result=json.loads((ROOT/'results/study_result.json').read_text())
    fit=json.loads((ROOT/'results/fit_result.json').read_text())
    panels=json.loads((ROOT/'results/panel_comparisons.json').read_text())
    seeds=json.loads((ROOT/'results/seed_comparisons.json').read_text())
    diagnostics=json.loads((ROOT/'results/test_diagnostics.json').read_text())
    metrics=pd.read_csv(ROOT/'results/probe_metrics.csv')
    required=[r for r in panels if r['baseline'] in ['current_rff_mlp','current_reference']]
    primary=next(r for r in required if r['baseline']=='current_rff_mlp')
    reference=next(r for r in required if r['baseline']=='current_reference')
    code=result['decision']['classification']
    outcome={'advantage_worth_independent_replication':'The prespecified exploratory screen passed against both required references. This is a candidate advantage for independent replication, not confirmation.',
        'useful_gain_excluded_for_this_fixed_panel_and_predictor':'The screen failed, and both conditional interval upper bounds fell below the prespecified 5% useful-gain threshold. This excludes that mean gain only within the stated conditional analysis of these fixed fitted predictors and models.',
        'inconclusive':'The prespecified exploratory screen failed, while the required intervals did not jointly exclude the 5% useful-gain threshold. The result remains inconclusive.',
        'ineligible':'The main panel did not satisfy the prespecified all-six-model eligibility requirement. It cannot support the primary panel decision.'}[code]
    results_table=table(['Required comparison: history MLP vs','Mean relative MSE gain','Conditional 95% interval','Mean ΔR²','Mean MAE gain (nats)','Positive seeds','Screen'],
        [[r['baseline'],pct(r['mean_relative_mse_gain']),interval(r['conditional_gain_ci95']),f"{r['mean_delta_r2']:+.6f}",f"{r['mean_mae_advantage']:+.6f}",f"{r['positive_seeds']}/6",'Pass' if r['screen_passed'] else 'Fail'] for r in required])
    seed_table=table(['Seed','Eligible steps','Final test accuracy','Gain vs matched current MLP','Gain vs selected current reference'],
        [[s,', '.join(str(r['step']) for r in diagnostics if r['seed']==s and r['eligible']),
          f"{100*next(r['test_final_accuracy'] for r in diagnostics if r['seed']==s):.2f}%",
          *[pct(next(r['relative_mse_gain'] for r in seeds if r['seed']==s and r['baseline']==b)) for b in ['current_rff_mlp','current_reference']]] for s in SEEDS])
    dev_table=table(['Known oracle MSE reduction','Measured gain vs matched current MLP','Measured gain vs selected current reference','Complete screen'],
        [[f'{2*d:.0%}',pct(next(r['mean_relative_mse_gain'] for r in devpanels if r['case']==d and r['baseline']=='current_rff_mlp')),
          pct(next(r['mean_relative_mse_gain'] for r in devpanels if r['case']==d and r['baseline']=='current_reference')),
          'Pass' if all(r['screen_passed'] for r in devpanels if r['case']==d and r['baseline'] in ['current_rff_mlp','current_reference']) else 'Fail'] for d in [0.,.025,.1]])
    aggregated=metrics[metrics.eligible].groupby(['seed','arm'])[['mse','mae','r2']].mean().groupby('arm').mean()
    score_table=table(['Observer','Mean R²','Mean MSE (nats²)','Mean MAE (nats)'],
        [[a,f'{aggregated.loc[a,"r2"]:.6f}',f'{aggregated.loc[a,"mse"]:.6f}',f'{aggregated.loc[a,"mae"]:.6f}'] for a in ARMS])
    controls=table(['Secondary comparison: history MLP vs','Mean relative MSE gain','Conditional 95% interval','Contributing models'],
        [[r['baseline'],pct(r['mean_relative_mse_gain']),interval(r['conditional_gain_ci95']),len(r['seeds'])] for r in panels if r['baseline'] in ['current_mlp','shuffled_mlp','mismatched_mlp','history_ridge']])
    costs=[];choices=[];fitrows=[]
    for s in SEEDS:
        for step in STEPS:
            selected=json.loads((ROOT/f'results/models/seed_{s}_step_{step}/selection.json').read_text())
            choices.append({'seed':s,'step':step,'selected_current_reference':selected['current_reference']['alias_of']})
            for arm,detail in selected.items():
                fitrows.append({'seed':s,'step':step,'arm':arm,'alias':detail.get('alias_of',''),
                    **{k:detail[k] for k in ['input_dimensions','ensemble_parameters','affine_macs_per_example','extra_rff_projection_macs','history_float32_bytes_per_example','fit_seconds']}})
    for step in STEPS:
        d=388+36*(step-1)
        costs.append([step,d,2*(32*(d+2)+1),2*32*(d+1),388*36*(step-1),4*36*(step-1)])
    cost_table=table(['Step','Matched input width','Two-network parameters','Network affine MACs/example','Current RFF extra projection MACs','History bytes/example'],costs)
    pd.DataFrame(fitrows).to_csv(ROOT/'results/resource_costs.csv',index=False)
    pd.DataFrame(choices).to_csv(ROOT/'results/current_reference_choices.csv',index=False)
    dump(ROOT/'results/absolute_score_summary.json',aggregated.reset_index().to_dict(orient='records'))
    valid=sum(r['eligible'] for r in diagnostics)
    report=rf'''# T2-E2 — Bounded nonlinear observer study

**Research lead:** Saim 13.02  
**Date:** 13 September 2026  
**Status:** Completed; `{code}`.

{outcome}

History's mean relative prediction-MSE gain was **{pct(primary['mean_relative_mse_gain'])}** against the matched current-state MLP, with conditional 95% interval **{interval(primary['conditional_gain_ci95'])}**. Against the validation-selected current-only reference, it was **{pct(reference['mean_relative_mse_gain'])}**, interval **{interval(reference['conditional_gain_ci95'])}**. Positive values favor history; negative values mean greater error with history.

## Purpose and scope

[T2-E1](https://drive.google.com/file/d/17M_RRAgFbWj3SeYMuby9-IBgibgQbJre/view) found no robust history advantage with ridge predictors and fixed feature controls. [T2-C1](https://drive.google.com/file/d/1jVKM9ohkd6z6j0LdJEJhJuwY_WPe8mRq/view) showed that its acceptance rule could miss a small known synthetic signal. Neither established an actual Transformer history advantage.

T2-E2 addresses the remaining limited-predictor question with a fixed small nonlinear family, a development sensitivity check, and fresh held-out families. It retains the same remaining-loss target and all six frozen U7C models. It changes the probe family, introduces a validation-selected current-only reference, and prospectively defines a 5% relative-MSE continuation threshold. These are separately specified new-study choices, not revisions to earlier outcomes.

The underlying question concerns **usable predictive information under limited computation**. In this deterministic model, complete current state, step and weights determine future logits. A limited observer may nevertheless find some information easier to predict from history. Remaining true-label loss also depends on the label, so deterministic continuation alone cannot settle that target. No self-awareness, metacognition, causal self-monitoring, chronological mechanism, or controller benefit follows from probe accuracy alone.

## Preregistered sequence and separation

- Scientific protocol frozen: **{spec['frozen_utc']}**.
- Development calibration completed: **{dev['completed_utc']}**, in {dev['elapsed_seconds']:.2f} seconds. Its prespecified gate passed without tuning.
- Main protocol/code freeze: **{freeze['frozen_utc']}**, before main examples were generated.
- All empirical probe fitting and selection completed: **{fit['completed_utc']}**, in {fit['elapsed_seconds']:.2f} seconds, before any natural test predictions or metrics were computed.
- Held-out evaluation completed: **{result['completed_utc']}**, in {result['evaluation_seconds']:.2f} seconds.

Scientific protocol SHA-256: `{sha(ROOT/'STUDY_PROTOCOL.json')}`. `MAIN_FREEZE.json` binds the development decision and exact analysis sources. `FIT_MANIFEST.json` binds the selected model, transformation, data and tuning files before evaluation. The frozen inputs and fitting files remained unchanged.

## Development sensitivity check

The check used 3,000 new counterfactual families for each of six models, with 1,600/400/1,000 families for fitting, validation, and calibration evaluation. At step 2, real current and past features were extracted. One historical coordinate was replaced by an independent paired Gaussian variable Z; other feature columns were preserved. Independent paired Gaussian noise ε had within-family correlation 0.5, as did Z. Neither used true task labels.

With g(X) a current coordinate chosen for highest training variance and standardized using training moments, synthetic targets were

$$Y_\delta=1+\sqrt{{0.5}}g(X)+\sqrt{{\delta}}Z+\sqrt{{0.5-\delta}}\epsilon.$$

Because Z and ε are independent of current state and one another, the current oracle has population MSE 0.5 and the history oracle MSE 0.5−δ. The exact oracle relative-MSE reductions are therefore 0%, 5%, and 20% for δ=0, 0.025, and 0.1. Fitted-probe gains need not equal these oracle gains. These are MSE ratios, not the population R² increments used in T2-C1.

{dev_table}

The null failed the screen and the 20% oracle case passed against both required references, satisfying the one-panel development gate. The 5% oracle case had positive conditional lower bounds but its fitted gains remained below 5%, so it failed the useful-effect requirement. This retains a sensitivity limitation near the threshold.

This was one six-model panel at one step, not a power curve or detection-rate estimate. Its injected coordinate and simple synthetic target do not replicate all natural history geometry, nonlinear effects, heavy-tailed NLL targets, or later steps. Passing it permits the bounded main test; it does not guarantee power on the natural question.

## Empirical design

The main study used 36,000 new examples: 3,000 counterfactual families of two examples per seed, split 1,600/400/1,000 into probe training, validation and test. Both members and all steps stay together. Exact family matches to each model's training/validation streams and exposed U7B/U7C evaluation suites, every T2-E1 dataset, the development calibration, and earlier-generated families across this study were excluded. The generator does not canonicalize equivalent mapping-position rearrangements; this is exact-family separation, not compositional generalization.

The target is V_l=NLL_l−NLL_6 at steps 2–5, with true-class probabilities clipped to [1e−8,1]. Negative target values are retained. True labels, future outputs and current true-label NLL do not enter predictors. Only a step whose probe-training mean V and sample SD V are both at least 0.02 nats contributes to the primary aggregate; all 24 seed-step cells remain reported. **{valid}/24 cells were eligible.**

The current observation contains all 12×32 residual values plus two probabilities, entropy and probability margin: 388 inputs. History adds the 36 CLS/prediction-summary features from each past step 1 through l−1. Current inputs are an identical prefix in every corresponding history/control arm.

MLPs use one ReLU hidden layer of 32 units and a linear output. Two fixed initializations form an ensemble. Adam trains each candidate for 80 epochs at learning rate 0.001 and batch size 128, with the same epoch permutations across arms. Validation chooses among α={{0.1,10}} and saved epochs {{20,40,80}}, using ensemble MSE; exact ties prefer larger α and then fewer epochs. Every candidate trains the full budget. Feature and target normalization use training moments only; target scale is inverted for all reported scores. No train-plus-validation refit occurs.

History, matched current-RFF, shuffled-history, and mismatched-history MLPs have identical input width, hidden width, parameter counts, update budgets, initialization count and selection grids. The RFF arm appends fixed current-only cosine features. Equal parameter counts do not imply identical effective capacity or total inference cost. A simpler current-only MLP and three ridge references (current, current-RFF, history) are also fitted. Ridge α comes from {{0.01,0.1,1,10,100}}, chosen by validation MSE.

The necessary current-only reference is selected independently per seed-step by validation MSE from current MLP, current-RFF MLP, current ridge and current-RFF ridge. It has a larger family-selection budget and is a reference safeguard, not a claim of search-budget matching. Shuffled history at step 2 aliases ordered history exactly; it is excluded from chronology comparisons.

## Held-out results

{results_table}

For each eligible cell, relative MSE gain=(MSE_baseline−MSE_history)/MSE_baseline. Cells receive equal weight within seed, then seeds receive equal weight. This mean of ratios differs from a ratio of pooled errors. The same weighting applies to ΔR² and MAE differences.

Success required, against **both** required baselines: mean relative-MSE gain at least 5%, conditional 95% lower bound above zero, positive seed-level gain in all six models, and nonnegative mean MAE gain. The 5% threshold is a declared research continuation criterion; no deployment or controller-utility threshold has been validated.

{seed_table}

All six models were retained, including the three that failed the earlier competence gates. This study cannot rehabilitate their competence or establish generalization to an untouched population of reliably trained Transformers.

### Absolute prediction and secondary controls

{score_table}

These are equal-seed averages of eligible-step scores, not pooled-example R². The contemporaneous ridge rows are the relevant reference for assessing the effect of the nonlinear observer family. Directly comparing these aggregate values to T2-E1 also changes examples and potentially step eligibility.

On these same examples and eligible steps, the current-only MLP explains about 33.5% of target variation, compared with 18.7% for current-only ridge. The nonlinear family therefore improves prediction substantially in this comparison. The history MLP explains about 33.1%; its small gain against the matched RFF-augmented MLP does not imply superiority to every current-only observer. Mean R² and mean relative-MSE ratios also weight differences in target variance and baseline error differently.

{controls}

Secondary controls are descriptive and are not multiplicity-adjusted discoveries. The shuffled comparison includes only training-eligible steps with at least two past slots. A raw ordered-history advantage would not by itself establish chronological-order dependence.

## Resource accounting

{cost_table}

The matched MLP rows count the selected two-network ensemble. The plain current MLP has 24,962 parameters and 24,896 network affine MACs per example. RFF projection MACs are additional to network MACs; history bytes assume float32 cached observations. These counts omit normalization, cosine/entropy operations, memory movement, and feature extraction overhead. They therefore do not establish end-to-end speedups. Per-cell fitted resources and validation-selected reference identities are saved separately.

Development used 216 individual MLP training runs and 36 ridge paths. The main study used 456 MLP runs and 72 ridge paths. The full fixed budget was 53,760 MLP epochs and 1,344,000 minibatch updates, plus 108 ridge paths with five α candidates each. No base-model update occurred. Fitting shared selection candidates does not make the six model seeds independent samples from a Transformer population.

## Verification, limitations and stopping decision

Method fixtures checked selected-ensemble validation MSE, manual network inference against scikit-learn, exact serialized prediction round trips, training-only normalization, parameter/update counts, paired temporal permutations, partition-consistent control generation, and a known 36% error-reduction bootstrap fixture. The preflight record documents one correction to an epoch-counter assertion before any research data was generated; the training loop and scientific protocol were unchanged.

All 108 development and 216 main metric rows were independently recomputed from saved predictions. The audits verified reference selection, seed and panel aggregation, conditional intervals from saved bootstrap draws, screen components, and phase decisions without refitting. A separate data audit checked all 36,000 unique families across both phases, their raw-array hashes, complementary labels, and zero overlap with the specified prior exposures or other study phases/models. All model selection preceded natural test evaluation.

The 2,000 paired-family bootstrap replicates retain both counterfactual members and common draws across steps/arms, with independent resampling within each fixed model's test set. They condition on these six models, training data, selected probes and eligibility. They do not cover model retraining, probe-selection variability, arbitrary alternative predictors, or new architectures. Heavy-tailed loss targets can also make finite-sample intervals unstable; no outcome-driven trimming or clipping was introduced beyond the frozen probability clipping.

{outcome}

**This bounded observer branch is now closed.** Earlier T1/T2-E1 findings remain unchanged. T1-U7D and T1-U8 are not launched, and reserved confirmation seeds 1103/1201/1301 remain sealed. There is no automatic confirmation advancement. Any proposed independent replication or new regime requires its own prospective scientific justification; this study authorizes no further tuning on these exposed outcomes.

## Reproduction

The ZIP includes the original frozen Transformer code/checkpoints, prior-family exclusion tokens, exact development and empirical examples, selected numeric observer weights, validation candidates, predictions, all metrics and diagnostics, conditional bootstrap draws, protocol/code/fitting hashes, logs, verification scripts, and a Colab notebook. Full intermediate Transformer tensors are regenerated to keep the archive compact. The code was executed locally; the notebook's code cells are syntax-checked, but an interactive Colab session was not executed here. See `REPRODUCTION.md` for the guarded sequence and numeric tolerances.
'''
    (ROOT/'T2_E2_Report.md').write_text(report)
    print('Report generated:',code)


if __name__=='__main__':build()
