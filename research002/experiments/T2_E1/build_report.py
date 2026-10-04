from pathlib import Path
import json,hashlib,shutil,zipfile
import pandas as pd
root=Path(__file__).resolve().parent;out=root/'results'
r=json.loads((out/'study_result.json').read_text());panel=json.loads((out/'panel_comparisons.json').read_text());seed=json.loads((out/'seed_comparisons.json').read_text());m=pd.read_csv(out/'probe_metrics.csv');d=pd.read_csv(out/'target_diagnostics.csv')
a=m[m.eligible].groupby(['seed','observer','arm'])[['test_r2','test_mae']].mean().reset_index();pooled=a.groupby(['observer','arm'])[['test_r2','test_mae']].mean()
label={'cls_current':'CLS current view','full_current':'All-token current view'}
lines=['# T2-E1 — Does CLS history improve prediction beyond current observations?','','**Research lead:** Saim 13.02  ','**Completed:** 11 September 2026  ','**Status:** Complete; the frozen exploratory success criterion was not met.','',
'This study directly tested incremental prediction of remaining loss reduction on all six frozen U7C models. Ordered CLS history did not provide a robust advantage over the current-only nonlinear control with the same fitted coefficient count. The primary all-token comparison had mean ΔR² = −0.000884, with a conditional 95% bootstrap interval [−0.039843, +0.010258]. Mean absolute error was worse with history by 0.001185 nats.','',
'An apparent mean improvement over the simplest CLS-only linear predictor was not retained against the nonlinear current-only control. This makes baseline choice an empirical concern. It does not establish that baseline flexibility is the sole mechanism behind every apparent history gain.','',
'## Why this study is a new exploratory branch','',
'The user requested the next study after the matched-example audit. T2-E1 separately freezes an observational scope using all six exposed U7C checkpoints. It does not change U7C’s failed competence gate or advance T1-U8. Seeds 503/607/709 previously passed competence gates; 811/907/1009 failed. All six remain in this study and its primary average. No model was retrained, and reserved model seeds 1103/1201/1301 remain unopened.','',
'These are auxiliary-supervised, development-exposed models. This is not confirmation on an untouched population of reliably competent Transformers. The original T1 negative results remain part of the evidence.','',
'## Frozen question, target, and observations','',
'The primary target is V_l = NLL_l − NLL_6, using the true label and probabilities clipped to [1e−8, 1]. Negative values are retained. Labels and final outputs construct the training/evaluation target only; they never enter predictor inputs. Decision-flip prevalence is reported separately as a diagnostic. No binary decision-stability probe was fitted.','',
'| View | Current input | History added |','|---|---|---|','| CLS | 32 CLS residual values + two probabilities + entropy + margin: 36 columns | The same 36 CLS observables from steps 1 through l−1 |','| All tokens | 12 × 32 residual values in fixed token-position order + the same four prediction summaries: 388 columns | The same CLS history; full-token history is not tested |','',
'All-token observations are a richer current view. They are not a claim of matched computational cost with the CLS view. The history candidate always retains the identical current-input prefix used by its corresponding baseline. Attention features are absent; no attention-specific mechanism claim is made.','',
'## Data, partitions, and eligibility','',
'- Six fixed model seeds; 3,000 fresh counterfactual families per seed, with two examples per family: 36,000 examples in total.',
'- Per seed: 1,600 training families (3,200 examples), 400 validation families (800 examples), and 1,000 test families (2,000 examples).',
'- All steps and both counterfactual members of a family share a partition. Duplicate families across partitions and exact family matches to model training/validation streams or previously exposed clean/causal suites are excluded.',
'- Equivalent rearrangements of mapping positions are not excluded. This is exact-example family separation, not a test of compositional or permutation generalization.',
'- Steps 2, 3, 4, and 5 are all reported. A step enters the primary aggregate only if its probe-training target has both mean and sample SD at least 0.02 nats. Eligibility does not use validation/test gains.',
'- Fourteen of 24 seed-step cells were eligible, and all six seeds contribute at least one step.','',
'| Seed | Training-eligible steps | Final accuracy on fresh test pairs’ members |','|---:|---|---:|']
for s,g in d.groupby('seed'):
 lines.append(f"| {s} | {', '.join(map(str,g.loc[g.eligible,'step']))} | {g.test_final_accuracy.iloc[0]*100:.2f}% |")
lines+=['','## Fixed probes and controls','',
'Each probe is linear ridge regression with an intercept. The fixed alpha grid is {0.01, 0.1, 1, 10, 100}; validation MSE selects one value, with ties choosing the larger alpha. There is no train-plus-validation refit. Final feature standardization uses training means and population SDs only. Every arm has the same training/validation examples and five-value alpha budget.','',
'| Arm | Purpose |','|---|---|','| Current linear | Simple current-only reference |','| Current dimension control | Current plus fixed linear projections; matches history’s fitted coefficient count |','| Current nonlinear control | Current plus fixed cosine random Fourier features; same fitted coefficient count as history |','| Ordered history | Current plus chronological CLS past |','| Shuffled history | Current fixed; per-family past-step order permuted |','| Sample-mismatched history | Current fixed; past supplied by another family within the same partition |','| Residualized history | Current plus past residuals after a train-only current-to-past ridge fit |','',
'For the CLS view, matched arms have 73/109/145/181 fitted coefficients including intercept at steps 2/3/4/5; the simple reference has 37. For the all-token view, matched arms have 425/461/497/533 coefficients; the simple reference has 389. Equal fitted coefficient count does not establish equal effective capacity, fixed-transform size, or inference cost.','',
'At step 2 there is only one past step; shuffled history is an exact alias and supplies no chronology test. Its duplicate results are labeled. The panel has 336 reported cells but 324 distinct ridge fits and 1,620 alpha candidates.','',
'## Primary and supporting comparisons','',
'Positive ΔR² means history predicts better. Positive MAE advantage means history has lower absolute error. Each seed receives equal weight after averaging its training-eligible steps.','',
'| Current view | Comparator | Mean ΔR² | Conditional 95% interval | Mean MAE advantage (nats) | Seeds with positive ΔR² |','|---|---|---:|---|---:|---:|']
for obs in ['cls_current','full_current']:
 for baseline in ['current_linear','current_rff']:
  row=next(x for x in panel if x['observer']==obs and x['baseline']==baseline);lo,hi=row['conditional_ci95']
  lines.append(f"| {label[obs]} | {'Simple linear' if baseline=='current_linear' else 'Matched nonlinear'} | {row['mean_delta_r2']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {row['mean_mae_advantage']:+.6f} | {row['positive_seed_count']}/6 |")
lines+=['','The primary comparison is all-token current plus CLS history versus the matched nonlinear all-token current-only control. The simple all-token current-only reference is a necessary additional check. Both fail the frozen screen.','',
'The screen required mean ΔR² ≥0.01, a conditional bootstrap lower bound above zero, a positive effect in each of six seeds, and nonnegative mean MAE advantage. The 0.01 threshold is a fixed exploratory screening convention, not an established controller-utility threshold. The primary interval includes both zero and +0.01, so this study does not establish equivalence or rule out every practically useful effect.','',
'### Absolute predictive performance','',
'| Current view | Simple current R² | Matched nonlinear current R² | Ordered-history R² |','|---|---:|---:|---:|']
for obs in ['cls_current','full_current']:
 vals=[pooled.loc[(obs,arm),'test_r2'] for arm in ['current_linear','current_rff','ordered_history']]
 lines.append(f"| {label[obs]} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} |")
lines+=['','These are equal-seed means of within-seed eligible-step scores, not R² calculated on a pooled dataset. Overall predictive strength remains limited; no controller performance was evaluated.','',
'### Per-seed effects against the matched nonlinear control','',
'| Seed | CLS-view ΔR² | All-token-view ΔR² |','|---:|---:|---:|']
for s in [503,607,709,811,907,1009]:
 vals=[next(x['delta_r2'] for x in seed if x['seed']==s and x['observer']==obs and x['baseline']=='current_rff') for obs in ['cls_current','full_current']]
 lines.append(f'| {s} | {vals[0]:+.6f} | {vals[1]:+.6f} |')
lines+=['','![Per-seed and six-model history effects](T2_E1_History_Effects.png)','',
'## What the controls show','',
'Against shuffled history on chronology-eligible steps, ordered history has mean ΔR² +0.008745 for the CLS view, interval [−0.030659, +0.022776], and −0.011049 for the all-token view, interval [−0.099312, +0.008237]. Only five seeds contribute because seed 607 has no eligible step with two past slots. Neither supports a reliable chronological-order advantage.','',
'Against sample-mismatched history, mean effects are +0.023098 (CLS) and +0.010923 (all tokens), but both conditional intervals include zero. Residualization and dimension matching also do not produce a consistent panel-wide history advantage. All control results remain in the saved metrics and comparison files. These secondary comparisons are descriptive and are not multiple-testing-adjusted discoveries.','',
'## Uncertainty and interpretation limits','',
'The 95% intervals use 1,000 paired bootstrap resamples of the 1,000 test families within each seed. Each resample retains both pair members and uses the same draws across steps and arms. The six model seeds and trained probes are held fixed. These intervals describe held-out-example uncertainty conditional on that fixed setup; they do not cover fresh model initialization, retraining, training-set variation, hyperparameter selection variability, or the general population of Transformers. Steps are not counted as independent model replicates.','',
'The evidence concerns CLS history under two current-observation choices, a particular auxiliary-supervised training recipe, a two-hop task, and ridge probes with specified fixed features. It does not establish that the full current state is universally sufficient, that all useful history has been tested, or that metacognitive computation is absent. No claim about decision-stability advantage, attention-specific mechanisms, causal self-monitoring, or adaptive-computation savings follows.','',
'## Verification','',
'- Frozen protocol, source files, and six checkpoint hashes were checked before and after execution.',
'- Six method checks validated the ridge path against scikit-learn on collinear synthetic inputs, test-label isolation, train-only normalization, temporal permutation, future-state exclusion from past controls, and counterfactual family grouping.',
'- All 336 held-out metric rows were independently recomputed from the saved predictions using scikit-learn metrics. Seed-level and panel-level point aggregation were verified without refitting.',
'- Exact family overlap with excluded data and cross-partition family overlap were zero for every seed.',
'- One fixed study run completed in 50.78 seconds in the recorded CPU environment. No outcome-driven parameter expansion or study rerun occurred. Method-check fitting was synthetic and separate from the 324 research probe fits.','',
'## Decision and next boundary','',
'Close T2-E1 with a negative/inconclusive history-advantage result under its frozen exploratory criteria. Keep the original T1 negative evidence and U7 competence failures. Do not tune this exposed test panel to obtain a positive result. No confirmation advancement is authorized by these results.','',
'The useful methodological finding is that a comparison against a simple CLS linear baseline can give a more favorable impression than a comparison against a matched nonlinear current-only baseline. A later study should justify a new model/task regime and a concrete predictive-use threshold before new data or fitting. This result provides no basis yet for claiming that a history-based controller will improve decisions or save computation.','',
'## Reproduction and sources','',
'Open `T2_E1_Observer_Study_Colab.ipynb` in Colab and upload `T2_E1_Observer_Study.zip`. The notebook verifies package integrity, checks the methods, and writes a new reproduction directory. The bundle includes original source, six checkpoints, exact protocol, generated examples and labels, test predictions, full metrics, validation choices, figure code, and verification records. Intermediate full-state tensors are regenerated from frozen checkpoints to keep the bundle small.','',
'- [Original T1 report](https://drive.google.com/file/d/1e4x-TVQxDMjAx9nAn1KOUTyh2hKpwdnc/view).',
'- [Completed U7C report](https://drive.google.com/file/d/1k-ZR3KaprlnSEiOXjXSFEibPdRwKlaTl/view).',
'- [Matched U7B/U7C audit](https://drive.google.com/file/d/1q4ALgMG3ZvX3MI-4y4Ki68ADrOicNNWU/view).',
'- `STUDY_PROTOCOL.json`, `FROZEN_INPUT_SHA256.json`, `results/study_result.json`, `results/data_checks.json`, and `results/result_verification.json` document the new study.','']
(root/'T2_E1_Report.md').write_text('\n'.join(lines))
(root/'README.md').write_text('''# T2-E1 — observer sufficiency\n\nRead `T2_E1_Report.md` first. The fixed exploratory study is complete; no robust history advantage met the frozen criterion.\n\n## Reproduce in Colab\n\nOpen the enclosed notebook and upload `T2_E1_Observer_Study.zip`. It verifies every package entry before running code. CPU is sufficient. All model weights and data-generation code are included.\n\n## Reproduce locally\n\nWith Python 3.12, install `requirements.txt`, then run:\n\n```bash\npython verify_methods.py\npython run_study.py --out reproduced_results\n```\n\nThe output directory must be new. Original distributed results remain intact. The study regenerates exact fresh data from its fixed seeds and exclusions. Intermediate full residual tensors and fitted probe coefficients are reproducible from the frozen model checkpoints, code, exact data, and saved alpha selections; they are not retained separately.\n\n`verify_results.py` checks the distributed `results` directory without fitting probes. `make_figure.py` regenerates the scientific figure from saved comparisons.\n\n## Key files\n\n- `STUDY_PROTOCOL.json`: scope, observables, targets, controls, budgets, and screen frozen before the research fits.\n- `frozen/`: original model code, six U7C checkpoints, original training protocol, and prior audit record.\n- `results/data/`: exact generated paired examples, ordered training/validation/test.\n- `results/predictions/`: held-out targets, predictions, and training-derived eligibility.\n- `results/probe_metrics.csv`: all 336 cells, including ineligible steps and explicit aliases.\n- `results/validation_selection.json`: all five alpha scores and the selected alpha per cell.\n- `results/seed_comparisons.json` and `panel_comparisons.json`: paired effects and conditional intervals.\n- `results/result_verification.json`: independent score and aggregation checks.\n- `execution.log`: the single completed study run.\n\nReserved seeds are not instantiated. No base-model training runs. This is an exploratory fixed-model comparison, not an untouched-seed confirmation.\n''')
# Package every reproducible source and result, excluding import caches.
files=[p for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_SHA256.json']
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(root/'PACKAGE_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=root.parent/'T2_E1_Observer_Study.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files+[root/'PACKAGE_SHA256.json']:z.write(p,Path(root.name)/p.relative_to(root))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for name,digest in manifest.items():assert hashlib.sha256(z.read(f'{root.name}/{name}')).hexdigest()==digest
for name in ['T2_E1_Report.md','T2_E1_Observer_Study_Colab.ipynb','T2_E1_History_Effects.png','STUDY_PROTOCOL.json']:
 target='T2_E1_Protocol.json' if name=='STUDY_PROTOCOL.json' else name
 shutil.copy2(root/name,root.parent/target)
print('Report and verified reproduction package complete')
print('ZIP bytes:',archive.stat().st_size)
print('ZIP SHA256:',hashlib.sha256(archive.read_bytes()).hexdigest())
