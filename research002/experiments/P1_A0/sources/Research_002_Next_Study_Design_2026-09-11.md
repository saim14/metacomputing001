# Research 002 — next hypothesis study: design decisions

Date: 11 September 2026
Status: design memo only; no executable probe protocol or training authorization.

## Preserve the question

Does a fixed, affordable predictor use computational history to predict future decision change or future loss reduction more accurately than a comparable predictor using specified current observables? A positive result establishes incremental decodability under those observables, task, training regime, and predictor budget. It does not alone establish causal use or metacognition.

## Clarify what was actually observed

The recovered `extract_trajectory` function records the token at position 0 (`CLS`), not the complete residual tensor and not the query token at position 1. Its state features are the 32-dimensional CLS vector, two class probabilities, predictive entropy, and probability margin. `build_control_features` concatenates these features from step 0 through the current step. Calling this simply “current state” hides an important restriction.

For the next design, distinguish two comparisons:

| Comparison | Current observables | What a history advantage would establish |
|---|---|---|
| Limited observation | Current CLS vector and prediction summaries | History improves accessible prediction beyond this restricted current view |
| Stronger current-state check | All token residual vectors at the current step, with the same current prediction summaries | Whether the advantage survives a more complete current observation under the declared predictor budget |

An additional attention-history comparison must include a corresponding current-attention baseline. Do not add attention features only to the history side.

## Separate the targets

- Decision target: current prediction disagrees with the final prediction. It measures change, not whether the final answer is correct.
- Loss target: current true-label NLL minus final true-label NLL. It can be negative. Labels and final outputs are used only to construct the evaluation target; neither enters predictor inputs.
- Keep both original T1 results in the evidence record. A later loss-target result cannot count as confirmation of the earlier decision-target test.

## Requirements before any future fit

1. Select the task and model regime without using future history-versus-current results. This closed U7 branch has not earned confirmatory probe advancement. Do not select only seeds 503/607/709 or pool failed seeds into a passing mean. Any use of these frozen models for an exploratory study requires a separately versioned change of scope and must report all six.
2. Freeze the primary target, primary metric, eligible steps, smallest practically relevant improvement, and negative/equivalence/ineligible outcome rules. Choose the practical threshold from the intended use and prediction cost, not from observed test gains.
3. Fix current and history observables, the history window, predictor families, parameter counts, scaling, regularization, optimization and tuning budgets. Repeating the current vector matches input width but does not add independent current-state features or by itself prove matched effective capacity. Include a stronger current-only predictor at a comparable compute budget.
4. Preserve dimension-matched, sample-mismatched, and train-only-residualized history controls. A chronology claim additionally needs a per-example past permutation control with the current observation fixed and at least two nontrivial past slots. A fixed reversal of flattened history is just a feature-column permutation.
5. Keep all steps of an example and related counterfactuals in one data split. Fit normalization and residualization on the training split only. Group seed-step outcomes by model seed; do not treat the five steps of one model as five independent training replicates.
6. Freeze a finite development budget and the conditions for untouched confirmation before seeing any new outcomes. Reserved seeds 1103/1201/1301 remain sealed under the current decision.

## Immediate design recommendation

Treat the observer definition as the next decision: the source supports a practical claim about limited CLS observables. A stronger claim requires the all-token comparison as well. Decide which claim is intended before choosing another training recipe. The matched-example audit closes the intervention-effect comparison; it does not resolve this hypothesis.

Sources: the verified U7B/U7C `src/t1_experiment.py`, original T1 report, completed U7C report, and accepted direction freeze. This is a source-grounded design proposal, not a new empirical finding.

