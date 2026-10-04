# F1-A3 — Cheaper decision-specific bias estimation

Status on 4 October 2026: methods development started. The main experiment is not frozen or run. There are no new planning-value or controller findings.

## Motivation

F1-A2 found 6.89% lower prediction MSE with estimated candidate-plan cost bias, but its controller missed the adjusted uncertainty gate and ran 2.08–3.46 times slower. This follow-up tests whether useful prediction can survive lower online cost. It preserves all completed F1 results.

## Prototype implemented

1. A compact quadratic residual map trained only on a transition-calibration bank. It replaces per-query neighbor searches and local linear solves with one matrix multiplication. A full 32-candidate rollout still costs 320 model transitions; faster correction does not by itself reduce model calls.
2. An eight-candidate subset selected deterministically by evenly spaced ranks of current model cost, including the incumbent. Ten-action corrected rollout would cost 80 model transitions. Subset bias summaries must be defined separately from F1-A2's 32-candidate, eight-elite summaries; reusing those formulas unchanged would be invalid.

The methods pilot compares residual estimators on a fixed 1,536/512 split of the inherited transition-calibration bank, with new deterministic split seed 73002. Neither estimator is fit to the held-out transition residuals. It reads no planning evaluation arrays, benefit labels, later candidate plans, or reserve families. Fixed quadratic degree and ridge 0.001 are engineering choices, not parameters selected on held-out error. It records residual RMSE and warm-run query timings; these are feasibility diagnostics, not independent confirmation or evidence of controller improvement.

## Proposed main comparison

Keep the world models, task, horizon and planner fixed. Generate fresh scenes and noise. Compare a tuning-selected current-state reference, the F1-A2 local-linear estimator on 32 candidates, the compact estimator on 32 candidates, and compact correction on eight fixed candidates. Fit/tune/evaluation IDs must be shared across all arms and all model conditions.

Score prediction of true improvement from round 3 to round 6; then score a prefix-only stop/continue controller. Charge the actual model transitions required by the chosen estimator and continuation. Measure end-to-end single-decision wall time with warm-up, paired case order and repeated trials. Count calibration/training cost separately from online cost. Preserve privileged simulator diagnostics as diagnostics only.

Before opening fresh evaluation outcomes, freeze the exact subset summaries, fixed primary arm, sample count, seeds, controller charge, practical runtime gate, multiplicity correction and effect thresholds. Do not select the best estimator on test outcomes. A provisional prediction target is at least 5% reduction versus the stronger reference with an adjusted positive lower bound; practical control also needs positive charged improvement and no runtime regression. An inferior feasibility pilot may motivate an explicit prospective amendment, not a retrospective success criterion.

## Interpretation limits

Fast one-step correction can lose accuracy during rollout. Lower residual RMSE, fewer model calls, improved prediction and improved control are four different claims. This study cannot establish subjective awareness, general metacognition or an attention-specific mechanism.
