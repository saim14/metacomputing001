# F1-A3 main-v1: candidate-sparse decision-specific error estimation

Author: Saim 13.02. Prospective development study, 4 October 2026.

## Durable starting point

F1-A2 established an estimated candidate-cost-bias prediction gain, but not a practical controller improvement. F1-A3's saved calibration-only pilot found that a quadratic correction was fast but inaccurate. The GitHub draft and pilot at commit 587bcac2c7f237468abfe647d3f865b6cdd375f9 are preserved. Conversation context mentions a later main-run freeze/start, but no corresponding protocol or run was recoverable from the repository or Library. This separately identified main-v1 run therefore freezes its complete specification before generating outcomes, using fresh 82xxx/83xxx seeds; it does not claim to resume or reproduce unrecorded results.

The recovered F1-A2 archive SHA-256 is dd36e474ba3072c960eb6fca1e52ecc5a21629a2633e24599d8c714188e3639e. Only frozen world models, inherited algorithms/configs and calibration transitions are dependencies. No old planning evaluation arrays are used. No Transformer reserve seeds or arithmetic holdouts are opened. Earlier conclusions remain fixed.

## Question and bounded budget

Can local-linear correction on eight current candidate plans retain useful prediction of the value of continuing, while costing less than correction on all 32 plans? There is one primary candidate, B8, chosen before outcomes. The failed quadratic prototype remains a recorded engineering result, not a competing arm searched on these outcomes.

Use the inherited two-dimensional navigation world, three frozen neural models (3109, 3119, 3137), exact dynamics plus half/full error conditions, horizon 10, 32 candidates, 8 CEM elites, and six planning rounds. Observe at the end of round 3. The target is Y = true_cost(plan_3) - true_cost(plan_6), evaluated by the simulator. Larger Y favors continuation. All arms see the same planner and noise; their features never change candidate generation.

Generate exactly 2,400 fresh independent scenes with seed 82001; split by whole scene with seed 82002 into 1,200 fit / 400 tune / 800 evaluation. Every counterfactual model condition and arm for a scene uses the same split. Noise seed 83001. The inherited 2,048-transition calibration bank is disjoint in purpose from all planning scenes and unchanged; correction sees only state/action residual labels. This is new-scene development evidence conditional on three previously studied models, not independent new-model confirmation.

## Available features and controls

Keep seven inherited reference arms: C (100 current planner descriptors), CR (C plus four local/general residual magnitudes), CE and CRE (matched deterministic nonlinear expansions), CRAW and CRRAW (all current candidate paths), X (C plus two earlier planner summaries, geometric path descriptors, and reliability). The strongest predictive reference is selected by tune MSE separately for each condition. The controller reference is selected from the same seven arms by tune charged objective, using each arm's MSE-selected predictor. All selections precede evaluation scoring.

Add B32 = C plus the inherited twelve corrected-cost-bias summaries over all candidates. B8 = C plus twelve analogous summaries over current model-cost ranks [0,4,9,13,18,22,27,31], selected by stable sort. The subset contains the current best candidate; it does not mean choosing eight low-cost candidates. For B8, 'elite' means the lowest two sampled ranks and 'rest' the other six, retaining a one-quarter elite fraction. The remaining summaries are incumbent bias, mean/SD/min/max bias, bias-cost slope, cost correlation, discordant pair fraction, subset regret and fraction better than the incumbent. They describe the sampled set and are not unbiased estimates of the full-set summaries.

Additional fixed controls: MB8 pairs the entire B8 descriptor with a different scene, strictly deranged within each split; B1 adds incumbent bias only; XB8 adds the B8 descriptor to X. No new reference is chosen from these controls. B8 and B32 have identical input dimension and fitting budget. CE/CRE test additional feature capacity; MB8 tests scene pairing; B1 tests candidate-set information; XB8 describes history/geometry integration without making it a primary claim.

All arms use the inherited fixed 12-candidate predictor grid: linear ridge and two random Fourier feature kernels (128 features, bandwidths 0.5/2), each at regularization 0.0001/0.01/1/100. Normalization, intercept and coefficients use fit rows only. Tune MSE selects the predictor. Seven conditions times twelve arms times twelve candidates = 1,008 candidate fits and 84 selected pipelines. No hyperparameter, arm, seed or scene-count extension is permitted after evaluation.

## Primary tests, uncertainty and decision rules

Equal-weight the six imperfect model conditions within each evaluation scene. The exact condition is a diagnostic. Scene-clustered paired bootstrap uses 4,000 draws with seed 86001; both half/full errors and all arms stay paired. The three primary tests use 98.333% two-sided intervals (Bonferroni family size three); no claim treats 4,800 correlated condition-scene rows as independent.

1. Useful B8 signal: at least 5% pooled MSE reduction versus the tune-selected predictive reference, with the adjusted lower bound of paired MSE improvement above zero.
2. Retained accuracy: adjusted upper bound of (MSE_B8 - MSE_B32) / MSE_B32 is below 0.02. This is an explicit noninferiority test; a nonsignificant difference is not enough. B32's own gain on fresh scenes is reported descriptively, so equivalence between two unhelpful estimators cannot satisfy test 1.
3. Charged control: at least 1% reduction in pooled charged objective versus the tune-selected controller reference, with adjusted lower bound above zero.

Continue when predicted Y > 0.01. Three additional planner rounds cost 960 state-action transitions. The online correction costs B32=320, B8/MB8/XB8=80 and B1=10 additional transitions. For arm a, J_a = true_cost(chosen plan) + 0.01*[continue + correction_calls(a)/960]. The shared first three rounds cancel from J but are included in runtime and reported transition totals. Correction cost is charged even when stopping. Reference features incur no additional model transitions; all feature extraction, nearest-neighbor work, transforms and prediction are included in runtime. The charge is a declared toy-world exchange rate, not hardware energy. Offline calibration and observer fitting time are reported separately.

Runtime is an additional engineering gate, not a fourth efficacy claim: fully replay 32 fixed evaluation scenes per condition for five repeats with one numeric-library thread, warm-up, paired scenes and randomized mode order. Compare B8, B32, the controller reference, always-stop and always-continue. B8 must be faster than the controller reference in every imperfect condition and reduce mean per-condition runtime by at least 10% relative to B32. Practical success requires all three primary tests and this runtime gate. Report raw paired timings and scene-bootstrap 95% ratios as hardware-specific descriptive evidence.

## Diagnostics and management

Report B8 vs MB8, B1, XB8 and B32 with unadjusted 95% intervals marked exploratory, every condition's outcomes, and always-stop/continue/hindsight-oracle references. Decompose stop/continue regret into false continuation and false stopping, weighted by |Y-0.01|; summarize cases with predicted |Y-0.01| <= 0.01. These diagnostics cannot rescue a failed primary gate and do not select a new controller on evaluation data.

Generate each expensive planning trajectory once. Offline B8 and B1 correction can be sliced from deterministic B32 corrected rollouts; independently recompute the subset online to verify identical features. Saving this shared offline work does not reduce the recorded online cost. At each condition save data atomically before fitting, then save all selected models and a completion record. Resume only under identical source/protocol/dependency hashes. No status update uses evaluation results before tuning selections are fixed. Audit reconstructs all tuning scores, reference choices, selected predictions, target costs, intervals, subset features and model-call counts without refitting. Block simulator access while reconstructing accessible features for full-error neural models. Zero/half-error conditions intentionally use simulator-derived intervention dynamics and are not deployable models.

Stop if numerical, leakage, source-integrity or replay checks fail; preserve the failed checkpoint. A negative/inconclusive result closes this run. Any later modification receives a new protocol and fresh evaluation scenes. Record a research decision in the project state rather than automatically starting another fit.

## Scope and related work

This is a narrow computation-allocation experiment in one toy environment. It cannot establish awareness, general metacognition, Transformer mechanisms or real-world deployment utility. The connection is deciding when further computation is worthwhile. Related primary literature includes [Sung et al., Learning When to Quit (2021)](https://arxiv.org/abs/2103.04374) on stopping motion planning and [Callaway et al., Learning to select computations (2018)](https://arxiv.org/abs/1711.06892) on computation selection including metareasoning overhead. These motivate the question, not the present thresholds or evidence of novelty.
