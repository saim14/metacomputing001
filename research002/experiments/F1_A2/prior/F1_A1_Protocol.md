# F1-A1 — Reliability, history and geometry in metacomputation

Research 002 · Saim 13.02 · 17 September 2026 · version 1.0

Prospective development protocol, frozen with source hashes before generating new study outcomes or fitting observers. This is a new study informed by F1-A0, not an untouched confirmation, and it does not replace that pilot's negative result. No P1 reserve family or T1 sealed seed is used.

## Purpose and fixed scope

F1-A0 associated less accurate world models with harmful additional planning, but did not manipulate model error. F1-A1 separates:

1. The effect of intervening on the planner's transition-model error.
2. Prediction from current planner observables C, past planning states H, and imagined-future geometry G.
3. Added value of a validation-error descriptor R, acquired without evaluation outcomes.
4. A per-case stop/continue rule with an explicit computation charge and a separately measured runtime.

One environment, three previously frozen neural checkpoints, seven transition conditions, 2,400 new scenarios, sixteen feature arms, twelve candidates per arm/condition: 1,344 observer fits, 112 selected observers, **zero neural retraining**. No outcome-driven extension, model selection, feature search or second run is authorized within this protocol. Numerical audit and deterministic replay are allowed and are not additional fitted candidates.

## Controlled world-model intervention

Use the unchanged F1-A0 2D dynamics, ten-action cost function and neural checkpoints 3109, 3119, 3137. Let f be the true one-step simulator and f_m the frozen neural transition function. Define:

`f_(m,alpha)(x,a) = f(x,a) + alpha * (f_m(x,a) - f(x,a))`.

There is one exact condition (alpha=0), and alpha=.5 and 1 for each of three checkpoints. Exact dynamics are counted once, not as three independent models. At every fixed state/action, one-step vector error is scaled exactly by alpha. Multi-step errors and optimized decisions need not change monotonically because the planner visits different states.

The intervention changes only the planner's transition function. Pair scenarios, goals, obstacles, planner random draws, candidate budget, true dynamics and scoring. This can support a causal statement about these specific substituted error fields. It cannot establish a population-wide law about neural model quality. The exact simulator and interpolated models are experimental controls that would not be available to an ordinary deployed agent.

Retain every condition. Do not discard inaccurate conditions using a competence gate, since error is the manipulated variable. Report one-step and ten-step error on a disjoint diagnostic set, never choose conditions using planning outcomes.

## New data and planner

Generate 2,400 scenarios from the same F1-A0 scene distribution with new seed 52001. Split by whole scene, permutation seed 52002: 1,200 fit, 400 tuning, 800 evaluation. All condition variants of a scene share its split. The exact scenario distribution remains development-exposed; fresh samples are not confirmation of architectural generality.

Retain the F1-A0 planner: six cross-entropy rounds; 32 ten-action sequences per round; eight elites; initial action means clipped to [-.8,.8]; initial SD .65; later SD floor .08; incumbent and current-mean candidates retained. Use seed 53001 for noise, shared across conditions. No outcomes select a restart or extra round.

Move the observation checkpoint from round 2 to round 3, so H contains two nontrivial past slots. The target is:

`V_3 = L_true(U_3) - L_true(U_6)`.

All negative values are retained. This is true plan-cost reduction, not decision agreement, remaining NLL, Brier score or optimal-policy regret. It differs from F1-A0's checkpoint; report that difference explicitly.

## Observables and sixteen arms

C has 100 coordinates: start/goal/obstacle context, next-round action means and SDs, incumbent actions, sorted current candidate estimated costs and incumbent estimated cost. H concatenates the corresponding 100-coordinate C snapshots from rounds 1 and 2 in order (200 coordinates). G retains F1-A0's 12 Euclidean geometric summaries of the round-3 candidate rollouts. Raw rollouts are every position from those same 32 ten-step candidate trajectories (704 coordinates).

The reliability bank uses 2,048 independent state/action samples, seed 51001, positions uniform [-4,4]^2 and actions uniform [-1,1]^2. For each condition, store true one-step squared coordinate error on this bank. No scene target or evaluation trajectory enters it. For each point/action of the round-3 incumbent's predicted ten-step path, query the 16 nearest bank inputs after scaling position by 4 and action by 1. Local error is the square root of mean bank error across those neighbors. R has four coordinates: mean and maximum local error along the path; maximum distance to the 16th neighbor; and global bank coordinate RMSE. The last coordinate is constant within a condition. R is a validation-error descriptor, not a guaranteed calibrated uncertainty or a direct observation of error on the evaluated plan.

| Arms | Meaning |
|---|---|
| C, CH, CG, CHG | Current / current+history / current+geometry / all three |
| CR, CHR, CGR, CHGR | The same four with the reliability descriptor |
| C-expanded, CR-expanded | 212 fixed cosine features added to fit-standardized C or CR; widths match CHG and CHGR respectively |
| C-raw, CR-raw | Current, with/without R, plus all raw round-3 candidate rollouts |
| C-mismatched-H-G | Keep C and G; use an unrelated scene's complete H |
| C-H-mismatched-G | Keep C and H; use an unrelated scene's G |
| C-shuffled-H-G | Independently permute the two H slots per scene, keeping C and G fixed |
| C-H-G-mismatched-R | Keep C,H,G; use an unrelated scene's R |

Sample-mismatch mappings are strict derangements within each partition and condition: a random ordering is cyclically shifted by one position. No self-pair or cross-partition pair is allowed. Use seed 54002 plus condition index; generate H, G and R mappings in that order. The temporal shuffle uses independent random swaps (seed 54003 plus condition index), not one global reversal. It does not guarantee removal of chronology inferable from the state values. No general chronology claim is planned.

All observer input components are rounded to float32 then fitted in float64. This bounds cache size equally across representations; complete source and data are retained. Geometry remains a transformation of current cached imagined trajectories. Added predictive access does not mean extra information beyond the full computational state.

## Fixed observers and references

Fit each arm separately for each of seven conditions. Candidates: standardized linear ridge, or 128 fixed Gaussian random Fourier features at bandwidth factors .5 or 2; penalties .0001, .01, 1 and 100. Twelve candidates per arm. The larger maximum penalty is a prospective change motivated by F1-A0 boundary selections. Fixed Fourier seed 55001, expansion seed 54001. Fit every normalization and intercept using only fit scenes. Select minimum tuning MSE, breaking ties by listed order. Never clip predictions or tune on evaluation data.

The strong reference without R is chosen on tuning MSE among C, C-expanded and C-raw. The strong reference with access to R is chosen on tuning MSE among C, CR, C-expanded, CR-expanded, C-raw and CR-raw. Their larger reference-selection budgets are conservative controls; equal input width alone is not claimed to match effective capacity.

All coefficients, preprocessing statistics, candidate tuning scores, selected pipelines and evaluation predictions are saved. Fourier matrices are exactly regenerated from their frozen seed/dimension/bandwidth recipe to avoid storing identical large matrices repeatedly. Store data atomically and verify each numerical archive on write.

## Prespecified endpoints and inference

Report separate outcomes; do not rescue one failed question by switching to another. All intervals are conditional on the fixed checkpoints/error fields, scene distribution and fitted observers. Resample the same 800 scene IDs jointly across all conditions and arms, seed 56001, 4,000 bootstrap replicates. Do not resample model-step rows independently. Use two-sided 98.75% percentile intervals for the four headline endpoints (Bonferroni allocation of a 5% familywise error budget); other 95% intervals are explicitly descriptive.

1. **Error intervention:** risk difference for materially harmful continuation (V<−.01), full neural error averaged across the three models minus exact dynamics. Require at least +5 percentage points and positive headline lower bound. Retain each model and half-error condition. This endpoint concerns the intervention, not geometry.
2. **History + geometry:** pooled CHG MSE reduction versus the strong no-R reference, equally weighting the six imperfect conditions. Require at least 5% relative reduction and positive headline lower bound. For attribution, separately require positive descriptive lower bounds versus CH, CG, both mismatches and the strong reference. If these fail, report the joint comparison without claiming both components contributed. Exact-model results are a diagnostic, not part of the imperfect-condition pool.
3. **Reliability:** pooled CHGR MSE reduction versus CHG, same six conditions. Require at least 5% relative reduction and positive headline lower bound; require a positive descriptive lower bound against mismatched R before claiming access to correctly paired reliability information mattered.
4. **Online decision rule:** candidate CHGR versus a tuning-selected current-based controller, using the objective below. Require at least 1% relative reduction in charged objective and positive headline lower bound. A claim of actual time saving additionally needs the prespecified runtime comparison, not merely fewer planning iterations.

The percentage cutoffs are screening conventions, not economic values supplied by the user. These thresholds and wide intervals do not turn a development study into confirmation. Per-condition outcomes, constant-fit-mean baselines, harmful continuation fractions, width/candidate counts and selected penalty boundaries remain visible.

## Per-case controller and costs

After round 3, continue to round 6 iff predicted V exceeds **.01**. This threshold is fixed before outcomes and applied per case, without sorting an evaluation batch or enforcing a quota. The primary charged objective is true executed-plan cost plus .01 if continuing. Common prefix cost is omitted because every compared controller pays it. Charge .01 corresponds to the same additional three planning rounds (960 transition predictions) within each condition. It excludes descriptor/readout work and is **not** a full wall-clock cost or a universal utility conversion.

Select the current-based controller on tuning charged objective among C, CR, C-expanded, CR-expanded, C-raw and CR-raw, using each arm's MSE-selected predictor. This controller reference may differ from the prediction-MSE reference. Report charged cost, uncharged cost, continuation rate and harmful executed continuation. Preserve always-stop, always-continue and hindsight-oracle comparators. Charges 0, .005, .02 and .05 are fixed descriptive sensitivities; none can replace the primary .01 endpoint.

Runtime: for the first 24 evaluation scene IDs in their frozen split order, replay candidate CHGR, its chosen current-based controller, always-stop and always-continue online in every condition. One untimed warm-up and three timed repetitions; rotate method order each repetition. Use a single CPU thread. Include planning, collection of required history, feature extraction, cached reliability-bank queries, feature transforms, readout and actual extra rounds. Exclude loading fixed parameters, building the bank, true-simulator outcome scoring and serialization. Use identical per-scene random draws. Check resulting decisions and executed plans against offline counterfactual evaluation. This subset is fixed in advance, not selected for favorable timing.

Report the one-time bank acquisition cost separately (2,048 true transitions and predictions for each condition, with a common bank). Its estimation work is available only because the simulator supplies calibration observations. Single-process batched candidate evaluation and a small timing panel cannot establish hardware-general latency savings or performance on new environments.

## Verification and stop rule

Before outcomes: fixed-method fixtures for exact interpolation endpoints, error-vector scaling, strict derangements, scene-split isolation, and a synthetic controller value/cost case. No pilot tuning run on the actual study targets.

After outcomes: reconstruct all candidate tuning scores and selected predictions without refitting; independently replay saved plans in the true simulator; verify two-round history and third-round inputs can be constructed without later computation; reconstruct intervals/decisions; verify unchanged frozen sources and checkpoint hashes. Any fault is recorded and corrected as an implementation amendment, never hidden as new evidence. No additional neural training or observer candidate is added after seeing results.

A positive error-intervention result does not validate the representation/controller hypotheses. A failed representation result does not reverse the model-error finding. Save the complete result matrix, including failures, and stop this study after audit and reporting.

## Related work and scope

Model exploitation and empirical model generalization are established concerns in [Janner et al., When to Trust Your Model](https://arxiv.org/abs/1906.08253). [Yu et al., MOPO](https://arxiv.org/abs/2005.13239) uses uncertainty-penalized rewards in offline model-based policy optimization. F1-A1 is neither method nor a claimed first discovery of model bias; its contribution is a controlled project-specific test linking these concerns to metacomputation observables.

No flow-matching training, recursive attention, learned latent manifold, partial observability or consciousness measurement is included. Those remain separate potential interventions after the current value-of-computation mechanism is assessed.
