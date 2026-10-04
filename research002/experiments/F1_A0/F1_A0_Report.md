# F1-A0 — Geometry of Foresight meets metacomputation

Research 002 · Saim 13.02 · 17 September 2026

**A separate foresight branch has been implemented and tested. The frozen pilot did not establish that geometric features improve prediction or planning allocation beyond its stronger controls.** Its main contribution is a working experiment linking imagined world trajectories to the actual value of further computation, with a clear model-reliability limitation.

## The connection between the three angles

| Angle | Question | Existing evidence or new test |
|---|---|---|
| Computational past | Does the way a solution developed help predict what further computation will do? | Research 001 trajectory results; the separate T1/T2 Transformer tests retain their own bounded negative outcomes. |
| Computational present | How much useful prediction can a reader extract from the current working state? | P1-A1 met its 0.01 Brier tolerance in two arithmetic Transformers; cached-history prediction was faster. |
| Imagined futures | Does the shape of candidate future world trajectories help decide when to plan longer? | F1-A0, the new neural world-model pilot reported here. |

These are a proposed organizing framework for the research. They are not three established components of an already successful architecture. Their published metrics cannot be pooled into one score.

A world model predicts what may happen in the environment. A metacomputation observer predicts what further internal computation may contribute to the decision. That makes the connection operational: **use available representations to allocate the next unit of thinking according to its expected decision value.**

“Past,” “present” and “future” have distinct references here. Prior planning revisions concern the computation's past; candidate rollouts concern the world's hypothetical future. Both can be stored in the computer's current memory. Complete deterministic state sufficiency is compatible with differences in how cheaply a limited predictor can extract useful information.

## What was actually run

- Three independently initialized, fixed neural world models learned a nonlinear 2D navigation transition function from 8,000 transition examples. Each used two 32-unit tanh layers and exactly 180 training epochs.
- 1,600 navigation scenarios with a start, goal and circular obstacle: 1,000 observer-fit, 300 tuning and 300 evaluation cases. The same scenario identities were paired across the three models.
- A six-round planner considered 32 ten-action candidate plans per round. After round 2, observers predicted the value of continuing to round 6.
- Five observer arms: current planner features; width-matched nonlinear current features; current plus geometry; current plus all raw candidate rollouts; and current plus permuted geometry.
- Geometry comprised 12 summaries of imagined path length, turning, clearance, goal distance, endpoint dispersion and path efficiency. These are physical-coordinate measurements, not learned latent-space geodesics.
- Each arm received nine fixed ridge/feature candidates: **135 candidate fits and 15 selected predictors**. The reference was selected using tuning data among three current/raw-rollout arms, giving it a conservative 27-candidate combined search budget.

The primary target was:

`V_2 = true executed-plan cost after 2 planning rounds − true executed-plan cost after 6 rounds`.

Positive values mean extra computation helped; negative values mean it harmed the plan. Outcomes came from the true simulator, not from the learned world's own estimate. This is a comparison of two finite plans, not regret against a globally optimal controller.

The prospective protocol and executable were hashed before training. This is one development pilot, not externally preregistered or untouched confirmation. No earlier sealed seed or arithmetic reserve family was opened.

## Primary result: the advancement criterion was not met

| Measurement | Result |
|---|---:|
| Pooled selected-reference MSE | 0.04001394 |
| Pooled geometry MSE | 0.03977254 |
| Absolute MSE reduction | +0.00024140 |
| Conditional paired 95% interval | [−0.00017841, +0.00067431] |
| Relative MSE reduction | +0.603% |
| Prospectively required relative reduction | At least 5%, plus the remaining checks |

The confidence interval crosses zero. Geometry also failed to establish a positive difference against mismatched geometry: mean MSE reduction +0.00021677, interval [−0.00010152, +0.00055564]. All five combined advancement conditions failed. The pooled raw-current arm alone had MSE 0.03972418, slightly below geometry; this descriptive comparison is retained even though the primary reference was chosen on tuning data rather than evaluation outcomes.

Per-model results must not be replaced by the pooled mean:

| Model seed | Ten-step world-model coordinate RMSE | Selected reference | Reference MSE | Geometry MSE | Relative reduction |
|---|---:|---|---:|---:|---:|
| 3109 | 0.22688 | Current | 0.07268645 | 0.07283373 | −0.20% |
| 3119 | 0.22354 | Expanded current | 0.04690162 | 0.04605442 | +1.81% |
| 3137 | 0.01226 | Expanded current | 0.00045374 | 0.00042947 | +5.35% |

The positive point estimate for model 3137 is a descriptive subgroup observation, not a rescued primary result. Its uncertainty is wide. Neither of the first two geometry observers beat its fit-mean constant baseline on evaluation MSE, which limits their predictive usefulness despite passing the predeclared target-variation gate.

Intervals resample the 300 evaluation scenario IDs jointly across all three fixed models, using 2,000 paired resamples. There are 900 model–scenario outcomes but only 300 distinct paired evaluation scenarios. No model-population or new-environment uncertainty is estimated.

## Did geometry help allocate a fixed planning budget?

At the primary 50% quota, each method continued exactly 150 of 300 cases for each model. Mean total planning budget was four rounds per case: two for everyone and four extra rounds for half. The ranking used each observer's predicted value; true outcomes only evaluated the resulting allocations.

| Allocation method | Mean true executed-plan cost; lower is better |
|---|---:|
| Hindsight oracle, same quota | 0.185931 |
| Geometry observer | 0.226700 |
| Selected reference observer | 0.228034 |
| Uniform random allocation, exact expectation | 0.235330 |

Geometry versus reference improved mean cost by **0.001334**, with paired interval **[−0.002981, +0.005481]**. A useful incremental allocation benefit was therefore not established.

Geometry beat random allocation descriptively and under this conditional interval: +0.008630 cost reduction, interval [+0.002289, +0.015301]. That does not establish that geometry was the useful extra ingredient; the decisive comparison is against a predictor with the current information. The prespecified 25% and 75% quota checks also did not establish an advantage over that reference.

Always stopping cost 0.251152 on average; always continuing cost 0.219507. These use different planning budgets. The oracle uses future outcomes and is an unattainable hindsight benchmark. This experiment uses batch allocation with a fixed quota, not an online stopping controller. Policy intervals resample the observed cost differences with the original batch allocations held fixed; they do not rerank each bootstrap queue.

## A diagnostic worth investigating next

Although the planner's estimated objective never worsened as it retained its incumbent, extra planning **materially increased true cost by more than 0.01** in:

- 32.0% of evaluation cases for model 3109;
- 30.7% for model 3119;
- 0.0% for model 3137.

The two former models also had much higher ten-step prediction error. All three passed the same prospective RMSE<0.25 readiness gate; none was discarded. The gate was permissive, and passing it did not establish that an imagined plan could be trusted for control.

This is an association across three fixed models, not a causal test of model fidelity. A concrete next hypothesis is: **a geometry-based compute controller needs information about the reliability of the world model, not just the shape of its predicted futures.** Apparent improvement inside an imperfect model can differ from improvement in the world.

## Computation accounting

The original run took 0.52–0.92 seconds per model to perform the first two planning rounds over all 1,600 cases, then 0.97–1.66 seconds for the extra four rounds. Extracting geometric summaries took another 0.081–0.091 seconds per model across those cases. These are single-run, batched CPU observations, not a hardware-general performance result.

Seven-repeat median inference for the selected geometry predictors was 0.102–0.111 milliseconds per prepared batch of 300 cases. Selected current references took 0.100–0.116 milliseconds. Those predictor timings exclude obtaining the cached rollouts, constructing added current features and allocating the batch. They must not be presented as end-to-end controller savings. Matching planning-round counts also does not match every feature-acquisition or readout cost.

## Numerical verification and retained limitations

The audit reconstructed all **15 selected predictors' evaluation predictions and selected tuning MSEs exactly**, with maximum absolute error zero. Independent execution of the saved action plans reproduced the true value targets to approximately 4.9×10⁻¹⁵. A separate two-round replay reproduced all current, geometric and raw-rollout features for the first 100 scenarios of each model exactly, without executing the later planning rounds. Paired intervals, allocation quotas and decisions were reconstructed; frozen source hashes remained unchanged.

One derived planning-data NPZ was incomplete on disk. Its original 1,324,075 bytes were preserved, and the same planner was replayed from frozen weights, scenarios and noise to reconstruct the complete 9,819,324-byte file. The entire original file was an exact prefix of the reconstruction; all saved evaluation outcomes were identical. No additional model or observer fit, new scenario or outcome-based selection was performed. The original bytes and recovery record are included.

The random-permutation control allowed a small number of self-matches: 2 of 300 evaluation cases for seed 3109, zero for 3119, and 1 for 3137. Across all partitions/models, there were nine self-matches among 4,800 model–scenario rows. It is therefore not a strict derangement control. A future protocol should specify derangements explicitly. No post-result control refit was added here.

Other limits: one known 2D physical system; in-distribution scenes; three fixed world models with shared training data; substantial fidelity differences; fixed observer budgets with some penalty-boundary selections; fixed candidate counts; soft navigation penalties; and open-loop execution of a selected action sequence. There is no learned latent representation, partial observability, model-error estimator, flow-matching generator, recursive attention mechanism, novel architecture claim or consciousness measurement.

## How to combine the branches without losing interpretability

**Next, specify F1-A1 as a new controlled study, not a repair that silently replaces F1-A0.** Keep F1-A0 and its failed advancement criterion in the record.

1. Establish whether the world model is reliable enough for the value target. Use an exact-model baseline and a predeclared model-error intervention, so fidelity's contribution can be tested rather than inferred from seed differences. Add held-out rollout-error diagnostics or calibrated model uncertainty that can be obtained without revealing future evaluation outcomes.
2. In the same task and outcome, cross the available inputs: current alone; current+planning history; current+future geometry; current+history+geometry. Test added reliability information separately. Preserve stronger current-only/raw-rollout baselines, width/capacity controls, mismatches and temporal-order controls where chronology is claimed.
3. Train a stop/continue or allocation rule on development data, then evaluate it on new scenes and models. Count acquisition, memory and inference cost. Demonstrate improved executed reward at matched total cost, or lower total cost within an explicit reward tolerance.
4. Only then introduce flow matching and recursive attention as separate, removable architectural changes. Their data, training and rollout budgets need matching controls. An attractive combined architecture is a hypothesis until the ablations work.

A possible eventual control signal is:

`estimated value = E[L_world(U_k) − L_world(U_(k+Δ)) | current features, planning history, imagined-future geometry, model reliability] − λ × additional compute cost`.

This is the proposed unifying objective, not a policy already validated by this pilot. It connects memory, present-state evaluation and foresight to the practical decision of when to act and when to keep simulating.

## Relation to prior research

Adaptive imagination already has substantial precedent: Hamrick et al.'s [Metacontrol for Adaptive Imagination-Based Optimization](https://arxiv.org/abs/1705.02670) learns how much and which imagined computation to perform. [DreamerV3](https://arxiv.org/abs/2301.04104) illustrates learning behavior through world-model imagination. Our narrower experimental question concerns the incremental usefulness and cost of particular trajectory representations.

[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) concerns vector fields along probability transport paths. It is a possible future generator intervention here; ordinary physical dynamics and CEM search are not flow matching. The original project proposal supplies motivation, while its uncited performance numbers and speculative wave-function analogies supply no evidence for this pilot.

## Files and reproduction

- `F1_A0_Protocol.md`: the unchanged prospective design.
- `run_f1.py`, `config.json`: frozen experiment source; `run/manifest.json` records source hashes and versions.
- `run/`: model weights, scenarios/splits, complete cached plans/features/targets, selected observer weights, evaluation predictions, candidate selection scores and full results.
- `audit_and_plot.py`, `run/audit.json`: numerical reconstruction and figure generation, without fitting.
- `recover_derived.py`, `run/recovery/`: documented derived-file recovery.
- `F1_A0_Results.png`: result figure; `F1_A0_Study.zip`: portable source and results bundle.

To inspect the existing run after extracting the bundle, install the recorded dependencies and execute `python audit_and_plot.py`. The experiment runner refuses to overwrite an existing run. A reproduction requiring fresh fits should use a separate directory containing the frozen source, config and protocol, preserving the original bundle. No further experiment is running.
