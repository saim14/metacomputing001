# F1-A0 — Foresight geometry and the value of computation

Research 002 · Saim 13.02 · 17 September 2026 · version 1.0

Status at writing: prospective development pilot. This protocol and executable are hashed before the first training run. This is not an external preregistration or a confirmatory experiment. No P1 reserve family or T1 sealed seed is used.

## Research question

Can geometric summaries of already imagined world trajectories help a bounded observer predict when additional planning will improve the executed plan, beyond declared current planner features, dimension-matched nonlinear features, and raw imagined trajectories?

The practical link to metacomputation is **estimating the value of additional computation**. The original history and present-state results are preserved as separate evidence. A result here cannot confirm their different architectures, targets or thresholds.

Working interpretation of the requested three angles:

1. **Computational past:** how the internal solution has evolved (Research 001; T1/T2).
2. **Computational present:** what can be predicted from the current working state (P1).
3. **Imagined futures:** whether the geometry of possible world trajectories helps allocate planning (F1, opened here).

The original Geometry of Foresight proposal combines flow matching, recursive future reasoning and dynamical attention. This first experiment isolates the predictive-value component with an ordinary neural transition model. It does not yet implement flow matching, a learned latent geometry, recursive attention or a quantum model. In particular, flow-matching transport time, simulated world time, and planning-iteration index are different quantities.

## Two levels of prediction

The world model predicts a physical transition: `x_(h+1) = f_theta(x_h, a_h)`.

The metacomputation observer predicts a change in plan quality from additional internal planning:

`V_2 = L_true(U_2) - L_true(U_6)`.

`U_k` is the incumbent ten-action plan after planning iteration k. `L_true` executes that whole plan in the known simulator. Positive V means that continuing from two to six planning rounds improved actual simulated plan quality; negative values are retained. This is neither final-answer agreement nor regret against a globally optimal policy. The true simulator is used for training data and outcome measurement, never as an observer feature or as the planner's transition model.

## Fixed toy environment and learned model

- Fully observed 2D point dynamics, time step 0.25:
  `x' = x + .25 * (a + [.25 sin(1.8 y) + .08 x, .20 sin(1.5 x) - .10 y])`.
- Two bounded action components, each in [-1, 1]; planning horizon 10.
- 8,000 independent transition-training samples: positions uniform in [-4,4]^2 and actions in [-1,1]^2, data seed 41001. A separate 1,024-transition diagnostic and 256 ten-step random-action rollouts use seed 41002.
- Three independently initialized MLP dynamics models: seeds 3109, 3119, 3137. Input position and action (4 coordinates); output displacement (2 coordinates). Two tanh layers, 32 units each. Fit-only standardization. Adam, learning rate .003, batch 256, alpha .0001, exactly 180 epochs, no early stopping or model selection. Models share the physics training data; these are three fixed model instances, not three independent environments.
- Before any observer fitting, every model must have diagnostic ten-step coordinate RMSE below .25. Otherwise stop the observer stage and report the failed model(s); do not select passing models or retrain.

## Scenarios, cost and planner

1,600 independently sampled scenarios (seed 42001): start and goal uniform in [-1.25,1.25]^2, separated by at least .75. Circular obstacle centered at their midpoint plus independent N(0,.35^2) noise; radius uniform in [.15,.35]. Reject scenarios with either endpoint within radius+.08 of the obstacle. These are in-distribution development cases, not unseen task families.

The entire scenario is assigned once, by permutation seed 42002, to 1,000 observer-fit, 300 tuning, or 300 evaluation cases. The same assignment and scenario identities apply to every world model and feature arm. No episode crosses partitions.

Plan cost, with h=1,...,10 and r the obstacle radius:

`L(U) = ||x_10-goal||^2 + .03 mean_h ||x_h-goal||^2 + 12 mean_h [max(r-||x_h-obstacle||,0)/r]^2 + .01 mean_h ||a_h||^2`.

The planner uses its learned dynamics for this cost. A continuous collision penalty permits the toy optimizer to trade competing costs; this is not a safety-certified navigation controller.

Cross-entropy planning: six rounds, 32 candidate action sequences per round, eight elites. Initial mean is the constant straight-to-goal action, clipped to [-.8,.8]; initial SD .65. Subsequent mean/SD are elite moments, with SD floor .08. Candidate 0 repeats the incumbent, candidate 1 uses the current mean; others use clipped Gaussian samples. The incumbent must never worsen on the learned-model objective. Planning noise is fixed by seed 43001 and shared across world models. No outcome-dependent restarts or extra rounds.

## Inputs available at the decision point

After round 2, before rounds 3–6:

- **C, current planner features:** start, goal, obstacle center/radius; next-round action means/SDs; incumbent actions and estimated cost; sorted costs of the 32 current candidates. This is a declared observable set, not a claim to expose the entire computational state or RNG.
- **C-expanded:** C plus 12 fixed cosine features of fit-standardized C; width matches C+geometry. Fixed feature seed 44001.
- **C+geometry:** C plus 12 summaries from the 32 candidate trajectories already evaluated in round 2: mean/SD path length; mean/SD average turning angle; mean/SD goal distance at the endpoint; minimum and mean minimum obstacle clearance; fraction of candidate paths intersecting the obstacle; RMS endpoint dispersion; mean/SD displacement-to-path-length ratio. Geometry is in physical coordinates with declared Euclidean units, not an arbitrary learned latent metric.
- **C+raw-rollouts:** C plus every position in those same 32 trajectories. This stronger reference preserves the underlying observable data used by the geometric summaries.
- **C+mismatched-geometry:** C plus the geometry rows of other scenarios, independently permuted within each partition. No cross-partition borrowing. Fixed seed 44002 plus model index.

Geometry is a deterministic transformation of cached rollouts. A benefit can establish accessible structure under a predictor budget; it cannot establish information beyond complete current computation or prove that history is unnecessary. Global rigid-transform invariance of the geometric summaries is checked on a method fixture; general latent-coordinate invariance is not established.

## Fixed observers and selection budget

For each of five arms and three world models, use exactly nine candidates: standardized linear ridge, or 128 fixed Gaussian random Fourier features at bandwidth factors .5 or 2, each with penalties .0001, .01 or 1. The fit objective is mean squared error plus alpha times coefficient squared norm. Fit input normalization, feature normalization and intercepts only on fit cases. Nonlinear feature random seed 45001 is shared where dimensions match. Select the candidate minimizing tuning MSE, with listed order breaking ties. Predictions are not clipped.

Each arm receives nine candidate fits (135 total). A **strong reference** is selected on tuning MSE among C, C-expanded and C+raw-rollouts; it therefore has a deliberately conservative 27-candidate combined selection budget. All arm results and widths are reported. Equal input width alone is not asserted to mean equal effective capacity or compute.

Before observers, each model's fit/tuning partitions must each have V standard deviation at least .01 and at least 25 examples with V>.01. Otherwise report an uninformative-target pilot and stop, retaining every model. Evaluation targets do not select eligible examples or hyperparameters.

## Outcomes and interpretation rule

Primary outcome: mean evaluation MSE reduction of C+geometry against the tuning-selected strong reference, with equal weight per scenario and fixed world model. The relative reduction divides the pooled MSE difference by pooled reference MSE.

Uncertainty: 2,000 paired bootstrap resamples (seed 46001) of the 300 evaluation scenario IDs, preserving pairing across arms and all three world models. Two-sided percentile 95% intervals are conditional on these fixed models and the development design. Per-model results are always retained.

Secondary practical test: allocate additional rounds to the top 50% of evaluation scenarios ranked by predicted V, separately per model. Every method gets exactly the same number of extra planning rounds. This is **batch allocation with a fixed quota**, not an online stopping policy. Compare realized true cost against the strong reference and random allocation at the same quota. Report oracle ranking by realized V as an unattainable hindsight benchmark. Preplanned 25% and 75% quotas are descriptive sensitivity results, not alternative primary outcomes.

Report trajectory-generation and feature/predictor time separately; no wall-clock savings claim follows from matching iteration counts. More computation can worsen true cost when the world model is wrong.

An encouraging pilot requires all of: at least 5% pooled relative MSE reduction against the strong reference; positive lower paired interval for its absolute MSE reduction; positive point improvement for each of three fixed models; positive lower interval versus mismatched geometry; and positive lower paired interval for batch-allocation cost improvement at the 50% quota. The 5% value is a prospective screening convention for this toy pilot, not a user-derived economic threshold. Failure means the combined advancement criterion was not established here, not universal falsification. This task authorizes one frozen pilot only; no extra models, feature searches or outcome-driven retuning.

## Combination plan after separate evaluation

A later F1-A1 study can add history of planning revisions and run the factorial ablation C, C+past, C+future-geometry, C+past+future-geometry in the **same** world and target. Include stronger current-only controls, mismatched/order controls, fixed compute and acquisition budgets, model error, negative-value cases, and an online stop/continue policy with measured net cost. The three existing branches cannot be combined by pooling their incompatible metrics.

Flow matching becomes a separate architectural intervention only after the value-of-computation target works: match data, sample count, training and inference cost against ordinary learned dynamics. Recursive attention is another intervention, with a finite predeclared budget. Introduce each mechanism so its contribution can be removed and tested.

## Research grounding and boundaries

- The saved *Wave_Function_of_Foresight.pdf* (17 June 2026) supplies project motivation, not verified experimental evidence. Its uncited performance percentages, quantum analogy and sweeping method comparisons are not carried forward as findings.
- Hamrick et al., [Metacontrol for Adaptive Imagination-Based Optimization](https://arxiv.org/abs/1705.02670), already studies choosing the amount/type of imagined computation. Adaptive imagination itself is not a novelty claim here.
- Hafner et al., [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104), provides established context for learning behavior through imagined scenarios; the present toy is not a Dreamer implementation or benchmark.
- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), defines probability-path vector-field learning. A physical drift model or CEM optimization does not become flow matching merely because trajectories are drawn.

No consciousness measurement, biological claim, general intelligence result or general world-model improvement is implied by this protocol.
