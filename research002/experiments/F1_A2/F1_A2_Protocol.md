# F1-A2 — Decision-specific model bias and the value of computation

Research 002 · Saim 13.02 · 18 September 2026 · version 1.0

Prospective development protocol. Freeze this document, configuration, source
and inherited checkpoint hashes before generating new outcomes or fitting
observers. F1-A1 informed this design; fresh samples do not make it an untouched
confirmation. Preserve F1-A0/F1-A1 results. No sealed T1 seed or P1 reserve family
is used. Stop after this fixed run, audit and report.

## Question and scope

F1-A1 established a controlled model-error effect, but history, imagined-future
geometry and a one-step error-magnitude descriptor did not meet prediction or
control criteria. F1-A2 asks whether cost-bias information on the **already
available round-3 candidate set** is useful, whether an accessible estimate of
that information is useful, and whether any gain survives its computation cost.

The privileged signal below is a diagnostic, not a deployable sensor or a
guaranteed successful positive control. Neither signal may use round-4-to-6
candidates, U6, future true costs or continuation labels. An oracle using the
true final continuation benefit is retained only as a hindsight bound.

One world, three frozen neural checkpoints, seven transition conditions,
2,400 fresh scenes, fifteen observer inputs, twelve candidates per
input/condition: **1,260 observer candidates, 105 selected pipelines, zero neural
retraining**. Conditional local regression for the error estimate is a fixed
algorithm; no alternative neighbor count, regularizer or estimator is searched.

## Inherited task and new samples

Use F1-A1's 2D simulator, ten-action cost, six-round cross-entropy planner,
32 candidates, eight elites, incumbent retention, initialization and noise
distribution. Observe after round 3 and predict signed
`V3 = L_true(U3) - L_true(U6)`. Harm is V3 < -0.01. Retain negative values.

Use one exact transition condition and half/full error from each frozen neural
checkpoint 3109, 3119 and 3137: `f_alpha = f + alpha*(f_m-f)`.
The interpolation is a laboratory intervention. Only full neural conditions
could be available without knowing the true simulator in deployment.

Generate scenes with seed 62001 and whole-scene split permutation 62002:
1,200 fit, 400 tuning, 800 evaluation. All variants of a scene stay together.
Use shared six-round planning noise seed 63001. Use no prior study scene
outcomes for training. All seven conditions remain visible; primary prediction
and controller results equally weight the six imperfect variants.

## Matched cost-bias signals P and B

At round 3 cache the original 32 action sequences, their original model costs
`q_j`, and the incumbent index `i = argmin_j q_j` (stable tie breaking).
Original planning and U3/U6 are never changed by a bias signal.

**P, privileged:** replay only these cached action sequences in the true
simulator and compute true costs `t_j`. Set `b_j = t_j - q_j` and summarize
the fixed set as the twelve coordinates below. This requires 320 additional
true transition evaluations per scene, is unavailable to a deployed controller,
and includes information about a component of the eventual V3 target.

**B, estimated:** acquire 2,048 independent calibration inputs with seed 61001:
positions uniform in [-4,4]^2 and actions uniform in [-1,1]^2. Store observed
true next states and signed transition residuals `f(x,a)-f_alpha(x,a)`.
Use the same bank for the old magnitude descriptor R. No scene outcome enters
this bank and no evaluation observation updates it.

For a queried state/action, scale position coordinates by 4 and action
coordinates by 1. Retrieve 16 nearest calibration inputs. Center their scaled
coordinates at the query and divide offsets by the 16th-neighbor distance
(floor 1e-9). With design `[1, offset_1,...,offset_4]`, weight neighbor j by
`exp(-0.5*(distance_j/radius)^2)` and solve weighted least squares, adding
0.001 times the identity on the four slopes and **no intercept penalty**.
The fitted intercept is the signed two-coordinate transition correction.

Replay the same cached action sequences with
`f_corrected(x,a) = f_alpha(x,a) + estimated_residual(x,a)` and obtain corrected
costs `t_hat_j`. The planner itself keeps using f_alpha. B applies exactly the
same twelve summaries as P, replacing t with t_hat. This adds 320 model
transition predictions, 320 neighbor queries and 320 small local regressions
per scene. It is model-error propagation through the task cost, not training a
world model with a value-aware loss. No test-time true transitions are used by B.

For either t or t_hat, define b = corrected cost minus original q. Summaries:

1. Incumbent bias b_i.
2. Mean b over all candidates.
3. Standard deviation of b.
4. Minimum b.
5. Maximum b.
6. Mean b among eight model-ranked elites minus b_i.
7. Mean b among the remaining candidates minus b_i.
8. Slope of b on q, `mean((q-mean(q))*(b-mean(b)))/(var(q)+1e-12)`.
9. Correlation of q with corrected costs, denominator floor 1e-12.
10. Fraction of strictly discordant candidate pairs (ties do not count).
11. Corrected incumbent cost minus minimum corrected cost among this set.
12. Fraction of candidates improving corrected incumbent cost by more than 0.01.

The incumbent-only controls below distinguish richer candidate-set information
from simply revealing or estimating L_true(U3), a component of V3. A privileged
gain alone cannot establish foresight about U6 or an independent causal role for
geometry. The bank, neighborhood support and cost-estimation errors are saved.

## Inputs, observers and references

C (100), H (200), G (12), R (4) and raw round-3 paths (704) are as in F1-A1.
Let X = [C,H,G,R], width 316. All inputs round to float32, then fit in float64.

| Arm | Input |
| --- | --- |
| C, CR | Current, with/without magnitude descriptor |
| C-expanded, CR-expanded | Add 12 fixed cosine coordinates to fit-standardized C or CR (seed 64001) |
| C-raw, CR-raw | All cached candidate positions, with/without R |
| X | C + H + G + R, the earlier combined representation |
| CP, CB | C plus privileged or estimated cost-bias summary |
| XP, XB | X plus privileged or estimated summary |
| C-mismatched-P, C-mismatched-B | C plus an unrelated scene's complete bias summary |
| CP-incumbent, CB-incumbent | C plus only summary coordinate 1 |

P/B mismatches are strict derangements within each split and condition. Generate
P then B mappings with seed 64002 plus condition index, by random ordering and
a one-position cyclic shift. No self-pair or cross-split pair is allowed.

For every arm/condition, use the unchanged F1-A1 observer recipe: fit-standardized
linear ridge or 128 fixed random Fourier features at bandwidth 0.5/2; penalties
0.0001, 0.01, 1, 100. Fourier seed remains 55001. Fit normalizers/intercept on fit
scenes only; select minimum tuning MSE, ties by listed order. Do not clip outputs.
All candidate coefficients and scores are retained. No observer grid expansion
after seeing outcomes, even if a boundary is selected.

The **strong observed reference** is chosen on tuning MSE among C, CR,
C-expanded, CR-expanded, C-raw, CR-raw and X. This includes the previous
history/geometry/reliability combination. Select a controller reference
separately on tuning charged objective from the same seven arms using their
MSE-selected predictors. Neither reference can access P/B. Larger selection
budgets for the reference are intentional; equal width is not equal capacity.

## Three headline endpoints and interpretation

Use 4,000 paired scene bootstrap resamples, seed 66001. Resample the 800 scene
IDs jointly across conditions/arms; never treat variant rows as independent.
Use **98.333333% two-sided percentile intervals** for three headline endpoints
(tail probability 0.05/(2*3), nominal Bonferroni familywise allocation). Other
95% intervals are descriptive. Coverage is approximate and conditional on
fixed neural checkpoints, fitted observers and this scene distribution.

1. **Privileged signal:** CP versus strong observed reference, pooled imperfect
   MSE reduction. Require >=5% relative reduction and positive adjusted lower
   bound. Require a positive descriptive lower bound versus mismatched P to
   attribute a gain to correctly paired P. A claim that the richer candidate
   set adds beyond incumbent cost additionally requires positive 95% lower
   bound versus CP-incumbent.
2. **Accessible estimate:** CB versus the same reference, >=5% relative MSE
   reduction and positive adjusted lower bound. Require positive descriptive
   lower bound versus mismatched B for paired-signal attribution; compare
   CB-incumbent separately before attributing benefit to the full set.
3. **Practical control proxy:** CB versus the tuning-selected controller
   reference, >=1% relative reduction in the charged objective below and
   positive adjusted lower bound. A wall-clock saving claim additionally
   requires CB's median runtime below its reference in every imperfect
   condition; otherwise report the timing limitation.

Report XP-vs-CP, XB-vs-CB, CB-vs-CR and XB-vs-X comparisons descriptively, with no
rescue of failed headlines or assertion that all historical/geometric components
contributed. No post-result arm becomes the primary candidate. Describe bias
estimation with all-candidate and incumbent cost-bias MSE against a zero-bias
baseline, descriptor-wise RMSE, ranking-discordance agreement and corrected
rollout RMSE on a separate 256-sequence diagnostic set (seed 61002).

A positive privileged result plus negative estimated result suggests a gap
between accessible estimates and useful privileged information under this
readout. It does not by itself locate the gap uniquely. A negative privileged
result does not prove there is no decision-relevant signal: summaries, readouts,
sample size and future randomness remain limitations. A successful B does not
establish novelty or generalization outside this task.

## Decisions and compute accounting

Per-case decision: continue after round 3 iff predicted V3 > 0.01. No batch quota.
Primary objective: `true executed-plan cost + 0.01*continue + feature charge`.
For CB and XB, feature charge is **0.01*(320/960) = 0.0033333333** per case,
representing their 320 extra model transition calls relative to 960 calls in
three more CEM rounds. For CB-incumbent, the analogous charge uses ten calls.
The reference arms use cached rollouts and carry no additional model-call
charge. Neighbor search, local solves, other features and readout are excluded
from this call-count proxy, and measured in runtime. This differs from F1-A1's
proxy, which did not charge descriptor model calls.

Privileged CP/XP scores are diagnostics with their 320 true queries disclosed;
they are never described as deployable controllers or given an efficiency claim.
Always-stop, always-continue and the hindsight true-V oracle remain visible.
Prespecified charges 0, 0.005, 0.02 and 0.05 scale both continuation and model-call
feature charge proportionally; reference identities stay fixed at primary-charge
tuning and sensitivities never replace the primary endpoint.

Runtime: first 24 evaluation IDs in frozen order, one warm-up per method,
three timed repetitions rotating order. Benchmark CB, XB, selected reference,
always-stop and always-continue, on a single CPU thread in all seven conditions.
Include actual conditional planning, required history/geometry/R work, corrected
candidate rollout/local regression, transformations and readout. Exclude initial
bank acquisition/index construction, parameter loading, true outcome scoring and
serialization. Verify decisions/plans against offline results. Report calibration
data and one-time costs separately. No online true simulator call is permitted
for bias estimation B (apart from the laboratory alpha controls inside f_alpha).

## Verification and stop rule

Before outcomes, run fixed synthetic fixtures for candidate summaries, a known
linear transition residual, prefix-step equality to F1-A1, strict derangements
and the call-charge controller arithmetic. These are not pilots on study targets.
Freeze all inputs in a manifest before generation/fitting. After outcomes,
reconstruct every tuning score and selected prediction without refits, replay
true costs independently, reconstruct prefix features without future rounds,
verify paired inference, mismatch maps, fit-only normalization and source hashes.

Save all outcomes including failures, selected penalty boundaries and latency.
Any implementation fault must be disclosed. No outcome-driven redesign, second
fit run or extra model training is part of F1-A2. Report and stop after audit.

## Related work

Task utility rather than state-prediction accuracy alone is an established
motivation in [Grimm et al., The Value Equivalence Principle](https://arxiv.org/abs/2011.03506)
and [Voelcker et al., Value Gradient weighted Model-Based Reinforcement Learning](https://arxiv.org/abs/2204.01464).
F1-A2 tests descriptors for metacomputation in this project's toy planner; it is
not an implementation of their learning objectives. There is no transformer,
flow matching, latent manifold or subjective-awareness measurement in this study.
