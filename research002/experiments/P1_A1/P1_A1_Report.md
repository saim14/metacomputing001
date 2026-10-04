# P1-A1 — Current-state and history prediction

Research 002 · 15 September 2026 · completed development protocol v1.1.0

**Result: the planned criteria for bounded practical current-state sufficiency passed.**
For these two trained arithmetic networks, at steps 3 and 4, the selected
current-state predictors were noninferior to the selected ordered-history
predictors within the declared **0.01 absolute Brier-loss tolerance**. This is a
specific development result about accessible prediction, not subjective awareness.

Mean Brier loss was **0.10416 for current state** and **0.10981
for ordered history**. History advantage Δ = current loss minus history loss was
**-0.00565**, with conditional family-bootstrap
95% interval **[-0.01100, -0.00039]**. Negative values favor current state.
The conclusion uses the predetermined one-sided bounds and all-cell rules below,
rather than a failure to reject equality.

## What this establishes

The complete current state, combined with a modest learned predictor and fixed
nonlinear features of that same state, was enough to forecast whether the model's
current answer would change by its final computation step, to the stated
tolerance against this ordered-history comparator. Current prediction improved
Brier loss by **36.6%–60.6%**
relative to a fit-prevalence constant baseline across the four primary cells.

This does not imply that history is useless. Correctly paired history outperformed
mismatched history on average. It also had **lower inference time than the selected
current-state predictor when history was already cached**. The result supports
accuracy sufficiency at the tested budgets, not global computational optimality.

## Transparent amendment before observer fitting

The first v1.0 attempt stopped before any natural or synthetic-control observer
fit: some step-4 fit/tuning partitions had fewer than 10% decision changes.
Every partition nevertheless passed the separately specified event and family
counts. Version 1.1 removed the additional percentage cutoff and retained the
counts, all models, steps, splits, predictors, budgets, metrics and interpretation
thresholds. The smallest step-4 minority supports were 196 fit examples from
54 families and 37 tuning examples from 11 families.

This amendment was informed by observed support counts. It is preserved in
`AMENDMENT.md`, with the original stopped protocol, code and audit under
`stopped_v1/`. It must not be described as an untouched preregistration. No
prediction outcomes informed the amendment; no extra model or observer budget
was consumed by the stopped attempt beyond the separate method fixtures.

## Fixed experiment

- Arithmetic: `(a+b+c) mod 17`, the six-step weight-tied Transformer from P1-A0B.
  Final regularized model seeds 2017 and 2027; no arithmetic-network retraining.
- Target: current argmax differs from the final argmax, at steps 3 and 4. It is
  not mathematical correctness, conscious experience or metacognitive awareness.
- Current observation: all 96 token-state coordinates, 17 current probabilities,
  entropy and probability margin. History contains the same observations from
  steps 0 through t−1. Future values and targets never enter features.
- Fit/tuning: 465/116 disjoint training families, repeated with three fixed group
  splits. Evaluation: the same 193 families/950 examples from P1-A0B. These are
  exposed development conditions; the 195 reserve families remain unevaluated.
- Six arms, nine candidates each: linear ridge and two Gaussian landmark-feature
  ridge alternatives, each at three penalties. Tune on family-weighted Brier;
  choose the current reference between raw and dimension-matched expanded current.
- Exactly **648 natural** and **432 synthetic-control** candidate fits completed.
  All 120 selected pipelines, 1,080 candidate tuning scores and predictions are
  retained. No selection used the evaluation scores.

All four primary current references selected the expanded-current arm: fixed
cosine features of the present state, with no earlier state as input. Their
linear readouts and ordered-history readouts had matching coefficient counts.
This is a stronger current benchmark than testing a raw-state linear readout alone.
All primary readouts selected the weakest penalty in the fixed grid (0.0001).
The grid therefore does not exclude improvements from still weaker regularization
or other predictor families. No extra candidates were added after this observation.

| Model | Step | Arm | Readout kind | Penalty | Fitted coefficients incl. intercept |
|---|---|---|---|---|---|
| 2017 | 3 | Expanded current | linear | 0.0001 | 461 |
| 2017 | 3 | Ordered history | linear | 0.0001 | 461 |
| 2017 | 4 | Expanded current | linear | 0.0001 | 576 |
| 2017 | 4 | Ordered history | linear | 0.0001 | 576 |
| 2027 | 3 | Expanded current | linear | 0.0001 | 461 |
| 2027 | 3 | Ordered history | linear | 0.0001 | 461 |
| 2027 | 4 | Expanded current | linear | 0.0001 | 576 |
| 2027 | 4 | Ordered history | linear | 0.0001 | 576 |

## Primary results and noninferiority checks

Brier loss is mean squared probability error, with each operand family weighted
equally and each permutation weighted equally within its family. The tolerance
0.01 is a development choice in Brier units, not a percentage point of accuracy.

| Model seed | Step | Current Brier | History Brier | History advantage Δ | One-sided 95% upper bound | Current skill vs constant |
|---|---|---|---|---|---|---|
| 2017 | 3 | 0.12727 | 0.13022 | -0.00295 | +0.00886 | 48.7% |
| 2017 | 4 | 0.06086 | 0.06097 | -0.00011 | +0.00440 | 60.6% |
| 2027 | 3 | 0.13950 | 0.16116 | -0.02167 | -0.00941 | 41.9% |
| 2027 | 4 | 0.08901 | 0.08689 | +0.00212 | +0.00769 | 36.6% |

All four one-sided 95% upper bounds are below 0.01. The largest is
**0.00886**, so the finding is
specific to this tolerance and does not establish arbitrarily close equivalence.
Every current predictor also passed the declared usefulness criterion: at least
10% relative Brier improvement and a positive lower bound against the constant.

| Required condition | Outcome |
|---|---|
| all primary cell upper bounds below margin | Pass |
| other splits pooled upper below margin | Pass |
| all primary current predictors useful | Pass |
| synthetic sensitivity passed | Pass |

## Refitting the entire observer pipeline

The other two family splits refit scaling, residualization, nonlinear features,
candidate coefficients and tuning selection. Their pooled upper bounds also
remained below 0.01. These refits examine pipeline variation; they are not new
independent neural-network seeds.

| Observer split | Pooled Δ | Two-sided 95% interval | One-sided 95% upper bound |
|---|---|---|---|
| 314159 | -0.00565 | [-0.01100, -0.00039] | -0.00114 |
| 314160 | -0.00398 | [-0.01006, +0.00195] | +0.00090 |
| 314161 | -0.00785 | [-0.01354, -0.00235] | -0.00337 |

The bootstrap resamples the same family IDs jointly across models, steps and
observer splits. Intervals are conditional on two fixed networks and exposed
development data; they do not quantify generalization to a population of trained
networks or correct for the research program's prior development choices.

## All six arms, all cells

Every entry is evaluation Brier, lower is better. No favorable cell or seed was
removed. The table is descriptive outside the frozen primary comparison.

| Observer split | Model | Step | Current | Expanded current | Ordered history | Residual history | Mismatched history | Shuffled history |
|---|---|---|---|---|---|---|---|---|
| 314159 | 2017 | 3 | 0.13649 | 0.12727 | 0.13022 | 0.13020 | 0.14589 | 0.13389 |
| 314159 | 2017 | 4 | 0.06392 | 0.06086 | 0.06097 | 0.06134 | 0.07233 | 0.06273 |
| 314159 | 2027 | 3 | 0.16310 | 0.13950 | 0.16116 | 0.17478 | 0.18295 | 0.16299 |
| 314159 | 2027 | 4 | 0.08802 | 0.08901 | 0.08689 | 0.08952 | 0.08546 | 0.09109 |
| 314160 | 2017 | 3 | 0.13856 | 0.13345 | 0.13301 | 0.13515 | 0.14721 | 0.13726 |
| 314160 | 2017 | 4 | 0.06488 | 0.06213 | 0.06525 | 0.06962 | 0.07241 | 0.06720 |
| 314160 | 2027 | 3 | 0.16934 | 0.15586 | 0.17037 | 0.18307 | 0.17566 | 0.17438 |
| 314160 | 2027 | 4 | 0.08594 | 0.08408 | 0.08469 | 0.08670 | 0.09430 | 0.08483 |
| 314161 | 2017 | 3 | 0.14163 | 0.11673 | 0.13513 | 0.13276 | 0.14687 | 0.13927 |
| 314161 | 2017 | 4 | 0.06636 | 0.06830 | 0.06768 | 0.06675 | 0.07320 | 0.06862 |
| 314161 | 2027 | 3 | 0.16917 | 0.15529 | 0.16539 | 0.17403 | 0.17343 | 0.16908 |
| 314161 | 2027 | 4 | 0.08332 | 0.08676 | 0.08490 | 0.08866 | 0.08477 | 0.08570 |

The primary pooled mismatched-minus-ordered loss was
**+0.01185**, with a positive
one-sided lower bound **+0.00696**.
Thus correspondence between present and past did matter to the history probes.
For seed 2027 at step 4, mismatched history had a slightly better point score;
the paired effect's interval included zero. That exception remains visible.

Residual history is a linear-residualization diagnostic, not proof of conditional
information beyond the present. Shuffling removes explicit slots but may leave
the computation stage inferable from state values. Neither control establishes a
general absence of useful historical structure.

## Sensitivity through the fitted pipeline

Synthetic controls use family-level current and history signals, noisy observed
features, and Bernoulli targets with known latent history effects. The same
preprocessing, candidate fitting and tuning selection used in the actual study
recovered **4/4 strong effects**, **4/4 near-margin effects**, and produced
**0/4 material false positives in null cases** under the frozen rules.

| Synthetic seed | Ideal latent gain | Observed Δ | One-sided 95% lower bound |
|---|---|---|---|
| 9101 | 0.00 | +0.00088 | -0.00050 |
| 9101 | 0.01 | +0.00535 | +0.00042 |
| 9101 | 0.04 | +0.02900 | +0.01976 |
| 9102 | 0.00 | -0.00012 | -0.00175 |
| 9102 | 0.01 | +0.00901 | +0.00489 |
| 9102 | 0.04 | +0.04191 | +0.03334 |
| 9103 | 0.00 | -0.00137 | -0.00456 |
| 9103 | 0.01 | +0.01097 | +0.00609 |
| 9103 | 0.04 | +0.04076 | +0.03162 |
| 9104 | 0.00 | -0.00121 | -0.00288 |
| 9104 | 0.01 | +0.01213 | +0.00732 |
| 9104 | 0.04 | +0.03792 | +0.02854 |

These are twelve diagnostic datasets, not a power study. Success for this planted
signal does not guarantee detection of every nonlinear or distributed history
effect. It does provide a check that the actual fitted pipeline can register a
history contribution at approximately the declared tolerance in this setting.

## Computation and memory tradeoff

Median milliseconds for the same 950 evaluation examples, after three warm-ups
and over 15 timed repetitions:

| Model | Step | Current ms | History ms | Full continuation ms | Current / history time |
|---|---|---|---|---|---|
| 2017 | 3 | 12.63 | 3.14 | 108.07 | 4.03× |
| 2017 | 4 | 17.18 | 3.69 | 80.25 | 4.66× |
| 2027 | 3 | 12.80 | 3.28 | 114.91 | 3.90× |
| 2027 | 4 | 17.49 | 4.14 | 79.10 | 4.23× |

The selected current predictor was **4.5–9.0 times faster
than full network continuation**, but slower than its ordered-history counterpart.
Feature transformations are included. Acquisition of existing state/readout and
recording of history are excluded. Therefore this timing comparison assumes
history is available already; it is not an end-to-end storage/latency benchmark.

The current observation has 115 coordinates. Ordered history adds 345 coordinates
at step 3 or 460 at step 4. Expanded current computes an equal number of nonlinear
features from current data, trading extra computation for avoiding those earlier
observations. Constant model parameters and stored random-feature matrices have
their own memory cost. No fastest or smallest implementation is established.

## Mathematical and philosophical boundary

With fixed weights and a known step count, the complete current state determines
the final state: `S_6 = F^(6−t)(S_t)`. Exact continuation is therefore a property
of this architecture. The empirical result adds that a much less expensive
predictor can extract a useful decision-change estimate from that state, with
the stated tolerance against a bounded history predictor.

The complete state may already carry effects of prior computation and of
arithmetic training. "Current-only" does not mean free of memory or prior causes.
The data support a limited statement: historical influence need not be supplied
again as a separate trajectory for this prediction task. They do not establish
present-moment consciousness, human mindfulness effects, or awareness in a
mathematical neural network. No untrained-versus-trained observer comparison was
performed, so training-induced changes in sufficiency remain untested.

## Verification and reproducibility

The audit reconstructed every selected evaluation and tuning prediction from
saved pipelines, checked all 1,080 tuning scores and selection rules, verified
selected ridge solutions by their first-order optimality equations, regenerated
all neural trajectories, and independently reconstructed family metrics,
bootstrap intervals and decision conditions. Maximum prediction discrepancy:
**0**; maximum ridge stationarity residual:
**4.55e-16**. Frozen inputs and the
stopped v1.0 evidence remained unchanged. This is a numerical audit, not an
independent replication.

One saved intermediate training-trajectory cache was truncated. The damaged
bytes are preserved, and pure forward computation from the frozen weights
reconstructed the cache. The entire damaged file matched the exact byte prefix
of the reconstruction. The audit then verified the reconstructed trajectory and
every saved predictor output; no observer fitting or network training was added.
The cause of truncation is not established. The recovery record and original
bytes are included under `panel_run/recovery/`.

Execution: Python 3.12.14, NumPy
2.3.5, CPU. The amended panel took
**37.3 seconds** in this runtime, excluding design,
artifact preparation and the separate numerical audit. Exact wall time and
floating-point behavior can differ elsewhere.

`P1_A1_Study.zip` includes the unchanged trained model inputs, both protocol
versions and amendment, execution and verification scripts, selected pipelines,
all tuning/evaluation predictions, control maps, trajectories and a SHA-256
content manifest. See `README.md` for verification and optional reproduction.
No Colab or external compute service is required.

## Next research boundary

The next stage should be a separately frozen confirmation design with new model
seeds and evaluation conditions that have not informed this development. Define
the accuracy/storage/latency tradeoff explicitly, since history was faster when
cached. Review the penalty-grid boundary using development conditions before
opening confirmation. Keep the current nonlinear benchmark, all model/step outcomes, and the
same Brier tolerance unless a new tolerance is justified prospectively. Confirm
adequate class support before designating confirmation data; do not tune to its
prediction outcomes. No confirmation evaluation has begun.

## Methodological context

Control tasks help distinguish representation access from what the probe learns;
this motivates our fitted synthetic diagnostics.
[Hewitt and Liang (2019)](https://aclanthology.org/D19-1275/).
Probe complexity and predictive performance should be considered together; this
motivates reporting coefficient counts and measured time.
[Pimentel et al. (2020)](https://arxiv.org/abs/2010.02180).
The data splits, margin and decision rules here are our development choices.
