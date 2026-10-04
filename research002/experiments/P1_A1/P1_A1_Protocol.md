# P1-A1 — Predicting decision change from the present state and history

Research 002 · 15 September 2026 · version 1.1.0

This prospective development protocol is frozen locally before observer fitting.
It follows the user's authorization to perform the comparison after P1-A0B.
Version 1.0.0 stopped before any natural or synthetic-control observer fit because
of a prevalence eligibility rule. This version is explicitly informed by that
support audit; see `AMENDMENT.md` and the preserved `stopped_v1/` evidence. It makes
one eligibility change and retains every model, step, split, candidate, budget,
outcome metric, uncertainty calculation and interpretation threshold.

## Question and mathematical scope

For each fixed trained network and step t, let S_t be the complete four-token
state, and let Y_t indicate disagreement between the current and final argmax.
With fixed weights and a known remaining step count, S_6 = F^(6-t)(S_t).
Consequently Y_t is already a deterministic function of S_t. This architecture
does not permit a discovery that history adds information absent from the full
state. The empirical question is whether a specified finite predictor can extract
the decision-change target from the current state as effectively and cheaply as
from current state plus history.

The target concerns decision change, not correctness. This study cannot establish
subjective awareness, human mindfulness, universal sufficiency, or a causal change
in history value caused by arithmetic training.

## Fixed models and data

- Use only the final, 6,000-update matrix-decay models with seeds 2017 and 2027.
  No network training, extra model seeds, or earlier checkpoints are permitted.
- Use steps 3 and 4, the common eligible steps selected in P1-A0B. Retain all four
  model/step cells in every result; no favorable-cell selection.
- Reuse the original 581 training families (2,992 ordered examples) and 193
  development evaluation families (950 examples). The 195 reserve families are
  not expanded or evaluated. Earlier T1/T2 reserve seeds remain unused.
- The evaluation families influenced P1-A0B and this design. Evaluation here is
  development work, not untouched confirmation.
- Three observer splits use PCG64 seeds 314159, 314160, and 314161. In each, sort
  the 581 training family IDs, permute them, assign the first 465 to observer fit
  and the last 116 to tuning. All permutations of a triple stay together. Split
  314159 is primary; the other two examine variation from refitting the entire
  observer pipeline. Evaluation always uses the same 193 families.
- Before fitting any natural cell, require at least 50 fit and 20 tuning examples
  in each target class, at least 25 fit and 10 tuning contributing families in
  each class. Record class proportions descriptively; there is no additional
  class-fraction cutoff in this amended development protocol.
  Any failure makes this planned panel ineligible; do not silently drop a cell.

## Observations and six arms

At a single step, the observation contains all 96 state coordinates, 17 current
class probabilities, entropy and top-two probability margin: 115 coordinates.
These are functions of the current state and the already trained shared readout.
History concatenates the same observation at steps 0 through t-1, in order.
No future state, final probability, arithmetic label or target enters a feature.

| Arm | Observation | Purpose |
|---|---|---|
| current | Current observation | Complete-state reference |
| expanded_current | Current plus fixed cosine features of current, equal in number to history coordinates | Match input dimension and coefficient count using current information |
| ordered_history | Current plus this example's chronological history | Primary history comparison |
| residual_history | Current plus history after subtracting a ridge prediction from current | Remove linearly predictable history; diagnostic only |
| mismatched_history | Current plus another example's complete history, from a different family in the same partition | Break current/history correspondence while preserving history's empirical marginal distribution |
| shuffled_history | Current plus this example's past observations in an independently permuted order | Test access to explicit chronological slots |

Fit all centering/scaling using observer-fit families only. Use equal family
weight and equal weight for permutations within a family. Each standardized
block is divided by the square root of its width; concatenate current/history
blocks and divide by sqrt(2). This keeps the two blocks at comparable scale.

Expanded-current features are sqrt(2) cos(Z W + b), with W drawn from
N(0,1/current_width) and b uniform on [0,2 pi], then standardized on fit data.
Residualization fits standardized history from standardized current with ridge
penalty 0.01 and an intercept, using fit data only. It is not nonlinear conditional
independence testing. Linear augmented models retain the same span under this
residualization, although regularization geometry changes.

Mismatching permutes complete histories separately within fit, tuning and
evaluation. Randomize family blocks and rows within blocks, then rotate by the
largest family size; every donor must have a different family ID. Shuffling
permutes past-step slots independently for each example without consulting
labels. Neither transformation is a conditional-randomization test. Shuffled
states may still reveal their computational age.

## Predictor family, selection and finite budget

Every arm gets the same nine candidates: a linear ridge predictor and two
nonlinear Gaussian landmark-feature ridge predictors (64 fit-family landmarks;
bandwidths 0.5 and 2), each with penalties 1, 0.01, 0.0001. The ridge objective is
family-weighted mean squared error plus lambda times squared coefficient norm;
the intercept is unpenalized. Gaussian features are centered/scaled using fit
data and divided by sqrt(64). Clip predictions to [0,1].

Landmarks come from 64 distinct fit families and one random example per selected
family, chosen before target fitting. Use the same landmark row IDs across arms
within a cell. Linear coefficients count is input width plus one; nonlinear
coefficient count is 65, with input-dependent landmark storage and cost also
reported. Equal coefficient count alone does not establish equal computation.

Choose the smallest family-weighted tuning Brier score within each arm; exact
ties use candidate order: linear, bandwidth 0.5, bandwidth 2; within each, strongest
penalty first. Select the current reference between current and expanded_current
using tuning Brier only, with current first on ties. This gives the current
reference 18 candidates versus 9 for ordered history, deliberately providing a
strong current benchmark. Do not refit after selection and do not use evaluation
outcomes for model or threshold choice.

There are exactly 3 splits x 2 models x 2 steps x 6 arms x 9 candidates = 648
natural target fits, plus 432 synthetic-control target fits below. No extensions,
additional hyperparameters, or learned ensembles after seeing results. Synthetic
method fixtures and label-free residualization are recorded separately. A numeric
or implementation failure stops execution and is recorded, rather than replaced
with a favorable run.

## Metrics, uncertainty and interpretation rules

Primary metric: Brier loss, mean (p-Y)^2. First average within canonical families,
then equally across families, steps and the two fixed models. History advantage
Delta = Brier(current reference) - Brier(ordered history); positive favors history.
Report every cell and split, not only the pooled difference.

Use 5,000 paired family bootstrap draws with seed 73001, resampling the same
evaluation family IDs jointly across all cells and observer splits. Report 95%
two-sided percentile intervals and one-sided 95% upper/lower bounds. These are
conditional family-sampling summaries for the fitted pipelines and two fixed
models, not model-population uncertainty, exact tests, or fresh confirmation.

The development noninferiority margin is delta = 0.01 absolute Brier loss. This
is 1% of the maximum possible squared error, chosen prospectively as a small
screening tolerance; it is not one percentage point of accuracy, a deployment
cost threshold, or the earlier study's 5% MSE margin.

The fit-family target prevalence is the constant baseline. A useful current
predictor must improve Brier by at least 10% relative to that baseline and have a
positive one-sided 95% lower bound for its absolute improvement in each primary
cell. Record balanced accuracy and ROC AUC as secondary descriptions only.

Decision hierarchy:

1. A development history advantage requires a primary pooled one-sided 95% lower
   bound for Delta >0.01, positive pooled point effects in each model, a positive
   lower bound for mismatched-history loss minus ordered-history loss, and a
   pooled point effect >0.01 in both other observer splits. This is conditional
   evidence about prediction access, not additional information in the system.
2. Otherwise, bounded practical current-state sufficiency requires all four
   primary cells' one-sided 95% upper bounds for Delta <0.01; a pooled upper bound
   <0.01 in both other observer splits; useful current prediction in every
   primary cell; and passing synthetic sensitivity diagnostics below. A global
   assertion is made only when every required condition passes. No isolated
   favorable interval is enough.
3. Otherwise report inconclusive at these budgets. Weak current predictions,
   wide intervals, instability under refitting, or weak sensitivity prevent a
   positive sufficiency claim. No rule authorizes confirmation automatically.

Measure current and history inference cost and full network continuation on the
same evaluation examples (3 warm-ups, 15 timed repetitions; median time). Feature
transformation is included; existing state/readout acquisition and history
recording are excluded and disclosed. A sufficiency result is described as
computationally cheaper only if the current reference is faster than full
continuation in all four primary cells. Timing is descriptive for this runtime.

## Fitted-pipeline sensitivity and negative controls

Use synthetic seeds 9101, 9102, 9103, 9104 with the same family IDs, primary
fit/tuning/evaluation partition sizes, 115 current and 460 history coordinates.
Generate independent family-level Rademacher signals u and v. Current's first
8 coordinates contain u with independent Gaussian noise SD 0.1; history's first
32 contain v with the same noise. Other coordinates are independent N(0,1).
Generate Bernoulli targets with probability 0.5+0.2u+bv, for b=0,0.1,0.2. Use the
same underlying uniform draws and feature arrays for the three signal levels.
The ideal latent history improvements are 0,0.01,0.04 Brier units; observation
noise and finite fitting can reduce achieved improvements.

Run the actual preprocessing, nine-candidate fitting, tuning selection and
evaluation pipeline for current, expanded_current, ordered_history and
mismatched_history: 4 seeds x 3 levels x 4 arms x 9 = 432 target fits. Do not tune
the method to make these diagnostics pass. Use the same family bootstrap scheme.

Sensitivity passes only if all four strong-signal cases have a positive
one-sided 95% lower bound for Delta and point Delta >=0.01; at least three of
four near-margin cases have a positive lower bound; and no null case has a lower
bound >0.01. These small synthetic panels diagnose recovery of this planted
signal, not general statistical power or sensitivity to every history effect.

## Validation, preservation and next boundary

Before natural fitting, check exact ridge solutions against augmented least
squares, weighted-family calculations, synthetic closed-form metrics, no future
or target leakage in extraction, donor-family exclusion, fit-only preprocessing,
split disjointness and serialization round trips. Recompute saved evaluation
predictions, losses, bootstrap summaries and decisions in a separate audit.
Hash this protocol, execution source and immutable input copies before fitting.
Preserve the earlier stage unchanged and save all selected pipelines, tuning
scores, evaluation predictions, seeds, split maps, timing records and diagnostics.

Any later confirmation needs new explicit rules and untouched evaluation
conditions. This panel keeps confirmation data closed regardless of its result.

## Methodological context

Probe performance depends on the predictor and its ability to learn the task;
control tasks help distinguish these effects. Our synthetic diagnostics adapt
that principle, rather than reproducing the language tasks in
[Hewitt and Liang (2019)](https://aclanthology.org/D19-1275/).
Comparing performance alongside probe complexity is motivated by
[Pimentel et al. (2020)](https://arxiv.org/abs/2010.02180).
The present data split, Brier margin, candidate budget and decision rules are
our own prospective development choices, not prescriptions from those papers.
