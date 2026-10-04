# P1-A0 — Arithmetic readiness pilot

Research 002 · 15 September 2026 · protocol v1.0.0  
**Outcome: competence gate failed; no history predictors fitted.**

The new present-state arithmetic branch has started. The implementation checks
passed, but this one fixed training recipe did not learn to answer unseen operand
combinations reliably. The prespecified stop was respected. The present-state
sufficiency and practice hypotheses remain **untested**, not refuted or supported.

## What actually ran

- Task: `(a+b+c) mod 17`, with inputs `[CLS, a, b, c]`.
- One newly initialized model, seed 2003; no prior models or reserved seeds used.
- Original T1 NumPy Transformer block reused for six steps; width 24, two heads,
  feed-forward width 48; 5,873 trainable parameters.
- Exactly 1,200 Adam updates, batch size 64, learning rate 0.001, final-only loss.
- 2,992 training examples from 581 canonical families; 950 calibration examples
  from 193 different families. All operand permutations share a partition.
- 195 additional families reserved; no reserved examples or outputs evaluated.
- Local execution took 25.4 seconds, including scheduled
  evaluation and saving; Colab runtime may differ. No interactive Colab session
  was executed here. The delivered notebook is syntax-checked and uses the same
  locally executed code with NumPy only.

## Learning across the fixed checkpoints

| Training updates | Training accuracy | Unseen-combination accuracy |
|---:|---:|---:|
| 0 | 4.91% | 4.84% |
| 100 | 8.92% | 0.95% |
| 400 | 20.35% | 0.84% |
| 1,200 | 68.11% | 3.16% |

The final calibration result was 30 correct answers out of 950, versus the
declared 90% development competence target. Uniform random guessing among 17
classes has expected accuracy 5.88%; this is a mathematical reference, not a
fitted baseline or an independent significance test. Permutations are correlated.

The train/calibration gap is **consistent with fitting training combinations
without learning a reliably generalizing arithmetic rule**. This single short
run does not establish that the architecture cannot learn modular arithmetic,
that longer training would succeed, or that its failures arise from one specific
mechanism. No post-outcome extension or hyperparameter search was run.

## Decision-change target at update 1,200

| Computation step | Changed / total | Flip rate | Distinct families: changed / unchanged | Eligible |
|---:|---:|---:|---:|---|
| 2 | 798 / 950 | 84.00% | 170 / 38 | Yes |
| 3 | 654 / 950 | 68.84% | 145 / 76 | Yes |
| 4 | 475 / 950 | 50.00% | 121 / 118 | Yes |
| 5 | 244 / 950 | 25.68% | 77 / 170 | Yes |

All four candidate steps had enough examples and distinct families in both
classes under the declared anti-degeneracy screen. A family can contribute to
both classes through different permutations; the two family counts are not
disjoint sample sizes. The target gate passed, but competence did not, so readiness
failed. Changes in an unreliable network are not evidence of useful self-monitoring.

## Mathematical and implementation checks

Every method check passed: exact labels, deterministic family assignment, split
separation, unique weight-tied parameters, gradients checked by finite differences,
probability normalization, checkpoint round trips, and observer interfaces that
do not use labels or future states.

At all four saved training checkpoints, resuming from any of steps 0..6 reproduced
the final logits for the 16 audited calibration examples with maximum absolute
difference **0.0** in this runtime. This checks exact computational continuation
from the complete current state. It costs the remaining Transformer computation
and does not show that an inexpensive current-only observer predicts equally well.
It is expected for this deterministic architecture, even before training.

Finite binary truth-table controls gave oracle Brier losses 0.25 for an incomplete
current view and 0.0 when the missing bit was revealed; a complete current view
already had oracle loss 0.0. These are mathematical fixtures, not fitted-probe
power estimates and not natural Transformer-history findings.

The separate verification script recomputed all calibration accuracy/flip metrics
from saved arrays, all checkpoint training accuracies from saved weights, source
identity against the original ZIP, all 1,200 update records, and the stopping
decision. The prospective local protocol/source freeze remained unchanged.

## Scientific boundary and next step

P1-A0 is complete with status `competence_failed`. No history advantage, practical
sufficiency, training-induced sufficiency, consciousness or causal metacognition
claim has been established. Earlier T1/T2 findings remain unchanged.

The next step is a separately versioned, bounded **arithmetic competence design**,
before observer fitting. It must justify any task/training change, record that
this development outcome informed the choice, and set a finite budget and fresh
evaluation plan before fitting. The old reserved seeds remain sealed. This report
does not automatically authorize additional runs or use of held-out families.

## Reproduction and files

`P1_A0_Colab.ipynb` is standalone: upload it into Colab. It contains the source,
protocol and recorded report, runs method checks, and defaults to **not** rerunning
training. Set `RUN_REPRODUCTION = True` only to reproduce the same exposed pilot;
it is not an independent replication and may vary slightly across numerical
library versions. No GPU or external model download is needed.

`P1_A0_Study.zip` contains source, original source archive, the exact protocol,
four trained checkpoints, calibration trajectories, development examples, optimizer
moments, logs, local freeze, verification and this report. Existing run directories
are never overwritten by the experiment runner.

### Sources and provenance

- [Original T1 source](https://drive.google.com/file/d/1116fOMjOUzIfyT-oYoI9gKv1bfoX63g9/view).
- [T2-E2 report](https://drive.google.com/file/d/1DyJAbpnvQXhIr_pqDwGJvQTU13VfypXf/view).
- [Observer design constraints](https://drive.google.com/file/d/1MwJIAc5stQ-6_987_ZPyBaa7L5M4ruLD/view).
- [Grokking study](https://arxiv.org/abs/2201.02177): background on algorithmic
  learning/generalization, not evidence for our new hypothesis.
