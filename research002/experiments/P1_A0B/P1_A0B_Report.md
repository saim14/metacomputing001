# P1-A0B — Arithmetic competence revision

Research 002 · 15 September 2026 · protocol v1.0.0  
**Development candidate: matrix_decay; observer fitting still unopened.**

## What was completed

Four fixed runs tested 6,000 updates with and without matrix-only decoupled weight
decay, for new model seeds 2017 and 2027. All 24,000 updates were completed; no
extra seeds, extensions or best-checkpoint selection were added. Initial weights
and the entire training batch stream were identical within each seed pair.

The task, split and model were unchanged from P1-A0: `(a+b+c) mod 17`, a six-step
weight-tied Transformer, width 24, two heads, feed-forward width 48 and 5,873 trainable
parameters. Both arms used final-only cross entropy, batch size 64, learning rate
0.001, Adam beta1=0.9/beta2=0.999 and gradient clipping at 1.0. Decay was either
zero or 1 on matrices only; biases and LayerNorm vectors were excluded. At learning
rate 0.001, decay 1 multiplies each matrix by 0.999 before the adaptive update.

P1-A0's observed train/calibration gap motivated this development revision.
Longer training and weight decay have precedent in small algorithmic studies,
but that literature uses different architectures and much larger budgets; it
does not guarantee success in our setting. [Grokking](https://arxiv.org/abs/2201.02177),
[decoupled weight decay](https://arxiv.org/abs/1711.05101).

## Final results at the fixed 6,000-update endpoint

| Seed | Regimen | Final training accuracy | Final calibration accuracy | Eligible steps | Competence >=90% |
|---:|---|---:|---:|---|---|
| 2017 | no_decay | 97.73% | 0.63% | 2, 3, 4, 5 | No |
| 2017 | matrix_decay | 99.93% | 99.58% | 2, 3, 4 | Yes |
| 2027 | no_decay | 98.46% | 1.26% | 2, 3, 4, 5 | No |
| 2027 | matrix_decay | 98.40% | 98.00% | 3, 4 | Yes |

The paired accuracy differences, decay minus no-decay, were seed 2017: +98.95 percentage points; seed 2027: +96.74 percentage points.
The arithmetic mean was **+97.84 percentage points**.
These are descriptive differences for two seeds on one exposed development split,
not population estimates or independently confirmed effects. Matrix-decay versus
no-decay is the controlled within-seed contrast; comparisons with P1-A0's seed 2003
are not controlled optimizer comparisons.

## Calibration learning curves, without best-checkpoint selection

| Seed | Regimen | Update 0 | Update 1,200 | Update 3,000 | Update 6,000 |
|---:|---|---:|---:|---:|---:|
| 2017 | no_decay | 5.79% | 1.05% | 2.42% | 0.63% |
| 2017 | matrix_decay | 5.79% | 28.74% | 94.84% | 99.58% |
| 2027 | no_decay | 6.21% | 2.00% | 1.05% | 1.26% |
| 2027 | matrix_decay | 6.21% | 4.84% | 89.37% | 98.00% |

Only update 6,000 determined readiness. Training and calibration accuracies for
every computation step and checkpoint are saved in the numeric outputs. A high
training score cannot substitute for performance on unseen operand combinations.
Uniform random guessing has expected accuracy 1/17=5.88%; this is a mathematical
reference, not a fitted baseline or a significance test.

## Joint readiness decision

| Regimen | Both seeds competent? | Common eligible steps | Ready for observer design? |
|---|---|---|---|
| no_decay | No | 2, 3, 4, 5 | No |
| matrix_decay | Yes | 3, 4 | Yes |

Competence requires final calibration accuracy >=90% in **both** seeds. Target
readiness requires at least two common steps among 2..5, each with >=50 examples
and >=25 distinct families in both decision-change classes, and each class at
least 10% of the examples. These are development screening rules, not an observer
power analysis. The decision target is disagreement of current and final argmax,
not mathematical correctness, controller utility, or consciousness.

The selection rule prefers no-decay only if both regimens pass. No single model
or favorable checkpoint can override the all-seed gate. Current status:
`candidate_ready_for_observer_design`.

## Data and interpretation limits

- The models trained on 2,992 examples from 581 canonical operand families.
- Calibration used 950 examples from 193 disjoint families; every permutation of
  an operand triple remains in the same partition.
- These calibration examples were already exposed in P1-A0 and influenced this
  research direction. They remain unseen to gradient-based training, but are
  **development data**, not fresh confirmation data.
- The 195 reserved families were not expanded into examples or evaluated. Old
  reserved model seeds 1103/1201/1301 and all closed T1/T2 weights were untouched.
- A family can contribute to both decision-change classes through different
  permutations. Permutations and computation steps are not independent models.
- This tests only same-modulus, same-length operand combinations. No claim of
  universal arithmetic ability or length extrapolation is established.

No history predictors were fitted. Therefore neither practical present-state
sufficiency nor a training-induced reduction in history advantage was tested.
Exact continuation from a complete state is an architectural property, not a
finding of subjective awareness or inexpensive predictability.

## Verification and provenance

Inherited method checks passed, together with exact zero-decay equivalence to the
original Adam, synthetic closed-form decay/moment updates, the matrix/vector mask,
pairing of initializations and batches, and the all-seed selection logic. The
original Transformer, autodiff and P1-A0 implementation were preserved unchanged.

The separate verification script recomputed all training and calibration logits
for all 16 saved checkpoints from the stored weights and inputs. Maximum absolute
logit difference was **0**. It also rechecked every accuracy,
flip count, family count, gate, paired batch streams and initial weights, and the
final decision. All 24,000 log entries were checked for consecutive update indices
and finite loss/gradient norms; saved optimizer moments were checked for shape,
finiteness and step count. Training updates were not replayed. This is a numerical
audit, not independent model replication.

At every checkpoint, resumption from each full current state was audited on 16
calibration examples at steps 0..6; the maximum recorded discrepancy was
**0**. This resumption pays for the remaining Transformer steps and
does not demonstrate that a cheaper predictor can match a history-aware one.

Scientific sources and protocol were locally frozen before natural training and
remained unchanged. The source-hash record is an audit trail, not an independently
timestamped external registration. The verification script is post-run analysis
code and is identified separately from frozen scientific execution sources.

Local panel execution took **7.4 minutes**. Environment:
Python 3.12.14, NumPy 2.3.5.
No GPU was used. Colab runtime can differ. An interactive Colab session was not
executed here; notebook code cells were syntax-checked and embed the same locally
executed source.

## Next boundary

The prospectively frozen selection policy identifies `matrix_decay` as a development candidate. This supports designing the next observer protocol, but not automatically running probes or interpreting present-state sufficiency. Both seeds and every earlier checkpoint remain in the evidence record; a later confirmation study needs untouched evaluation conditions.

Before any observer fit, freeze the predictor families and budgets, strong
current-only baselines, matched/mismatched/permuted/residualized history controls,
train-only preprocessing and eligibility, a useful-prediction criterion and
noninferiority margin, fitted-pipeline sensitivity, family/seed-aware uncertainty,
and untouched confirmation rules. The prior 5% MSE threshold is not automatically
a valid Brier-score margin or deployment threshold. No automatic advancement
was triggered by this panel.

## Files and reproduction

- `P1_A0B_Colab.ipynb`: standalone NumPy notebook; method checks and this recorded
  report by default. Full reproduction requires explicitly enabling its flag.
- `P1_A0B_Protocol.md`: readable prospective specification.
- `P1_A0B_Study.zip`: exact protocol/source, original-code provenance, all model
  checkpoints, optimizer moments, paired batches, predictions, logs and audit.

Reproducing the same four exposed runs is not independent confirmation. Existing
run directories are never overwritten. Every scientific change requires a new
version; do not edit the completed protocol to relabel an observed outcome.
