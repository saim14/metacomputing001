# P1-A0B — Arithmetic competence revision

Research 002 · 15 September 2026 · v1.0.0  
Prospective, bounded development screen; no history probes or confirmation.

## Why this stage

P1-A0 finished 1,200 updates with 68.11% training accuracy and 3.16% accuracy on
unseen canonical operand combinations. That gap motivates a competence revision,
not a present-state-sufficiency conclusion. We now test a longer training budget
and one fixed regularization contrast. Earlier T1/T2 and P1-A0 findings remain
unchanged. P1-A1 remains the future observer stage; this is still readiness work.

Small algorithmic tasks can show late generalization, and weight decay can help
in some settings. The original Grokking paper used much larger models and budgets
than ours: its appendix describes AdamW with lr=0.001, decay=1 and up to 100,000
updates for most experiments. This is motivation, not a claim that its results
transfer here or that 6,000 updates are sufficient. We preserve our beta2=0.999
and use matrix-only decay, so this is not a reproduction of that recipe.

## Frozen contrast

Keep `(a+b+c) mod 17`, three operand tokens plus CLS, width24, two attention heads,
feed-forward width48, and one pre-norm Transformer block reused for six steps.
There are 5,873 trainable parameters. Keep final-only cross entropy, batch64,
lr0.001, Adam beta1=0.9/beta2=0.999, epsilon1e-8 and global gradient clip1.0.

| Setting | No-decay arm | Matrix-decay arm |
|---|---|---|
| Training updates | 6,000 | 6,000 |
| Model seeds | 2017 and 2027 | Same paired seeds |
| Batch stream | Fixed per seed | Identical paired stream |
| Decoupled weight decay | 0 | 1, on matrices only |
| Checkpoints | 0, 1,200, 3,000, 6,000 | Same |

All four runs start from initialization. No prior trained weights are loaded.
Within each seed, initial weights and all 6,000 batches must match exactly across
arms. This isolates the specified weight-decay change at the matched budget;
scheduled checkpoints describe learning with additional updates. Do not interpret
comparisons with P1-A0's different seed as a controlled optimizer contrast.

For decayed matrices, the optimizer uses
`theta_next = (1 - lr*decay)*theta - lr*m_hat/(sqrt(v_hat)+epsilon)`,
with moments from the globally clipped gradients at the pre-decay weights.
The multiplicative shrink is 0.1% per update, not 100%. Token/position embeddings
and linear weight matrices are decayed; biases and LayerNorm vectors are not.
Both arms include the same loop; setting decay to zero must be numerically
identical to the original Adam implementation. Unit fixtures check this, the
closed-form first and multiple updates, and the matrix/vector mask.

## Development data, not a new test set

Reuse exactly P1-A0's 2,992 training examples/581 families and 950 calibration
examples/193 families. All permutations of a sorted triple remain in one split.
These calibration outcomes already influenced the research direction; they are
**exposed development data**, although the models never train on their labels.
Two new model seeds are not untouched study confirmation on fresh data.

The 195 reserved families remain unexpanded and unevaluated. Do not reshuffle
them into training. Old reserved model seeds 1103/1201/1301 remain unused.
The model-seed-specific stream seed is `500000 + seed`; save the entire int64
batch-index matrix and its hash. Input labels are exact arithmetic targets.

## Outcomes fixed before training

The primary descriptive comparison is the 6,000-update calibration accuracy
difference, decay minus no-decay, for each seed and their mean. Report all four
runs, all scheduled checkpoints, and training/calibration accuracy. No best
checkpoint, best seed, p-value, or two-seed model-population interval is used.

A regimen is ready for future observer design only if BOTH model seeds reach
90% final calibration accuracy and at least two **common** steps among 2..5 have:

- At least 50 examples in each decision-change class.
- At least 25 distinct families contributing to each class.
- At least 10% of examples in each class.

The target is disagreement of the current shared-head argmax with the final
argmax, not mathematical correctness. A family can contribute to both classes
through different permutations. These gates avoid a near-constant target and
spurious support from permutations; they are not a power analysis.

Use only the checkpoint at update6,000 for readiness. If both regimens pass,
prefer the no-decay regimen as the simpler unchanged optimizer. If neither
passes, close this bounded screen without adding runs or updates. A single
successful seed cannot compensate for a failed seed. A numeric/method failure
invalidates completion and stops the panel; do not replace it with a new seed.

## Scope and resources

The development budget is exactly four natural training runs, 24,000 updates,
and no observer fits. Run order: seed2017 no-decay, seed2017 decay, seed2027
no-decay, seed2027 decay. Log progress at 1,000-update intervals. No schedule or
hyperparameter change is allowed based on emerging outcomes. New paths prevent
overwriting prior outputs; a local freeze binds scientific inputs before fitting.
This local audit trail is not an independently timestamped registration service.

Recheck original method fixtures, verify the new optimizer with synthetic tests,
and audit continuation from all-token states at every checkpoint. Resuming the
actual network from its current state remains an architectural property, not an
awareness finding or evidence for cheap prediction. Save logits/checkpoints;
regenerate full states when needed to keep the archive small.

Completion permits a decision about the **next design**, never automatic history
tests. Future observer fitting still needs explicit predictor budgets, leakage
controls, strong current-only references, a useful-prediction criterion and
margin, sensitivity tests through the actual fitted pipeline, family-aware
uncertainty and a genuinely untouched confirmation plan. No claim about awareness,
metacognition, present-state sufficiency or the benefit of history follows here.

## References

- Prior sources: bundled unchanged `prior/P1_A0_Report.md` and
  `prior/P1_A0_protocol.json`; original NumPy Transformer/autodiff is unchanged.
- [Power et al., Grokking](https://arxiv.org/abs/2201.02177), especially sections3.1,
  3.3 and appendixA.1.2: motivates longer optimization/regularization but uses a
  different regime and does not establish our hypothesis.
- [Loshchilov and Hutter, Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101):
  supports separating weight decay from the adaptive gradient update.
