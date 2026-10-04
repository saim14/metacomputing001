# T1-U Pilot Report

## Outcome

**Status: failed the preregistered learning gate.**

T1-U reused one Transformer block across six refinement steps on the balanced,
shortcut-free one-hop retrieval task. The independent probe-set final accuracy
was **0.75525**, below the preregistered minimum of **0.95**. The planned
multi-seed confirmatory run was therefore not started.

This is a learning-sufficiency failure, not a target-saturation failure. Future
decision flips were within the preregistered 0.05--0.45 interval at refinement
steps 2, 3, and 4.

## Preregistered design

- Seed: 101
- Data: 8 keys, 4 key-value pairs, balanced one-hop lookup
- Training / validation / independent probe samples: 8,000 / 1,800 / 4,000
- Transformer state size: 32
- Attention heads: 2
- Feed-forward size: 64
- Refinement steps: 6
- Weight tying: one block reused at every refinement step
- Structural role and pair embeddings: enabled
- Step embeddings: disabled
- Halting/controller mechanism: disabled
- Training: 3,200 updates at auxiliary weight 0.15 and learning rate 0.0012,
  then 800 updates at auxiliary weight 0.05 and learning rate 0.0006
- Learning gate: final independent accuracy at least 0.95
- Target-validity gate: future flip rate 0.05--0.45 at two or more steps

The tied model contained 9,666 unique scalar parameters, compared with 52,386
for the otherwise matched six-block untied model. These corrected counts use
the actual 11-token vocabulary (`key_count + 3`).

## Learning result

| Quantity | Result |
|---|---:|
| Best logged validation accuracy | 0.77250 |
| Best validation step | 3,000 |
| Final logged validation accuracy | 0.76750 |
| Independent probe-set accuracy | 0.75525 |
| Required independent accuracy | 0.95000 |
| Learning gate | Failed |

Independent accuracy and future target behavior by current refinement step:

| Step | Current accuracy | Final accuracy | Future flip rate | Future correction rate | Target valid? |
|---:|---:|---:|---:|---:|---|
| 1 | 0.50975 | 0.75525 | 0.50050 | 0.37300 | No: above maximum |
| 2 | 0.68375 | 0.75525 | 0.23950 | 0.15550 | Yes |
| 3 | 0.73650 | 0.75525 | 0.10325 | 0.06100 | Yes |
| 4 | 0.74300 | 0.75525 | 0.06925 | 0.04075 | Yes |
| 5 | 0.75275 | 0.75525 | 0.02900 | 0.01575 | No: below minimum |

## Diagnostic probes

These results are retained for diagnosis only. They must not be interpreted as
confirmatory evidence because the model failed the learning prerequisite and
only one seed was run.

For future-flip prediction at the three target-valid steps, the ordered state
history produced the following ROC-AUC differences:

| Step | Ordered AUC | vs current | vs shuffled | vs dimension-matched |
|---:|---:|---:|---:|---:|
| 2 | 0.88501 | +0.01142 | +0.00095 | +0.01001 |
| 3 | 0.93518 | +0.02129 | +0.00957 | +0.01382 |
| 4 | 0.97683 | +0.01225 | +0.01888 | +0.01307 |

The state-history signal is small and not uniformly order-specific: at step 2,
ordered and shuffled histories were nearly identical, and residual-history
controls remained competitive.

Ordered attention history did not show a credible temporal-order advantage.
Its ROC-AUC differences versus shuffled attention history at steps 2--4 were
0.00000, -0.00026, and -0.00135 respectively.

## Interpretation

T1-U establishes that the tied block produced a genuine multi-step refinement
trajectory: accuracy rose substantially across recurrent applications and
decision changes remained measurable at multiple steps. It did not establish
a sufficiently learned retrieval computation for a valid test of trajectory
sufficiency.

The result therefore does **not** support or refute the primary temporal-history
hypothesis. It narrows the immediate problem to optimization/capacity of the
weight-tied recurrent model under shortcut-free retrieval.

## Decision

The preregistered multi-seed stage is stopped. The next experiment should be a
separately labelled learning study, not an unregistered extension of T1-U. A
useful next design would preserve weight tying and the balanced task while
testing a small, explicit optimization/capacity grid before locking a new
confirmatory trajectory experiment.

## Reproducibility

Run from the project root:

```bash
python run_t1.py --config configs/t1_u_pilot.yaml --output outputs/t1_u_pilot
```

The raw trajectory, checkpoint, training history, probe metrics, paired effects,
figures, environment manifest, configuration snapshot, and source code are
included with the experiment archive.

