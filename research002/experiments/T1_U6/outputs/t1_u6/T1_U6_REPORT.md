# T1-U6 Frozen-Weight Residual-Pacing Intervention

## Direction audit and reformulation

The initially proposed T1-U6 would have trained separate models with constant,
increasing, and decreasing residual schedules. That design was not sufficiently
causal: changing the training schedule would also change the learned weights
and optimization basin, so a later trajectory difference could not be assigned
specifically to computation pacing.

T1-U6 was therefore reformulated before execution. It freezes the three
T1-U4 residual-0.5 checkpoints and changes only the residual scale applied at
each inference-time refinement step. This creates a paired intervention in
which learned weights, examples, total residual-scale area, architecture, and
readout are held fixed.

**Status: neither ordered schedule passed the preregistered development joint
gate; the sealed task-sample holdout remained unopened.**

No state-history, motion, or attention-history probe was fit.

## Scientific boundary

Seeds 503, 607, and 709 and their development samples had already been exposed
in T1-U4 and T1-U5. T1-U6 is a mechanistic development intervention, not a
fresh-initialization confirmation.

The constant schedule reproduced all 15 saved T1-U4 seed-by-layer trajectory
records with maximum absolute error `2.22e-16`. This verifies checkpoint
loading, sample reconstruction, and backward compatibility before interpreting
the interventions.

## Fixed weights and interventions

All schedules have six steps and residual-scale sum 3.0:

| Schedule | Step-wise residual scales | Selectable |
|---|---|---|
| Constant reference | 0.50, 0.50, 0.50, 0.50, 0.50, 0.50 | No |
| Back-loaded | 0.25, 0.35, 0.45, 0.55, 0.65, 0.75 | Yes |
| Front-loaded | 0.75, 0.65, 0.55, 0.45, 0.35, 0.25 | Yes |

The ordered interventions are exact reversals of the same scale multiset.
Consequently, their difference isolates order under the stated intervention;
it is not caused by a different average or collection of scale values.

The joint gate remained unchanged: final accuracy at least 0.95 and at least
two intermediate steps with future-flip rate in [0.05, 0.45] for every seed.

## Results

| Schedule | Seed 503 accuracy / valid steps | Seed 607 accuracy / valid steps | Seed 709 accuracy / valid steps | Mean accuracy | Passing seeds |
|---|---:|---:|---:|---:|---:|
| Constant | 1.00000 / 0 | 0.94850 / 3 | 0.96800 / 3 | 0.97217 | 1/3 |
| Back-loaded | 0.97300 / 1 | 0.81425 / 3 | 0.78800 / 4 | 0.85842 | 0/3 |
| Front-loaded | 0.99925 / 0 | 0.91150 / 4 | 0.87275 / 4 | 0.92783 | 0/3 |

Back-loading extended the observable computation window for seeds 607 and 709,
but it strongly damaged their final decisions. It also produced only one valid
intermediate step for seed 503. Front-loading preserved more accuracy than
back-loading but left seed 503 saturated and failed the 0.95 accuracy gate for
the other seeds.

Neither selectable schedule was eligible. The sealed sample remained unopened.

## Causal order comparison

Because the two ordered interventions share weights, examples, total scale, and
the same scale multiset, their paired difference provides a direct order test.

| Seed | Back-loaded minus front-loaded accuracy | Difference in valid-step count |
|---:|---:|---:|
| 503 | -0.02625 | +1 |
| 607 | -0.09725 | -1 |
| 709 | -0.08475 | 0 |
| **Mean** | **-0.06942** | **0** |

Front-loading achieved higher final accuracy for all three seeds. The mean
valid-step count was identical, although its seed-level distribution differed.
This establishes that residual allocation order causally changes the frozen
recurrent computation. It does **not** show that either schedule is useful.

## Realized state movement

Area matching controls the sum of nominal multipliers, not the nonlinear state
trajectory. Mean CLS-state update norms, averaged across seeds, followed the
intended intervention:

| Step | Constant | Back-loaded | Front-loaded |
|---:|---:|---:|---:|
| 1 | 4.599 | 2.306 | 6.908 |
| 2 | 4.027 | 2.714 | 5.201 |
| 3 | 4.294 | 3.814 | 4.573 |
| 4 | 4.492 | 4.857 | 3.844 |
| 5 | 4.552 | 5.782 | 3.029 |
| 6 | 4.505 | 6.621 | 2.167 |

The schedules successfully changed computation pacing. Their failure is
therefore substantive, rather than a failure to produce the intervention.

## General interpretation

T1-U6 answers the double-check constructively:

1. The original retraining proposal was confounded and required reformulation.
2. The frozen-weight reformulation was an appropriate causal diagnostic.
3. It falsified the tested inference-only pacing rescue.
4. It also showed that equal nominal residual-scale area does not preserve final
   computation in a nonlinear recurrent Transformer.

Most importantly, the one-hop task does not intrinsically require six
refinement steps. An early stable answer can be valid efficient computation,
not a trajectory defect. Continuing to engineer label flips risks creating the
phenomenon that the project intends to measure.

## Recommended next direction

Do not run a finer pacing grid. The next design should provide genuine
multi-step computational demand—for example, a carefully staged two-hop
shortcut-free retrieval task—while retaining the stable weight-tied architecture
and residual-0.5 training lessons. The primary target should return to the
original quantity, marginal future loss reduction or value of additional
computation. Prediction flips can remain a secondary diagnostic rather than a
required phenomenon at every useful layer.

This differs from the earlier failed three-hop curriculum: the next study
should begin with a bounded two-hop learning/validity stage and must not fit
trajectory probes until learning is established. Its exact curriculum, target
variance gate, and fresh-seed boundary require separate preregistration.

## Reproducibility

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python run_t1_u6.py \
  --config configs/t1_u6_frozen_pacing.yaml \
  --output outputs/t1_u6
```

The portable bundle includes the preregistration, runner, schedule-aware model
source, tests, frozen-checkpoint hashes, trajectory and state-movement tables,
gate decisions, reference verification, figure, environment manifest, report,
checksums, and copies of the three frozen T1-U4 checkpoints required to rerun
the intervention independently.
