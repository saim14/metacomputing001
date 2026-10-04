# T1-U5 Residual-Scale Calibration Study

## Outcome

**Status: failed the preregistered development joint gate; the sealed
task-sample holdout remained unopened.**

Increasing the constant residual scale restored an observable multi-step
decision trajectory, but it did not preserve reliable learning across all three
exposed development seeds. No tested scale in {0.6, 0.7, 0.8} simultaneously
met the final-accuracy and trajectory-validity requirements for every seed.

No state-history, motion, or attention-history probe was fit.

## Scientific boundary

Seeds 503, 607, and 709, together with their development trajectory samples,
had already been observed in T1-U4. T1-U5 is therefore a bounded development
calibration rather than a fresh-seed confirmation.

The separately generated task-sample holdout was preregistered with seed
formula `seed * 100 + 83`. It could be opened only for a candidate that passed
the joint development gate. Because no candidate was eligible, those samples
were never generated or evaluated.

## Fixed design

- Task: balanced shortcut-free one-hop retrieval
- Training / validation / development samples: 8,000 / 1,800 / 4,000
- Model: 32-dimensional state, 64-wide feed-forward block, 2 heads
- Computation: one shared Transformer block reused for 6 refinement steps
- Development seeds: 503, 607, and 709
- Total update budget: 4,000 per candidate and seed
- Optimizer: Adam with global gradient-norm clipping at 1.0
- Fixed supervision: the T1-U4 area-matched auxiliary taper
- Calibration factor: constant residual scale 0.6, 0.7, or 0.8
- Closed reference: T1-U4 residual scale 0.5 with the same taper
- Accuracy gate: at least 0.95 final accuracy for every seed
- Trajectory gate: at least two intermediate steps per seed with future-flip
  rate in [0.05, 0.45]
- Advancement rule: both gates must pass for every seed

## Candidate-level results

| Residual scale | Minimum tail validation | Mean tail validation | Minimum development accuracy | Mean development accuracy | Minimum valid steps | Passing seeds | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5 closed reference | 0.94833 | 0.96778 | 0.94850 | 0.97217 | 0 | 1/3 | No |
| 0.6 | 0.86083 | 0.94278 | 0.85300 | 0.94208 | 0 | 1/3 | No |
| 0.7 | 0.87875 | 0.95569 | 0.88025 | 0.95775 | 0 | 1/3 | No |
| 0.8 | 0.91125 | 0.92222 | 0.91350 | 0.92675 | 3 | 1/3 | No |

Scale 0.8 was the only setting with an adequate computation window for every
seed. Its validation minimum exceeded 0.90, but its mean tail validation score
of 0.92222 missed the 0.93 screen. More importantly, two of its development
accuracies were below 0.95.

## Seed-level joint results

| Scale | Seed | Final accuracy | Valid flip steps | Result |
|---:|---:|---:|---|---|
| 0.5 | 503 | 1.00000 | None | Failed trajectory validity |
| 0.5 | 607 | 0.94850 | 2, 3, 4 | Failed accuracy |
| 0.5 | 709 | 0.96800 | 2, 3, 4 | Passed |
| 0.6 | 503 | 0.85300 | 2, 3, 4, 5 | Failed accuracy |
| 0.6 | 607 | 0.97325 | 2, 3 | Passed |
| 0.6 | 709 | 1.00000 | None | Failed trajectory validity |
| 0.7 | 503 | 0.99300 | 2, 3 | Passed |
| 0.7 | 607 | 0.88025 | 2, 3, 4 | Failed accuracy |
| 0.7 | 709 | 1.00000 | None | Failed trajectory validity |
| 0.8 | 503 | 0.95175 | 1, 2, 3, 4 | Passed |
| 0.8 | 607 | 0.91500 | 1, 2, 3, 4, 5 | Failed accuracy |
| 0.8 | 709 | 0.91350 | 2, 3, 4 | Failed accuracy |

Every residual scale passed exactly one of the three seed-level joint gates,
but the passing seed changed with scale. This is evidence of a scale-dependent,
initialization-sensitive optimization surface rather than a single smooth
calibration optimum.

## Main finding

T1-U4 suggested that residual scale 0.5 stabilized learning but compressed one
seed's decision dynamics. T1-U5 shows that simply increasing one global scale
does not repair that tradeoff:

- At 0.5, learning was strongest overall, but seed 503 solved the task between
  steps 1 and 2 and left no valid intermediate step.
- At 0.6 and 0.7, seed 709 still saturated, while a different seed fell into a
  low-accuracy basin.
- At 0.8, all seeds had measurable multi-step dynamics, but seeds 607 and 709
  failed the accuracy gate.

The constant residual scale is serving two roles at once: it shapes both the
training basin and the pace of recurrent computation. The evidence indicates
that those roles cannot be jointly controlled by this one scalar in the tested
range.

## Interpretation and next design

The negative result narrows the design space. A finer constant-scale grid is
not well motivated because the response across seeds is jagged and the limiting
failure changes with scale. The next experiment should decouple optimization
stability from computation pacing.

A principled T1-U6 would compare **area-matched step-wise residual schedules**:
a constant schedule, a small-to-large schedule across refinement steps, and a
large-to-small order control. Holding the total residual-scale area fixed would
test whether allocation across computational depth restores a measurable
trajectory without sacrificing the stable training regime. This directly asks
whether step order matters beyond total update magnitude.

Fresh initialization seeds and trajectory probes remain deferred until one
development configuration passes both gates.

## Execution integrity

An initial three-worker execution completed all scale-0.6 runs but the process
pool terminated as scale 0.7 began, consistent with exceeding the runtime's
stable memory envelope. The runner was made checkpoint-resumable and restarted
with two workers. Completed scale-0.6 checkpoints reproduced the same
development statistics; incomplete scale-0.7 jobs had written no checkpoints
or histories and were rerun deterministically from their original seeds. No
scientific configuration, update count, data stream, threshold, or selection
rule changed.

## Reproducibility

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python run_t1_u5.py \
  --config configs/t1_u5_residual_calibration.yaml \
  --output outputs/t1_u5
```

The portable bundle contains the preregistration, runner, source modules,
tests, nine candidate checkpoints and histories, closed-reference comparisons,
trajectory statistics, gate decisions, figure, environment manifest, report,
and checksums.
