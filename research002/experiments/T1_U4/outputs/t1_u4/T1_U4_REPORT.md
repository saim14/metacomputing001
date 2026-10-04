# T1-U4 Optimization-Basin Stabilization Study

## Outcome

**Status: failed the preregistered independent joint gate after passing the
validation screen.**

Residual-update scaling substantially improved learning stability across the
three exposed development seeds. The validation-selected combined intervention
then achieved high independent accuracy, but it failed for two distinct reasons:
seed 607 was 0.00150 below the accuracy threshold, and seed 503 no longer had a
nonsaturated intermediate future-flip target.

No state-history, motion, or attention-history probe was fit.

## Scientific boundary

Seeds 503, 607, and 709 had already been observed in T1-U3. T1-U4 therefore
treats them strictly as development seeds. T1-U4 can identify a promising
optimization intervention but cannot establish fresh-seed generalization.

Candidate selection used validation metrics only. The independent dataset was
opened only for the selected candidate after every trainable candidate and seed
had completed and the fixed screening gate had passed.

## Fixed design

- Task: balanced shortcut-free one-hop retrieval
- Training / validation / independent samples: 8,000 / 1,800 / 4,000
- Model: 32-dimensional state, 64-wide feed-forward block, 2 heads
- Computation: one shared Transformer block reused for 6 refinement steps
- Total update budget: 4,000 for every trainable candidate
- Optimizer: Adam, global gradient-norm clipping at 1.0
- Factors: residual scale 1.0 versus 0.5; abrupt versus tapered auxiliary loss
- Auxiliary-loss area: 40 weight-updates under both schedules
- Primary screen: maximum minimum-seed tail-mean validation accuracy
- Screen requirements: minimum tail mean at least 0.90, overall mean at least
  0.93, and minimum-seed gain over reference at least 0.03
- Independent gate: accuracy at least 0.95 and at least two intermediate steps
  with future-flip rate in [0.05, 0.45] for all seeds

## Validation screening

| Candidate | Residual scale | Transition | Minimum seed tail mean | Mean tail accuracy | Minimum terminal accuracy |
|---|---:|---|---:|---:|---:|
| U4-A reference | 1.0 | Abrupt | 0.77417 | 0.86625 | 0.79667 |
| U4-B residual half | 0.5 | Abrupt | 0.94167 | 0.96528 | 0.95500 |
| U4-C auxiliary taper | 1.0 | Tapered | 0.76042 | 0.85194 | 0.79167 |
| **U4-D combined** | **0.5** | **Tapered** | **0.94833** | **0.96778** | **0.95667** |

The selected candidate improved the primary maximin statistic by 0.17417 over
the closed T1-U3 reference. It passed all three validation-screening conditions.

## Factor estimates

Effects below are differences in mean seed-level tail validation accuracy over
the complete 2-by-2 development matrix.

| Effect | Estimate |
|---|---:|
| Residual scale 0.5 main effect | +0.10743 |
| Auxiliary taper main effect | -0.00590 |
| Residual-by-taper interaction | +0.01681 |

Residual scaling is the dominant intervention. Tapering alone does not improve
stability and approximately reproduces the failed T1-U3 regimes. Its small
positive interaction with residual scaling was sufficient for the combined
candidate to win the fixed maximin ranking.

## Independent evaluation of U4-D

| Seed | T1-U3 independent | T1-U4 independent | Accuracy gain | Valid flip steps | Result |
|---:|---:|---:|---:|---|---|
| 503 | 0.96100 | 1.00000 | +0.03900 | None | Failed target validity |
| 607 | 0.80200 | 0.94850 | +0.14650 | 2, 3, 4 | Failed accuracy by 0.00150 |
| 709 | 0.86375 | 0.96800 | +0.10425 | 2, 3, 4 | Passed |

Mean independent accuracy increased from 0.87558 in T1-U3 to 0.97217 in the
selected T1-U4 candidate, a gain of 0.09658. The minimum independent accuracy
increased from 0.80200 to 0.94850.

These gains are development evidence rather than an unbiased estimate of
fresh-seed improvement because the seeds were already exposed.

## Target-validity tradeoff

Seed 503 reached perfect independent accuracy, but its future-flip rates were:

| Refinement step | Future-flip rate |
|---:|---:|
| 1 | 0.48400 |
| 2 | 0.00550 |
| 3 | 0.00100 |
| 4 | 0.00000 |
| 5 | 0.00000 |

The model changed from near-random prediction at step 1 to nearly complete
solution at step 2. No intermediate step fell inside the preregistered target
window. The intervention therefore made the base task more reliable while
compressing the observable decision dynamics for this seed.

Seeds 607 and 709 retained valid future-flip targets at steps 2--4, showing that
target saturation is not universal. Nevertheless, the all-seed target gate was
specified in advance and must be respected.

## Interpretation

T1-U4 provides strong development evidence that unscaled recurrent residual
updates were a major cause of optimization-basin sensitivity. Halving each
attention and feed-forward residual update moved all three seeds into much
stronger learning regimes. Auxiliary tapering was not independently helpful.

The result does not yet authorize trajectory probing. Reliable final learning
and informative intermediate variation are separate design requirements. T1-U4
improved the first but caused the second to fail for one seed.

The next design should preserve residual scaling while explicitly restoring a
measurable computation window. A bounded residual-scale calibration (for
example, values between 0.5 and 1.0) or a modest task-difficulty increase can be
tested, but either choice must be separately preregistered. Fresh-seed
confirmation should occur only after both the learning and target-validity gates
are satisfied in development.

## Reproducibility

```bash
python run_t1_u4.py \
  --config configs/t1_u4_optimization_stability.yaml \
  --output outputs/t1_u4
```

The portable bundle contains the preregistration, runner, source changes,
tests, nine new checkpoints, validation histories, the selected candidate's
three independent trajectories, screening tables, selection decision, figure,
environment manifest, and checksums.
