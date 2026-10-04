# T1-U3 Training-Stability Study

## Outcome

**Status: failed the preregistered multi-seed stability gate.**

The extended final-only schedule improved validation performance for all three
fresh seeds, but only one seed reached the independent accuracy requirement.
The compact recurrent Transformer remains capable of exact-like retrieval yet
is not reliably trainable across initializations.

No state-history or attention-history probes were run.

## Scientific boundary

T1-U3 is a separately preregistered stability study. It holds the T1-U2
U2-A architecture fixed and does not retroactively alter T1-U2. The three seeds,
4,000-update schedule, independent learning gate, and target-validity gate were
specified before training.

## Fixed protocol

- Fresh seeds: 503, 607, 709
- Task: balanced, shortcut-free one-hop retrieval
- Training / validation / independent samples: 8,000 / 1,800 / 4,000
- Model: 32-dimensional state, 64-wide feed-forward block, 2 heads
- Computation: one Transformer block reused for 6 refinement steps
- Step embeddings: disabled
- Phase 1: 2,000 updates, auxiliary weight 0.02, learning rate 0.0012
- Phase 2: 2,000 final-only updates, learning rate 0.0005
- Historical comparison horizon: update 2,400
- Required passing fraction: 3/3 seeds
- Per-seed gate: independent accuracy at least 0.95 and at least two steps
  with future-flip rate in [0.05, 0.45]

## Learning results

| Seed | Validation at 2,400 | Final validation | Gain | Independent accuracy | Result |
|---:|---:|---:|---:|---:|---|
| 503 | 0.93333 | 0.95833 | +0.02500 | 0.96100 | Passed |
| 607 | 0.71833 | 0.79667 | +0.07833 | 0.80200 | Failed accuracy |
| 709 | 0.82667 | 0.88333 | +0.05667 | 0.86375 | Failed accuracy |

Aggregate independent accuracy was 0.87558 with sample standard deviation
0.08016 and range [0.80200, 0.96100]. The passing fraction was 1/3 rather than
the required 3/3.

The mean validation gain after the previous 2,400-update horizon was +0.05333.
Thus the longer schedule helped, but additional time alone did not stabilize
which computational solution the recurrent model learned.

## Target validity

| Seed | Step 2 flip rate | Step 3 flip rate | Step 4 flip rate | Valid steps |
|---:|---:|---:|---:|---|
| 503 | 0.27375 | 0.11250 | 0.05350 | 2, 3, 4 |
| 607 | 0.19950 | 0.12300 | 0.06575 | 2, 3, 4 |
| 709 | 0.17450 | 0.11500 | 0.07425 | 2, 3, 4 |

The future-flip target window was consistent across every initialization.
Target saturation is therefore not the reason T1-U3 failed.

## Interpretation

T1-U3 rejects the hypothesis that a longer final-only phase is sufficient to
make this architecture reliably learn the task. It does not show that the
architecture lacks capacity: seed 503 reached 0.961 independent accuracy, and
T1-U2 seed 311 previously reached 0.9725.

The evidence instead points to optimization-basin sensitivity. Seeds enter
qualitatively different regimes under identical data-generation rules,
architecture, loss schedule, and training horizon. More updates produced gains
inside each regime but did not force the weaker seeds into the high-accuracy
solution.

## Decision

- Do not freeze the configuration for confirmatory trajectory probing.
- Do not fit probes to the successful seed alone.
- Preserve refinement steps 2--4 as the empirically stable target window.
- Stop treating training horizon as the sole stabilization intervention.

The next study should target optimization-basin selection directly while
holding the task and model size fixed. Appropriate bounded factors include
residual-update scaling and a less abrupt auxiliary-to-final-only transition,
tested across shared seeds before any trajectory analysis.

## Reproducibility

```bash
python run_t1_u3.py \
  --config configs/t1_u3_stability.yaml \
  --output outputs/t1_u3
```

The portable package includes the preregistration, runner, tests, three model
checkpoints, three independent trajectories, per-seed and combined training
histories, target statistics, summary, figure, environment manifest, and
checksums.

