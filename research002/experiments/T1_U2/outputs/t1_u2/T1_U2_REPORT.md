# T1-U2 Bounded Learning Study

## Outcome

**Status: failed the two-seed independent robustness gate.**

T1-U2 identified a compact weight-tied Transformer that solved the balanced,
shortcut-free retrieval task for the selection seed, but the result did not
replicate from a second initialization. The selected configuration is therefore
not ready to be frozen for confirmatory trajectory probing.

No state-history or attention-history probes were run in this study.

## Scientific boundary

T1-U remains a closed failed pilot. T1-U2 is a separately labelled learning
study and does not retroactively tune or reinterpret T1-U. Candidate selection
used validation accuracy only. Independent trajectory data were opened only
after selection and only for the selected candidate.

## Preregistered candidates

Every candidate reused one Transformer block for six refinement steps on the
same balanced one-hop task and used the same low-auxiliary training schedule.

| Candidate | Width / FF | Step embedding | Parameters | Tail validation mean | Terminal validation | Rank |
|---|---:|---|---:|---:|---:|---:|
| U2-A-low-aux | 32 / 64 | No | 9,666 | 0.95278 | 0.97000 | 1 |
| U2-B-wide | 48 / 96 | No | 20,642 | 0.81833 | 0.83833 | 2 |
| U2-C-wide-time | 48 / 96 | Yes | 20,930 | 0.75056 | 0.75667 | 3 |

The advancement threshold was a last-three-checkpoint validation mean of 0.85.
U2-A was selected under the preregistered ranking rule.

## Independent evaluation

The independent set contained 4,000 samples per seed. Passing required final
accuracy of at least 0.95 and future-flip rates between 0.05 and 0.45 at two or
more refinement steps for both the selection and replication seeds.

| Seed | Role | Final accuracy | Valid flip steps | Result |
|---:|---|---:|---|---|
| 311 | Selection | 0.97250 | 2, 3 | Passed |
| 421 | Replication | 0.81675 | 2, 3, 4, 5 | Failed accuracy |

### Refinement trajectory

| Seed | Step 1 acc. | Step 2 acc. | Step 3 acc. | Step 4 acc. | Step 5 acc. | Final acc. |
|---:|---:|---:|---:|---:|---:|---:|
| 311 | 0.49300 | 0.69475 | 0.89150 | 0.95675 | 0.97050 | 0.97250 |
| 421 | 0.49925 | 0.68850 | 0.75850 | 0.79725 | 0.81750 | 0.81675 |

Both seeds produced meaningful multi-step refinement. Seed 311 developed a
near-exact retrieval computation; seed 421 developed a weaker computation and
was still improving late in training.

## Interpretation

The strongest supported conclusion is that the compact tied architecture is
capable of solving the shortcut-free task, but the current optimization recipe
does not do so reliably across initializations.

Within the fixed seed-311 screen, increasing width did not improve learning and
adding learned step embeddings did not rescue the wider model. This does not
establish that width or time embeddings are generally harmful; it only rejects
those specific candidates under the preregistered schedule.

The historical comparison suggests that lowering intermediate supervision may
have removed an important constraint: T1-U reached 0.75525 independent accuracy,
whereas U2-A reached 0.97250 for seed 311 with the same model size. Because the
experiments used different initialization seeds and training lengths, this is
not a clean causal estimate of the auxiliary-loss effect.

## Decision

- Do not freeze U2-A for confirmatory history probes.
- Do not interpret the seed-311 trajectory as evidence for temporal-history
  sufficiency.
- Preserve U2-A as the leading learning configuration.
- Make the next study a preregistered stability experiment across multiple
  development seeds, focused on training horizon and schedule rather than
  additional width.

A defensible T1-U3 would hold the 32/64 architecture fixed, extend final-only
training, and require learning success across multiple seeds before any
trajectory probes are computed.

## Reproducibility

```bash
python run_t1_u2.py \
  --config configs/t1_u2_learning_study.yaml \
  --output outputs/t1_u2
```

The archive includes the preregistration, runner, tests, all candidate training
histories and checkpoints, sealed independent trajectories for the selected and
replication seeds, summary tables, figure, environment manifest, and checksums.

## Record correction

The earlier T1-U report stated 9,634 tied and 52,354 untied parameters. Those
counts used a 10-token vocabulary in a manual check. The executed task uses 11
tokens (`key_count + 3`), so the corrected counts are 9,666 and 52,386. This
arithmetic correction does not change any T1-U result or interpretation.
