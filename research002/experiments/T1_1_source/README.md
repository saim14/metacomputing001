# T1 — Transformer Trajectory Sufficiency

T1 asks whether earlier layer states predict the value of future Transformer
computation beyond the current layer representation and current prediction.

## Scientific question

For layer `l`, does the trajectory `0..l-1` add information about a future
prediction change or final loss reduction once `h_l` and `p_l` are known?

The baseline uses an encoder-only Transformer on a synthetic two-hop key/value
retrieval task. A shared readout exposes a probability distribution at the
embedding state and after every Transformer layer. Earlier layers receive a
small auxiliary training loss so their predictions are explicitly interpretable.

## Probe targets

- `future_flip`: current predicted class differs from the final-layer class.
- `future_correction`: current prediction is wrong and the final prediction is correct.
- `delta_nll_to_final`: current negative log-likelihood minus final negative log-likelihood.

## Controls

- `current`: current hidden state, probabilities, entropy, and margin.
- `ordered_history`: all state features from the embedding state through layer `l`.
- `shuffled_history`: per-sample random order of past layer blocks; current stays fixed.
- `dimension_matched`: norm-preserving repeats of current features to the same width.
- `sample_mismatched_history`: past trajectory from another sample; current stays fixed.
- `residual_history`: past features after a train-fit linear prediction from current features;
  near-zero numerical residuals are suppressed before probe standardization.

Fixed reversal is intentionally excluded as an order test. With a concatenated
linear probe, a fixed reversal is only a global feature-column permutation and
does not remove information.

## Interpretation boundary

A reliable ordered-history advantage would establish incremental decodability
under these controls. It would not yet prove that the Transformer causally uses
the historical signal, or that the system is metacognitive. Causal interventions
belong in a later experiment.

## Run

```bash
python run_t1.py --config configs/t1_smoke.yaml --output outputs/t1_smoke
python run_t1.py --config configs/t1_baseline.yaml --output outputs/t1_baseline
```

The code uses NumPy plus a small local reverse-mode autodiff engine, so it does
not require PyTorch or a GPU. Outputs include checkpoints, compressed
trajectories, raw probe metrics, paired effects, layer statistics, a figure,
and an environment manifest.

## T1.1 extension

T1.1 replaces direct lookup with a balanced three-hop pointer chain, expands the
Transformer to six layers, reduces auxiliary supervision, and preregisters a
run-validity requirement before scaling beyond one seed. It also adds an
attention-history family that holds the current state fixed while comparing
ordered per-head attention summaries against current-attention,
dimension-matched, shuffled, and sample-mismatched controls.

The T1.1b diagnostic adds label-independent structural embeddings: one role
embedding distinguishes query/source/destination positions, and a shared pair
embedding identifies which adjacent source and destination form a mapping. This
tests whether the raw pointer task failed because its relation was poorly
represented rather than because multi-hop computation was intrinsically too hard.

T1.1f introduces a three-phase curriculum in the same model: balanced one-hop
retrieval, two-hop composition, and three-hop composition. Probe data are drawn
only from the final three-hop task; curriculum validation is retained phase by
phase in the training history.

T1-R returns to the learnable four-layer architecture and tests shortcut-free
one-hop retrieval first. It requires at least 0.95 final probe accuracy and two
intermediate layers with nondegenerate future flips before trajectory or
attention-history effects are treated as evidence.

```bash
python run_t1.py --config configs/t1_1_pilot.yaml --output outputs/t1_1_pilot
python run_t1.py --config configs/t1_1_scaled.yaml --output outputs/t1_1_scaled
```
