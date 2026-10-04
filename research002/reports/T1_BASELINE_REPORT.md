# T1 — Transformer Trajectory Sufficiency

**Run:** baseline 0.1.0  
**Date:** 2026-09-01  
**Research lead:** Saim 13.02

## Question

Does ordered layer-wise computational history predict a Transformer's future
decision change or future loss reduction beyond its current representation and
prediction?

## System and task

- Encoder-only Transformer: 4 layers, width 32, 2 attention heads, FFN width 64.
- Controlled two-hop key/value retrieval with 8 possible keys and 4 pairs.
- Three model seeds: 101, 202, 303.
- 6,000 training, 1,200 validation, and 3,600 independent probe samples per seed.
- Shared readout at each layer; auxiliary loss weight 0.25 on intermediate layers.
- CPU-only NumPy implementation with local reverse-mode autodiff.

## Controls

The probe comparison included current state, ordered history, per-sample shuffled
past, a norm-preserving dimension-matched current-state control, sample-mismatched
past, and history residualized from the current state.

Fixed reversal was not used as an order control. For a concatenated linear probe,
fixed reversal is only a global permutation of feature columns; it preserves all
information and is therefore unable to test whether order matters.

## Training outcome

Final validation accuracies were 1.0000, 1.0000, and 0.9947 for seeds 101, 202,
and 303. The Transformer therefore learned the retrieval task reliably.

The future-flip target, however, saturated rapidly:

| Seed | Layer 1 | Layer 2 | Layer 3 |
|---:|---:|---:|---:|
| 101 | 0.3094 | 0.0000 | 0.0000 |
| 202 | 0.3139 | 0.0000 | 0.0000 |
| 303 | 0.3633 | 0.0169 | 0.0094 |

Only five seed-layer cells had enough positive and negative cases for a binary
future-flip probe. This is a central limitation, not a missing-data detail.

## Primary result: future decision flip

Ordered history did not improve over the current-state probe:

- Mean paired ΔROC AUC favoring ordered history: **−0.00079**
  (95% CI **[−0.00540, 0.00381]**, n = 5).
- Mean paired log-loss effect favoring ordered history: **−0.00017**
  (95% CI **[−0.00337, 0.00303]**, n = 5).

Ordered history also failed to show a reliable advantage over the
dimension-matched, sample-mismatched, or shuffled controls. The apparent
ordered-versus-shuffled ΔROC AUC was +0.00867, but its 95% CI
[−0.01268, 0.03002] included zero and was driven by very few valid cells.

## Secondary result: future NLL reduction

Ordered history produced only a small and uncertain improvement over current
state for regression of final negative-log-likelihood gain:

- Mean paired ΔR²: **+0.00295**
  (95% CI **[−0.00119, 0.00709]**, n = 9).
- Mean paired MAE effect favoring ordered history: **−0.00006**
  (95% CI **[−0.00024, 0.00013]**, n = 9).

The R² direction is weakly positive, but the confidence interval crosses zero
and MAE does not improve. This does not support a robust history advantage.

## Conclusion

T1 baseline 0.1 does **not support** the hypothesis that ordered Transformer
layer history contains stable predictive information beyond the current state
under the tested task, architecture, readout, and controls.

This is not yet a decisive falsification of the broader Transformer hypothesis.
The main reason is target degeneration: two models had already reached complete
decision stability after layer 1. The experiment verified the implementation and
controls, while showing that the next task must preserve meaningful computation
across more layers.

## Next experimental decision

T1.1 should modify the computational regime before adding more seeds:

1. Increase retrieval depth or introduce distractor chains so information is
   integrated over multiple layers.
2. Reduce or remove intermediate auxiliary supervision, which may encourage
   premature stability.
3. Use 6 layers and predefine a minimum valid future-flip rate per evaluated
   layer (for example 0.05–0.40) as a run-validity criterion.
4. Add attention-history probes using the already recorded per-head entropy,
   maximum attention, and query-token mass.
5. Preserve the current target, control, and paired-effect definitions so T1.1
   changes the task regime rather than the evidential standard.

## Interpretation boundary

Probe performance measures incremental decodability. Even a future positive
result would not by itself establish causal use, self-monitoring, or
metacognition. Those claims require interventions on the identified signal.

