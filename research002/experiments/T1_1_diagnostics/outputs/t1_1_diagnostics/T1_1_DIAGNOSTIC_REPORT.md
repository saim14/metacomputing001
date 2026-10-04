# T1.1 — Multi-hop Transformer Pilot Diagnostics

**Version:** 0.3.0  
**Date:** 2026-09-01  
**Research lead:** Saim 13.02

## Purpose

T1 showed rapid target saturation: two seeds had no future decision flips after
layer 1. T1.1 attempted to create a learned computation that remained active
across more Transformer layers, while adding direct attention-history controls.

The preregistered pilot required:

- final probe accuracy at least 0.85;
- future-flip rate between 0.05 and 0.45;
- at least three valid intermediate layers.

Failure of the learning criterion invalidated trajectory and attention probes as
evidence. No failed pilot was scaled to multiple seeds.

## Implemented extensions

- Balanced pointer-chasing data with one-, two-, and three-hop variants.
- Equal value-token counts in every sample to prevent count-based label leakage.
- Six-layer instrumented Transformer.
- Optional source/destination role embeddings and shared mapping-pair embeddings.
- Ordered attention-history probes using per-head entropy, maximum weight, and
  query-token mass.
- Current-attention, dimension-matched, shuffled-history, and
  sample-mismatched attention controls.
- Phase-logged curriculum training from one-hop to three-hop retrieval.
- Explicit run-validity evaluation in every summary.

## Pilot registry

| Run | Main change | Final probe accuracy | Maximum logged validation accuracy | Validity |
|---|---|---:|---:|---|
| T1.1 | Three-hop, six layers, weak auxiliary loss | 0.5253 | 0.5313 | Failed |
| T1.1a | Two-hop task | 0.5100 | 0.5344 | Failed |
| T1.1b | Three-hop plus role/pair embeddings | 0.5000 | 0.5375 | Failed |
| T1.1c | Balanced one-hop in six-layer regime | 0.5263 | 0.5656 | Failed |
| T1.1d | Balanced one-hop, T1 architecture, 900 steps | 0.6669 | 0.6567 | Failed |
| T1.1e | Same as T1.1d, 1,800 steps | 0.8188 | 0.8533 | Failed |
| T1.1f | One-hop → two-hop → three-hop curriculum | 0.5100 | 0.6938 | Failed |
| T1-R | Balanced retrieval plus late low-auxiliary fine-tuning | 0.7831 | 0.7850 | Failed |

For T1.1f, phase-ending validation accuracies were 0.6719 for one hop, 0.5031
for two hops, and 0.5281 for three hops. Curriculum sequencing did not produce
compositional transfer.

## Newly identified shortcut in T1

The original direct-lookup generator forced the queried value while sampling
the other three values independently. Consequently, the relative counts of
value-zero and value-one tokens carried label information.

A logistic probe using only the two value-token counts achieved:

| Generator | Count-only accuracy | Count-only ROC AUC |
|---|---:|---:|
| Original T1 direct lookup | 0.6885 | 0.7714 |
| Balanced one-hop generator | 0.4995 | 0.5000 |

The analytic accuracy of a majority-count rule with random tie-breaking in the
original four-pair task is 0.6875. This closely matches T1's layer-1 accuracies
of approximately 0.69.

The most plausible computational account of T1 is therefore:

1. Layer 1 exploits a global value-count heuristic.
2. Layer 2 performs sufficiently accurate query-conditioned retrieval.
3. Later layers preserve an already stable decision.

This explains both the layer-1 performance and the rapid disappearance of the
future-flip target. It also makes T1's result more specific: the baseline tested
whether trajectory history predicted correction from a count heuristic to exact
retrieval, not a generic form of prolonged Transformer reasoning.

## Scientific conclusion

T1.1 produced **no valid test of the state-history or attention-history
hypotheses**, because the balanced multi-hop target was not learned. The
nontrivial future-flip rates in failed runs reflect unresolved predictions and
cannot be interpreted as useful computational refinement.

The negative optimization results are nevertheless informative:

- reducing auxiliary supervision and adding layers prevented even balanced
  one-hop learning under the tested budget;
- structural role/pair embeddings did not solve equality-and-composition
  learning;
- a simple difficulty curriculum did not transfer from partial one-hop learning
  to two- or three-hop reasoning;
- the balanced one-hop primitive became substantially learnable only with the
  proven four-layer T1 architecture and longer optimization.

## T1-R outcome

T1-R used the proven four-layer architecture, 2,400 high-auxiliary training
steps, and 600 reduced-auxiliary fine-tuning steps. Its independent probe
accuracy was 0.7831, below the preregistered 0.95 threshold. Future-flip rates
were 0.4878, 0.1519, and 0.0233 at layers 1–3; only layer 2 met the declared
0.05–0.45 range. T1-R therefore failed both its learning and target-availability
requirements, and its state/attention probes are diagnostic only.

## Next valid experiment: T1-U

The appropriate next experiment is a weight-tied recurrent Transformer
(Universal-Transformer-style) on balanced one-hop retrieval:

1. One Transformer block reused for six refinement steps.
2. Two attention heads, width 32, and FFN width 64.
3. Four source/value pairs with equal value-token counts per sample.
4. Role and pair embeddings retained.
5. A one-seed pilot must pass learning and target-prevalence gates before scale.
6. State- and attention-history controls remain unchanged.
7. Step embeddings or halting mechanisms are excluded from the first pilot.

This separates two questions that T1.1 initially combined:

- Can the Transformer reliably learn retrieval without a statistical shortcut?
- Once it does, does computational history add information beyond its current
  representation and attention state?

Weight tying reduces parameter-learning burden and makes depth a repeated
computation operator. This gives the trajectory a stronger temporal meaning and
aligns more directly with the original metacognitive controller proposal. Only
after T1-U learns shortcut-free retrieval should multi-hop depth be reintroduced.
