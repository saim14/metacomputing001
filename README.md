# Metacomputation research

Author: MD Saim Islam · Research reports credited as Saim 13.02.

This repository preserves the original TC-CVE iterative-network notebook and extends it with **Research 002 — Metacognitive Analysis in Attention**: Transformer experiments, current-state studies and controlled foresight/world-model experiments.

The common question is whether accessible features of an ongoing computation help predict **when more computation will improve a decision**, beyond a strong current-state reference. Predictive information, causal diagnostics and practical controller performance are reported separately.

## Start here

- [Research 002 index and current state](research002/README.md)
- [Latest completed main study: F1-A3 report](research002/experiments/F1_A3/main_v1/F1_A3_Report.md)
- [Working state and next research decision](research002/PROJECT_STATE.md)
- [Original Research 001 overview](research001_README.md)
- [Original TC-CVE notebook](TC_CVE_Experiment_001.ipynb)

## Snapshot — 4 October 2026

The completed F1-A3 main study is added under `research002/experiments/F1_A3/main_v1/`, with its own archive and restoration script. Twenty-three recovered source/study archives are preserved byte-for-byte as checked parts in `research002/archives/`. Their code, notebooks, protocols, reports, recorded audits and figures are also available as browsable files under `research002/experiments/`. Checkpoints and numerical arrays are restored from the archive parts with the script below; no account or original Drive access is needed after cloning.

```bash
git clone https://github.com/saim14/metacomputing001.git
cd metacomputing001
python research002/restore_archives.py --verify-only
python research002/restore_archives.py --study F1_A2
```

Each study retains its own recorded environment and reproduction instructions. Older absolute scratch paths in original records document provenance; use each study's relative README instructions to reproduce it. Completed experiments are fixed development evidence, not newly independent replications.

## Findings and limits

- Research 001 found reproducible trajectory-based prediction gains within its tested iterative-network architecture and task. Its scope limits are preserved in the original overview.
- T2-E2's Transformer observer study did not meet its 5% useful-history-gain threshold under stronger current-state controls. P1-A1 separately passed a bounded current-state sufficiency criterion: current-state prediction was noninferior to ordered history within a declared 0.01 Brier-loss tolerance in its two arithmetic networks. Read the individual reports for exact endpoints, exposed development conditions and scope limits.
- F1-A1 isolated harmful planning caused by world-model error in its toy-world intervention: +20.7 percentage points on average relative to exact dynamics.
- F1-A2 found a useful decision-specific cost-bias signal: 30.25% lower prediction MSE for privileged diagnostic information and 6.89% for an accessible estimate. The practical controller criterion failed; the estimator added runtime.
- F1-A3 is complete and audited: eight-candidate correction yielded 4.34% lower prediction MSE (below the fixed 5% gate), unresolved accuracy retention versus 32 candidates, and an inconclusive 0.51% charged-controller improvement. It was 28.42% faster than 32-candidate correction but 1.29–1.96 times slower than the reference. The primary candidate did not advance. Its full evidence is saved with a Colab notebook and hash-checked archive parts.

Nothing here establishes subjective awareness, general machine metacognition, chronological-history necessity, or a deployed compute-saving controller. Historical positive and negative results remain visible.

The original overview's license description is retained; this snapshot does not introduce a new blanket license over separately authored papers or source records.

