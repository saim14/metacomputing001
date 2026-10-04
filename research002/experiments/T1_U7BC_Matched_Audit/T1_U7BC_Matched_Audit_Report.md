# T1-U7BC-M1 — matched-example audit

**Research lead:** Saim 13.02  
**Completed:** 11 September 2026  
**Status:** Complete. All-seed rescue rejected on both original evaluation suites.

Both U7B and U7C pass exactly 3 of 6 exposed seeds on each suite. Seeds 503, 607, and 709 pass; seeds 811, 907, and 1009 fail both performance gates. Matching evaluation examples corrects the intervention-effect comparison without changing the failed advancement decision.

## Question and scope

Does the U7C extension intervention repair all exposed seeds when the frozen U7B and U7C checkpoints are evaluated on identical examples? This is a retrospective audit of exposed models and evaluation suites. Its procedure was fixed before the new cross-evaluations; it is not an independent confirmatory experiment.

The original U7B suite uses offsets 37/43 for seeds 503/607/709 and 83/89 for 811/907/1009. The original U7C suite uses 37/43 for every seed. Both versions receive the exact same generated arrays within each seed/suite comparison. The two suite labels are aliases for the first three seeds and must not be counted as independent evidence.

Each clean set contains 4,000 examples; each causal suite contains 2,000 clean/counterfactual pairs and their two edge corruptions. The paired-counterfactual metric requires both members of a pair to be correct. Eighteen unique model/suite evaluations produce 24 labeled rows. All exposed seeds were included.

## Unchanged gates

| Gate | Threshold |
|---|---|
| Clean final accuracy | ≥95% |
| Paired-counterfactual accuracy | ≥90% |
| Each edge-corrupted accuracy | 40–60% |
| Remaining value | At least two steps with mean and sample SD each ≥0.02 nats |

All models pass the edge-control and remaining-value checks on each suite. The three weak seeds fail both performance thresholds on each suite. Remaining value is true-label NLL at the current step minus final NLL; its availability does not show that history predicts it better than current observables.

## Matched performance

Each cell below reports accuracy as a percentage. Deltas are U7C minus U7B in percentage points (pp).

### Original U7B evaluation suites

| Seed | B clean | C clean | Δ clean (pp) | B paired CF | C paired CF | Δ paired CF (pp) | B / C gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 503 | 99.250 | 99.200 | -0.050 | 99.15 | 98.75 | -0.40 | Pass / Pass |
| 607 | 100.000 | 99.750 | -0.250 | 100.00 | 99.75 | -0.25 | Pass / Pass |
| 709 | 99.800 | 100.000 | +0.200 | 99.65 | 99.80 | +0.15 | Pass / Pass |
| 811 | 71.225 | 70.675 | -0.550 | 56.60 | 58.30 | +1.70 | Fail / Fail |
| 907 | 72.900 | 73.650 | +0.750 | 64.55 | 66.30 | +1.75 | Fail / Fail |
| 1009 | 70.925 | 71.450 | +0.525 | 58.95 | 58.35 | -0.60 | Fail / Fail |

### Original U7C evaluation suites

| Seed | B clean | C clean | Δ clean (pp) | B paired CF | C paired CF | Δ paired CF (pp) | B / C gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 503 | 99.250 | 99.200 | -0.050 | 99.15 | 98.75 | -0.40 | Pass / Pass |
| 607 | 100.000 | 99.750 | -0.250 | 100.00 | 99.75 | -0.25 | Pass / Pass |
| 709 | 99.800 | 100.000 | +0.200 | 99.65 | 99.80 | +0.15 | Pass / Pass |
| 811 | 70.400 | 70.000 | -0.400 | 56.90 | 58.30 | +1.40 | Fail / Fail |
| 907 | 74.150 | 73.375 | -0.775 | 59.55 | 61.35 | +1.80 | Fail / Fail |
| 1009 | 70.775 | 71.100 | +0.325 | 60.90 | 61.15 | +0.25 | Fail / Fail |

Rows 503/607/709 duplicate the first table because their original suites are identical.

## What the corrected comparison changes

- Seed 811: paired-CF gains of +1.70 and +1.40 pp accompany clean-accuracy losses of −0.550 and −0.400 pp. No rescue.
- Seed 907: paired-CF improves by +1.75 and +1.80 pp. The historical mismatched comparison had suggested −3.20 pp; that earlier decline cannot be interpreted as the auxiliary intervention’s effect. Clean-accuracy changes have opposite signs across suites (+0.750 and −0.775 pp). No rescue.
- Seed 1009: clean accuracy improves by +0.525 and +0.325 pp; paired-CF changes are −0.60 and +0.25 pp. No rescue.
- Across the three previously failing seeds, mean matched paired-CF gains are +0.95 pp on the original U7B suites and +1.15 pp on the original U7C suites. Corresponding mean clean changes are +0.242 and −0.283 pp. These are descriptive means over three exposed seeds, not estimates of population-level repair effectiveness.

The result does not imply that auxiliary supervision has no effect. It changes individual predictions, with modest and metric-dependent movements, but it fails the specified all-seed rescue criterion. No significance or equivalence claim is made. `paired_effects.csv` records C-only and B-only correct counts so every accuracy difference can be reconstructed as (C-only − B-only) / N.

## Verification and execution

- The downloaded original U7B and U7C ZIPs match their published SHA-256 values. Their inner manifests verify 50 and 72 entries, respectively.
- The inference, data-generator, gate, and trajectory source files used by both experiments are byte-identical.
- All twelve native checkpoint endpoints reproduce their original evaluation arrays and every saved causal/gate/value metric. The largest native probability-array absolute difference is 0.0.
- Both models receive identical arrays within a comparison, and those arrays are checked unchanged after inference. Frozen input files are checked before and after the run.
- The inference-only run completed in 133.13 seconds on this CPU environment. Colab runtime may differ.
- Training updates: 0. Probe fits: 0. Reserved seeds opened: none. No source results were overwritten.

## Research decision

Close this audit and retain the closed training-repair branch. U7D remains paused; T1-U8 remains blocked; reserved seeds 1103/1201/1301 remain sealed. The original failed advancement decisions remain unchanged. This audit tests the training intervention, not the history-advantage hypothesis.

The next design decision is the observable supplied to the current-state predictor. The recovered code uses a 32-dimensional CLS representation plus two class probabilities, entropy, and margin; it does not expose the whole residual tensor. The accompanying `NEXT_STUDY_DESIGN.md` specifies the distinction between a limited-CLS claim and a stronger all-token current-state comparison, along with target, capacity, temporal-control, split, and stopping-rule requirements. It authorizes no further training or probe execution.

## Reproduce

Open `T1_U7BC_Matched_Audit_Colab.ipynb` in Colab and upload `T1_U7BC_Matched_Audit.zip`. It verifies the package and runs the audit in a new output directory. The compact bundle includes twelve frozen checkpoints, original configurations and reference tables, exact compact native reference arrays, original inference/generator source, audit specification, runner, per-example correctness arrays, results, and integrity records. Full training histories and full hidden-state trajectories remain in the original bundles.

## Sources

- [Completed U7C report](https://drive.google.com/file/d/1k-ZR3KaprlnSEiOXjXSFEibPdRwKlaTl/view).
- [Original U7B report](https://drive.google.com/file/d/1M6O9YX-XcZr16Lo2nKNuQPc5jcnVuvF0/view).
- [Original U7C closure audit](https://drive.google.com/file/d/1vT-i5N-1v5UwQ90LEgVkTWVOoQ0FYF3b/view).
- [Original T1 history comparison](https://drive.google.com/file/d/1e4x-TVQxDMjAx9nAn1KOUTyh2hKpwdnc/view).
- Verified original bundles, `AUDIT_SPEC.json`, `SOURCE_PROVENANCE.json`, and `results/audit_result.json` provide the computational provenance for the new results.
