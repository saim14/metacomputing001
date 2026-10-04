# T2-C1 — Synthetic sensitivity calibration

Research 002 · 12 September 2026 · Completed

The complete history-advantage screen passed **0/24 null panels, 0/24 panels with a known 0.01 population R² gain, and 24/24 panels at each of 0.05 and 0.10**. The prespecified null and strong-signal diagnostic checks passed. This establishes useful sensitivity for these larger, simple synthetic signals and poor acceptance of the small signal under the complete rule. It adds no evidence that natural Transformer histories improve prediction.

## Why this experiment

The completed [T2-E1 observer study](https://drive.google.com/file/d/17M_RRAgFbWj3SeYMuby9-IBgibgQbJre/view) found no robust history advantage on the frozen U7C models. Its full-state history versus capacity-matched current-state comparison had mean ΔR² −0.000884, with a conditional 95% interval of approximately [−0.039843, 0.010258]. That is neither confirmation nor an equivalence result.

T2-C1 asks a narrower methodological question: can the probe calculation and acceptance rule recover a deliberately known signal? It uses fresh synthetic observations, the earlier step-2 dimensions, and the earlier ridge fitting and screening rules. It does not reproduce the empirical task, neural representation geometry, multiple-step aggregation, or NLL target distribution.

The protocol was frozen at **2026-09-12 03:57:50 UTC**, before the first calibration run. Its SHA-256 is `bf4a889970419aa511eec51a6a78fc7e1dcfc215022b9f688192f6835ead8181`. All 24 panels ran once; the run completed at 04:04:49 UTC in 43.51 seconds on this CPU environment. There were no adaptive repeats or changes to the scientific protocol.

## Design and known ground truth

There are 24 independent panels, each containing six independent synthetic datasets. Each dataset has 3,000 families of two observations: 1,600 training families, 400 validation families, and 1,000 test families. Both members of a family remain in the same split. Each current coordinate X, historical coordinate H, and noise coordinate ε has a standard Gaussian marginal, constructed from independent family and member components to give within-pair correlation 0.5. Coordinates and X/H/ε blocks are independent.

For each δ in {0, 0.01, 0.05, 0.10}, the target is

$$Y_\delta = 1 + \sqrt{0.5}X_0 + \sqrt{\delta}H_0 + \sqrt{0.5-\delta}\,\epsilon.$$

By independence, the target variance is 0.5 + δ + (0.5 − δ) = 1. The oracle current-state conditional mean is 1 + √0.5 X₀, with population R² 0.5. The oracle given both current state and history also includes √δ H₀, giving population R² 0.5 + δ. The true population incremental R² is therefore exactly δ. All four scenarios within a dataset share X, H, and ε; the scenario outcomes are paired.

| Probe | Inputs | Fitted coefficients, including intercept |
|---|---|---:|
| Current linear | 388 current coordinates | 389 |
| Current RFF | Current coordinates plus 36 random Fourier features of current state | 425 |
| History | Current coordinates plus 36 past coordinates | 425 |

The history and RFF probes have equal coefficient counts, but different feature families. All use training-only feature means and standard deviations and training-only target centering. Validation MSE chooses ridge α from {0.01, 0.1, 1, 10, 100}; exact ties prefer larger α. No train-plus-validation refit occurs. Feature decompositions are reused across targets, with numerical equivalence verified against the original calculation.

For history against **both** current baselines, each panel must satisfy mean held-out ΔR² ≥0.01, a paired-family bootstrap 95% lower bound above zero, positive ΔR² in all six datasets, and nonnegative mean MAE improvement. Each dataset contributes equally. The 1,000 bootstrap draws resample test families with both members together, holding probes fixed, using common draws across scenarios and comparisons.

The fixed diagnostic requires at most one passing null panel and at least 22 passing panels at δ = 0.10. There is no required success count for the intermediate signals.

## Results

| True population ΔR² | Full screen passes | Detection fraction, 95% Wilson Monte Carlo interval | Mean fitted ΔR² vs current linear | Mean fitted ΔR² vs current RFF | Mean held-out oracle ΔR² |
|---:|---:|---:|---:|---:|---:|
| 0 | 0/24 | 0% [0%, 13.8%] | −0.008239 | −0.002026 | 0 |
| 0.01 | 0/24 | 0% [0%, 13.8%] | +0.003528 | +0.009721 | +0.009900 |
| 0.05 | 24/24 | 100% [86.2%, 100%] | +0.050470 | +0.056616 | +0.049803 |
| 0.10 | 24/24 | 100% [86.2%, 100%] | +0.109159 | +0.115253 | +0.099796 |

These intervals describe Monte Carlo uncertainty across the 24 independent simulated panels. They are distinct from the conditional bootstrap intervals used inside each panel. Zero null passes does not establish population false-positive probability below 5%; all 24 successes likewise does not establish certain detection.

The fitted-probe gains need not equal the known oracle gains. In particular, the fitted comparison can exceed δ because finite-data estimation errors differ between the fitted baseline and history probe. The population oracle relationship applies to the conditional means, not arbitrary fitted estimators.

### What blocked the 0.01 signal

| Requirement at δ = 0.01 | Panels passing vs current linear | Panels passing vs current RFF |
|---|---:|---:|
| Mean ΔR² ≥0.01 | 0/24 | 13/24 |
| Bootstrap lower bound >0 | 12/24 | 23/24 |
| Positive ΔR² in all six datasets | 5/24 | 16/24 |
| Mean MAE no worse | 21/24 | 24/24 |
| All four requirements for that comparison | 0/24 | 13/24 |

The signal was present in fitted scores, especially against the capacity-matched RFF baseline. Nevertheless, every panel failed the effect threshold against current linear: those panel means ranged from −0.002714 to +0.008189. The full conjunction therefore passed none. This identifies the observed decision bottleneck; it does not establish a unique cause such as insufficient sample size.

The planted population effect of 0.01 is also exactly the screen's fitted-effect threshold. High acceptance at that boundary is not guaranteed even when a signal exists. This experiment calibrates the full acceptance rule, not merely a zero-effect significance test.

## Verification and completeness

- 144 distinct synthetic datasets, 864,000 distinct observations, and 576 dataset-scenario evaluations completed. Scenarios reuse observations and must not be counted as independent new samples.
- 1,728 target-specific ridge selections evaluated 8,640 α candidates using 432 shared feature factorizations.
- Four target fits matched the original T2-E1 ridge code and 20 independent scikit-learn ridge candidates. Maximum prediction differences were below 1.4×10⁻¹⁴ on the verification fixture, which included rank-deficient and constant features.
- Test-target and test-feature perturbations did not change fitting, preprocessing, or α selection; the exact-tie rule passed.
- All 576 dataset score rows, 96 panel decisions, 1,728 α choices, and four scenario summaries were checked from saved records. All 24,000 panel bootstrap draws were checked without refitting.
- Protocol and reference hashes remained unchanged. The preflight record documents one floating-point boundary assertion correction in the verification script before any calibration panel ran; there was no change to the experiment.

The bundle contains the frozen protocol and references, generator and analysis code, every dataset and panel metric, validation choices, generation hashes, panel bootstrap arrays, verification scripts and outputs, execution log, and figure. Raw feature arrays and fitted probe weights are regenerated from the fixed seeds rather than stored. The Colab notebook verifies the bundle and reproduces the run into a new output directory.

## What changes in our research interpretation

Within this synthetic design, the screen successfully accepts larger accessible linear history signals and fails to accept the smaller planted signal. That supports interpreting a failed screen cautiously; it does not show that a small natural history advantage exists or that an actual neural advantage of 0.05 would necessarily be detected.

The calibration covers one step, Gaussian independent coordinates, fixed pair correlation, and only 24 independent panels. Actual neural states have different dependencies and targets; this is not a power estimate for T2-E1. No temporal-order claim is tested because there is only one historical slot.

T2-E1 remains closed with no robust observed history advantage. The T1 training-repair branch remains closed; T1-U7D and T1-U8 are not launched, and confirmation seeds 1103/1201/1301 remain sealed. No Transformer checkpoint was loaded or trained in T2-C1.

The bounded calibration is complete. A future empirical test would need to specify its meaningful effect size and representation-matched sensitivity assessment before opening new outcomes. Today's results provide no basis for loosening yesterday's acceptance rules or reopening its test data.
