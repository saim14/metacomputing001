# F1-A3 — Methods development

Started 4 October 2026. **Main experiment: not run. Protocol: draft.**

The calibration-only pilot uses 1,536 fit transitions and 512 held-out transitions from the inherited F1-A2 calibration bank. It accesses no F1-A2 planning evaluation outcomes. Both estimators use only fit-transition residuals; the held-out transition error is an engineering diagnostic.

| Full-error neural model | Original local-linear RMSE | Compact quadratic RMSE | Zero correction RMSE |
|---|---:|---:|---:|
| 3109 | 0.02185 | 0.03013 | 0.03008 |
| 3119 | 0.02188 | 0.03014 | 0.03009 |
| 3137 | 0.001775 | 0.001972 | 0.002685 |

The compact map was about 72–98 times faster **for the residual-correction query batch alone**. It was less accurate in all three conditions and marginally worse than zero correction in two. This is evidence of a speed–accuracy tradeoff in this prototype, not of better planning or lower end-to-end runtime.

The next prospective comparison should prioritize the original local-linear correction on eight fixed candidate ranks, versus its full 32 candidates, to reduce corrected-rollout calls from 320 to 80. Candidate-subset bias summaries and primary gates still need freezing before fresh scene evaluation. The compact method remains an engineering comparison; it is not selected as a successful controller.

- `cheap_bias.py`: quadratic residual map, deterministic candidate selection and synthetic checks.
- `methods_pilot.py`: calibration-only holdout benchmark.
- `methods_pilot.json`: exact diagnostic results, timings and source hashes.
- `F1_A3_Protocol_Draft.md`: prospective design and limits.
- `F1_A3_Colab.ipynb`: restore dependencies and run synthetic checks; reproducing the pilot requires a separate output directory so the recorded result is preserved.

```bash
python research002/restore_archives.py --study F1_A2
python research002/experiments/F1_A3/cheap_bias.py
```

Running `methods_pilot.py` refuses to replace an existing pilot result. Copy the F1-A3 sources into a new sibling directory, leave the saved JSON behind, and run the copied script to repeat the benchmark. The F1-A2 inputs are read-only.
