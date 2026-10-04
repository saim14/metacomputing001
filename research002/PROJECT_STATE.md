# Research 002 — working research state

Updated 4 October 2026. Author: Saim 13.02.

**Active question:** Which information about an ongoing computation is worth acquiring before deciding to continue or act?

**Latest completed study:** [F1-A3 main-v1](experiments/F1_A3/main_v1/F1_A3_Report.md), complete, independently reconstructed and saved. **No training or simulation remains running for this study.** No follow-up experiment has been executed.

| Evidence line | Current conclusion | Next constraint |
|---|---|---|
| Transformer history / T2-E2 | No qualified practical history gain under tested stronger observers | Keep completed results and sealed reserves fixed |
| Present-state arithmetic / P1-A1 | Bounded current-state sufficiency passed in two trained networks under the amended development protocol | Do not generalize to awareness or all networks |
| World-model intervention / F1-A1 | Model error increased harmful extra planning in the toy world | Separate causal diagnostics from deployable models |
| Decision-specific cost bias / F1-A2 | Accessible bias improved prediction; practical controller criterion failed | Include acquisition cost and runtime |
| Sparse bias correction / F1-A3 | Small prediction signal, cheaper than full correction; no qualified practical gain | Close this run; no test-set-driven search over subset sizes |

## Decision log

- Preserve the fast quadratic pilot as a negative engineering result. Its speed did not compensate for lost residual accuracy.
- Freeze one B8-vs-B32 main comparison before outcomes; commit `b549785c9a1552b9c071137b860f1791d1a1eee8`.
- Share fresh trajectories and splits across every arm and model intervention. Checkpoint each completed condition; use atomic writes and hash-guarded resume.
- Complete the fixed budget: 2,400 scenes, 1,008 candidate fits, 84 selected pipelines, 5,600 online replays.
- Close F1-A3 after all three efficacy gates and the reference-runtime gate were missed. A positive small MSE improvement is retained in the report; no evidence is relabeled as practical success.
- Keep new scene evaluation conditional on the three inherited models. A broad claim eventually needs untouched model seeds and task distributions.

## Proposed F1-A4 — not run or frozen

Test a 2×2 design: current-state versus current-state plus incumbent-only bias, crossed with MSE-trained prediction versus a stop/continue loss weighted by the cost of making the wrong decision. The point is to distinguish the value of extra information from the quality of the decision rule. B1 is only an exploratory motivation; it has not established useful control or runtime savings.

Before execution, freeze four arms, the continuation-cost exchange rate, fit/tune policy selection, fresh scene seeds and counts, a single primary policy contrast with multiplicity treatment for any additional primary contrasts, and a measured-runtime gate. Do not compare many subset sizes, add flow matching, retrain world models or extend the scene count in the same test. If a controller earns advancement, then allocate a separate untouched new-model confirmation budget.

## Continuation checklist

1. Read this file and the latest report; inspect completion and audit markers before running anything.
2. Never restart a completed run. Restore absent evidence with its hash-checking script.
3. Use the frozen protocol for interpretation; diagnostic successes do not override primary failures.
4. Update this state and the index only after evidence is audited and saved.

Entry points: [research index](README.md), [F1-A3 protocol](experiments/F1_A3/main_v1/F1_A3_Protocol.md), [Colab notebook](experiments/F1_A3/main_v1/F1_A3_Colab.ipynb), [audit](experiments/F1_A3/main_v1/run/audit.json).
