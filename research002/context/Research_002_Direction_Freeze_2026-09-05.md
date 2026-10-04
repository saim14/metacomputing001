# Research 002 — Research direction freeze

Date: 5 September 2026  
Research lead: Saim 13.02  
Status: Direction and development budget frozen; T1-U7C complete, audited, and closed with 3/6 development seeds passing. Reserved seeds remain sealed; T1-U8 and probes remain blocked.

This records the accepted decision to preserve the original question, close T1-U7C, and prevent repeated training repairs from replacing the hypothesis test. It is not a completed, executable T1-U8 preregistration. The 5 September closure recovered all six original endpoints and audited their saved checkpoints and measurements. No training, confirmation evaluation, or probe fitting was performed during closure.

## Original question and correction

Does ordered computational history improve prediction of a Transformer's future decision change or future loss reduction beyond its current representation and prediction, under comparable probe capacity and evaluation conditions?

The original T1 baseline report, dated 1 September 2026, already included both targets and used a two-hop retrieval task. Therefore, describing the subsequent work simply as a move to two-hop retrieval was inaccurate. The relevant changes include the redesigned shortcut-controlled task, a six-step weight-tied architecture, a different training regime, and promotion of continuous loss reduction from a secondary target to the main target of the later branch.

T1's negative findings remain part of the evidence. A later positive result would apply to its specified task, architecture, training regime, observable representation, and probe family. It would not retrospectively confirm the earlier decision-flip experiment or, by itself, establish causal metacognition.

## Evidence and outstanding work

| Stage | Available evidence | Decision |
|---|---|---|
| T1 baseline | No robust history advantage on decision flips or future NLL reduction in the saved report | Preserve the negative result and its limitations |
| T1-U7 | Seeds 607 and 709 passed; 503 failed clean and paired accuracy | Task competence is possible; this training schedule missed its all-seed criterion |
| T1-U7B | Development seeds 503/607/709 passed; newly opened 811/907/1009 all failed | Six seeds are now exposed; confirmation failed |
| T1-U7C | All six original outputs recovered and independently reproduced from their saved final checkpoints: 503/607/709 pass; 811/907/1009 fail | Complete; the all-six-seed repair criterion failed (3/6 pass) |
| T1-U7C seeds 907/1009 | Seed 907: clean 0.73375, paired CF 0.6135, edge controls 0.4915/0.5100, four value steps. Seed 1009: clean 0.71100, paired CF 0.6115, edge controls 0.5020/0.5050, three value steps | Both complete at 6,000 updates; both fail clean and paired-CF gates |

T1-U7C seed 811 had clean accuracy 0.7000, paired-counterfactual accuracy 0.5830, edge-corruption accuracies 0.5085/0.5105, and four valid value steps. These results were independently recomputed from the recovered original final checkpoint during closure. Its failure does not identify insufficient supervision strength as the cause.

T1-U7B results for seeds 907 and 1009 must never be substituted for their missing T1-U7C results.

## Immediate development budget and stop rule

1. Completed: original T1-U7C runs for seeds 907 and 1009 were recovered at their fixed 6,000-update endpoints. No completed work was rerun. All six final checkpoints reproduced their original trajectories and causal metrics. The checkpoints contain weights only; no optimizer/RNG resume was attempted.
2. Preserved the original executable T1-U7C protocol: auxiliary weight 0.005 during updates 4,001–6,000, with the same first 4,000 training updates, streams, model, and gates. Audit caveat: promoting seeds 811/907/1009 to development changed their active clean/causal evaluation offsets from 83/89 in T1-U7B to 37/43 in T1-U7C. Their historical differences are paired by model seed, not by evaluation example. Original outputs remain unchanged; small differences must not be treated as pure intervention effects.
3. Budget for new repair configurations under this decision: **zero**. Do not launch T1-U7D, raise auxiliary weight to 0.02, add training phases, alter gates, or replace failed seeds.
4. Completed: T1-U7C is closed with all six per-seed metrics, original seed-paired T1-U7B differences, and the evaluation-sample limitation. No run is missing. The recovery audit passed all 16 existing direct tests and verified CSV/JSON consistency, unchanged original output hashes, and the unopened confirmation boundary.
5. The all-seed repair outcome is failed. This development branch is now stopped under the present decision. Any future repair requires a separate, versioned rationale and fixed budget before execution; it is not an automatic continuation.

## Existing gates retained

Every seed must satisfy every gate. A pooled mean cannot compensate for a failed seed.

| Gate | Unchanged threshold |
|---|---|
| Clean final accuracy | At least 0.95 |
| Paired-counterfactual accuracy | At least 0.90 |
| First-edge corrupted accuracy | Within [0.40, 0.60] |
| Second-edge corrupted accuracy | Within [0.40, 0.60] |
| Remaining computation value | At least two intermediate steps with both mean and sample SD at least 0.02 nats |

Remaining computation value is V_l = NLL_l − NLL_L. It is a supervised analysis target, not an input available to a label-free controller. Its measurable variation establishes target availability, not a history advantage.

## Hypothesis-test boundaries

The later continuous-value branch retains V_l as its primary target. Decision flips remain separately reported diagnostics in that branch; any renewed confirmatory decision-stability test needs its own declared target and validity criteria.

The decisive comparison remains current representation plus current prediction/confidence versus the same observables plus computational history. Retain the original control families: dimension-matched current state, per-example shuffled past, sample-mismatched past, and history residualized from current state. Dimension matching alone does not establish matched capacity: specify parameter count, scaling, regularization, fitting budget, and tuning budget. Any chronology claim must additionally survive a valid temporal-order control.

Before any future probe execution, freeze the exact observable state (including whether it covers the query token or all tokens), history window, feature construction, probe models, matched budgets, data splits, target definitions, primary metric, smallest effect of interest, eligible steps, and uncertainty/aggregation rules. Keep related examples and all steps of an example within the same split. Fit preprocessing and residualization on training data only. Future states and label-derived losses must not enter predictor inputs.

Those executable choices were not all available in the recovered records and are not represented here as already preregistered. Do not select them by looking at confirmation outcomes. No reliable advantage, a sufficiently precise negligible effect, and an invalid base-model gate are distinct outcomes and must be reported distinctly.

## Confirmation boundary

Seeds 1103/1201/1301 remain sealed, as reported in the latest checkpoint. T1-U8 and trajectory probes remain blocked. A future all-exposed-seed pass would only permit the reserved training confirmation stage; it would not itself confirm the history hypothesis. A confirmation failure ends that protocol without tuning on or replacing the failed confirmation seeds. Any eventual history claim requires its own frozen held-out probe evaluation.

## Sources and closure audit

- [T1_BASELINE_REPORT.md](https://drive.google.com/file/d/1e4x-TVQxDMjAx9nAn1KOUTyh2hKpwdnc/view?usp=drivesdk), read directly on 5 September 2026.
- T1_U7_REPORT.md, saved 3 September 2026; read in full on 5 September.
- T1_U7B_REPORT.md, saved 4 September 2026; read in full on 5 September.
- The user-supplied 4 September checkpoint and subsequent direction discussion supply the T1-U7C status and reserved-seed information.

An earlier search did not recover the exact T1-U7C files. On continuation in the original workspace, all six completed runs, original code, and frozen configuration were recovered. `T1_U7C_REPORT.md` and `recovery_audit.json` document closure on 5 September 2026. The complete portable bundle preserves the original records and audit code. No experiment is running as part of this closed branch.
