# P1-A1 eligibility amendment — 1.0.0 to 1.1.0

Recorded on 15 September 2026, after the v1.0 pre-fit support audit and before any
natural observer fit or synthetic-control observer fit. The original attempt is
preserved, including its frozen protocol, code, support counts and stop record.
The original input files are identical to those under this version's `inputs/`;
they are not duplicated under `stopped_v1/`. No earlier result is relabeled.

Version 1.0 required both absolute class-support counts and a 10% minority-class
fraction in fit and tuning partitions. The step-4 fit partitions had minority
fractions between approximately 8% and 11%, and tuning between 6% and 15%.
Several therefore failed that added fraction rule. All 12 model/step/split cells
passed the separately specified counts: at least 50 fit and 20 tuning examples
per class, and 25 fit and 10 tuning contributing families per class. Across
step-4 cells, the smallest observed minority counts were 196 fit examples from
54 families, and 37 tuning examples from 11 families. The two minima may describe
different splits; the full table is preserved in `stopped_v1/panel_run/eligibility.json`.

For this exploratory comparison, absolute event and family counts are a more
direct feasibility criterion than carrying the earlier 10% readiness cutoff into
smaller observer partitions. Version 1.1 retains those absolute requirements and
reports prevalence without imposing an additional fraction cutoff. It retains
all original models, steps and random splits; no resampling to obtain favorable
class balance, class reweighting, extra training, outcome-dependent predictor
selection, or evaluation-threshold relaxation is introduced.

This is an explicit data-informed development amendment, not an untouched
preregistration. Its rationale concerns feasibility, not evidence for the
hypothesis. Imbalanced classes still matter: the constant-prevalence baseline,
relative usefulness threshold, family uncertainty intervals, secondary balanced
accuracy, and synthetic sensitivity requirements remain in force. A sparse
minority class may still make the final finding inconclusive.

The new protocol and source are frozen in a new output directory before observer
fitting. The old `INCOMPLETE.json` is retained as a stopped eligibility attempt;
it is not a numeric solver failure. No natural or synthetic-control candidate
budget was consumed by the old attempt (12 small method-fixture ridge solves
were executed separately).
