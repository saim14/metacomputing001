# T2-E1 — observer sufficiency

Read `T2_E1_Report.md` first. The fixed exploratory study is complete; no robust history advantage met the frozen criterion.

## Reproduce in Colab

Open the enclosed notebook and upload `T2_E1_Observer_Study.zip`. It verifies every package entry before running code. CPU is sufficient. All model weights and data-generation code are included.

## Reproduce locally

With Python 3.12, install `requirements.txt`, then run:

```bash
python verify_methods.py
python run_study.py --out reproduced_results
```

The output directory must be new. Original distributed results remain intact. The study regenerates exact fresh data from its fixed seeds and exclusions. Intermediate full residual tensors and fitted probe coefficients are reproducible from the frozen model checkpoints, code, exact data, and saved alpha selections; they are not retained separately.

`verify_results.py` checks the distributed `results` directory without fitting probes. `make_figure.py` regenerates the scientific figure from saved comparisons.

## Key files

- `STUDY_PROTOCOL.json`: scope, observables, targets, controls, budgets, and screen frozen before the research fits.
- `frozen/`: original model code, six U7C checkpoints, original training protocol, and prior audit record.
- `results/data/`: exact generated paired examples, ordered training/validation/test.
- `results/predictions/`: held-out targets, predictions, and training-derived eligibility.
- `results/probe_metrics.csv`: all 336 cells, including ineligible steps and explicit aliases.
- `results/validation_selection.json`: all five alpha scores and the selected alpha per cell.
- `results/seed_comparisons.json` and `panel_comparisons.json`: paired effects and conditional intervals.
- `results/result_verification.json`: independent score and aggregation checks.
- `execution.log`: the single completed study run.

Reserved seeds are not instantiated. No base-model training runs. This is an exploratory fixed-model comparison, not an untouched-seed confirmation.
