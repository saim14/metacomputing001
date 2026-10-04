# Reproduce T2-E2

The archive contains code, frozen model checkpoints, prior-example exclusion tokens, the immutable scientific protocol, development calibration results, fitted observer weights, main results when the development gate passed, and all verification records. No GPU or additional model download is needed. Use Python 3.12; the original environment used Python 3.12.14.

Install the pinned dependencies and verify the distributed outputs:

```bash
python -m pip install -r requirements.txt
python verify_methods.py
python verify_results.py --folder development --development
python verify_results.py --folder results
```

The final command applies only when the main study completed. To reproduce, work in a fresh extracted copy. First verify `PACKAGE_SHA256.json`, then rename the existing `MAIN_FREEZE.json` to `ARCHIVED_MAIN_FREEZE.json` in that disposable copy. The original record remains in the archive. Run:

```bash
python -u run_experiment.py development --development reproduced_development
python verify_results.py --folder reproduced_development --development
```

Read `reproduced_development/development_result.json`. Only if `go_to_main` is true, continue:

```bash
python run_experiment.py freeze --development reproduced_development --out reproduced_results
python -u run_experiment.py fit --development reproduced_development --out reproduced_results
python -u run_experiment.py evaluate --development reproduced_development --out reproduced_results
python verify_results.py --folder reproduced_results
```

Every output phase refuses an existing output directory. The main fitting phase performs Transformer inference only on its training and validation examples. All probe choices and their file hashes are fixed before the evaluation phase reads the natural test outcomes. The new main freeze records the reproduction's timing and development result hash; those provenance fields will differ from the archived run.

The notebook automates this sequence and compares the reproduced metric tables with the archived tables. The complete experiment normally requires several CPU minutes; runtime depends on the environment. Small floating-point and timing differences across BLAS implementations are possible, especially for iterative neural-network fitting. A difference outside the notebook's tolerance is flagged for inspection rather than silently ignored.

Raw intermediate Transformer tensors are regenerated from the included checkpoints and exact saved token arrays. Observer model files contain numeric arrays, not executable pickle objects. The same six development-exposed model seeds are used; reserved seeds are never loaded or trained. A deterministic reproduction is not an independent scientific replication.
