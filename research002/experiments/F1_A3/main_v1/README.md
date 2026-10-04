# F1-A3 main-v1 — completed and audited

Read [the report](F1_A3_Report.md) and [figure](F1_A3_Results.png). Protocol/code were frozen at commit `b549785c9a1552b9c071137b860f1791d1a1eee8` before execution.

**Decision: no advancement.** B8 lowered prediction MSE 4.34% (below the fixed 5% gate); its noninferiority to B32 was unresolved. Charged control improved 0.51% but its adjusted interval included zero. B8 was 28.42% faster than B32 and still 1.29–1.96 times slower than the strong reference.

## Reproduce or audit in Colab

Open [F1_A3_Colab.ipynb](F1_A3_Colab.ipynb). It defaults to synthetic checks and displaying the report; reconstruction of archived evidence is a separate explicit cell. No GPU is needed. Dependencies are NumPy, SciPy, matplotlib and threadpoolctl, already available in normal Colab runtimes. The recorded environment is NumPy 2.3.5 / SciPy 1.17.0 / Python 3.12.14; version differences may affect exact floating-point reproduction.

From this directory:

```bash
python study.py --fixtures-only
python restore_run.py
python audit.py
```

`restore_run.py` verifies every archive-part SHA-256, the joined ZIP hash and CRCs, and extracts only absent files. A differing existing file causes an error; saved results are never silently replaced. Selected small result/audit files are directly browsable in `run/`; bulk numerical evidence is in checked `archive_parts/`.

For a fresh deterministic reproduction, copy `study.py`, `config.json`, `F1_A3_Protocol.md`, `audit.py` and the entire `deps/` directory to a new directory, then run `python study.py`. Use `--resume` only for that incomplete run with unchanged input hashes. A deterministic rerun is not independent confirmation. Different seeds or designs require a separate protocol, never modification of these recorded results.

The full archive includes trajectories, calibrated dependencies, all selected predictor coefficients (including every tuning-grid candidate), observations, decision outcomes, raw timing records, audit and human-readable report. No access to prior Drive or Library files is required. The calibration prototype remains in the parent directory unchanged.
