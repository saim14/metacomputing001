# F1-A1 study release

Run and audited on 17 September 2026; reported and packaged on 18 September 2026.
Read `F1_A1_Report.md` for interpretation and `F1_A1_Protocol.md` for the
prospective frozen design. The protocol and experiment source were not changed
after outcomes. `make_report.py` is post-run presentation code, not part of the
fitted experiment.

## Reproduce the audit without fitting

Use Python 3.12 (recorded experiment version: 3.12.14). Install the versions in
`requirements.txt` in a separate environment, then run from this directory:

```bash
python audit.py
```

This reconstructs all 1,344 tuning scores and 112 selected pipelines from saved
coefficients and verifies outcomes, prefix inputs, selections and intervals.
It does not train neural models or fit observers. It rewrites `run/audit.json`
with an updated elapsed time; preserve the original release if exact byte
identity matters. Floating-point differences can occur across BLAS builds.

```bash
python make_report.py
```

This regenerates the figure and report from the existing result and audit.
No internet is needed once the dependencies are installed.

## Reproduce a fresh numerical run

The frozen script refuses to run when a `run/` directory already exists.
Create a separate empty directory and copy `study.py`, `config.json`,
`F1_A1_Protocol.md` and the complete `prior/` directory there. Do **not** copy
the top-level `run/`. Then execute:

```bash
python study.py
```

The `prior/run/` directory must be retained: it contains the three frozen
neural checkpoints and their original provenance. This fresh execution trains
1,344 small ridge readouts but does not retrain the neural world models.
Expect timestamps and runtime measurements to differ; the numerical design,
random seeds, selected coefficients and predictions should reproduce within
floating-point tolerance. Copy `audit.py` into that fresh directory to audit
its results. This is a reproduction, not an additional independent study.

## Contents

| Path | Purpose |
| --- | --- |
| `study.py`, `config.json` | Frozen experiment and candidate grid |
| `F1_A1_Protocol.md` | Prospective design and stop rule |
| `prior/` | Exact F1-A0 source, configuration, checkpoints, reports and provenance used here |
| `run/manifest.json` | Hashes and environment recorded before new outcomes/fits |
| `run/scenarios.npz` | All scenes and fit/tuning/evaluation IDs |
| `run/calibration_*.npz` | Independent reliability bank and per-condition one-step errors |
| `run/data_*.npz` | Prefix features, U3/U6 plans, true costs, signed targets and estimated costs |
| `run/controls_*.npz`, `run/expanders_*.npz` | Mismatch/shuffle mappings and fit-only expansion normalization |
| `run/observer_*.npz` | All candidate coefficient vectors, normalizers and selected indices |
| `run/candidate_grid.json` | All 1,344 candidate tuning scores |
| `run/evaluation.npz` | Selected tuning/evaluation predictions, targets, baselines and scene IDs |
| `run/result.json` | All headline, per-condition and descriptive outcomes |
| `run/runtime.json` | Timing repetitions and online replay comparisons |
| `audit.py`, `run/audit.json` | Post-run reconstruction and verification results |
| `make_report.py` | Result-derived figure/report generation |
| `F1_A1_Report.md`, `F1_A1_Results.png` | Human-readable findings |
| `package_study.py`, `SHA256SUMS.txt` | Reproducible packaging logic and file hashes |

No unseen T1 seeds or P1 reserve families are included. No neural retraining,
future F1-A2 experiment, or publication/deployment operation is part of this
release. Bytecode, residual temporary writes and the ZIP itself are excluded
from the archive. A residual incomplete `data_exact.npz.tmp` was not used by
any analysis; the completed `data_exact.npz` is included and validated.

## Integrity

In a shell providing `sha256sum`, run immediately after extracting the release:

```bash
sha256sum --check SHA256SUMS.txt
```

This checks release bytes, while `python audit.py` checks reconstruction of the
scientific results. Regenerating the report or audit can legitimately change
file bytes, so perform the checksum check first.
