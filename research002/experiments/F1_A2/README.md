# F1-A2 reproducible study

Run, audited and reported on 18 September 2026. Start with
`F1_A2_Report.md` and the prospective `F1_A2_Protocol.md`.

Use Python 3.12.14 and the recorded dependency versions in `requirements.txt`.
The release runs offline once its dependencies are installed.

## Verify the saved results without training

Immediately after extraction, check bytes with:

```bash
sha256sum --check SHA256SUMS.txt
```

Then reconstruct the scientific results:

```bash
python audit.py
```

This replays 1,260 candidate tuning scores and 105 selected predictions from
saved coefficients, independently scores true outcomes and checks prefix-only
features, split isolation, reference selection, intervals and frozen hashes.
It performs no additional observer fitting or neural training. It rewrites
`run/audit.json` with a new elapsed time, so run the checksum check first.

```bash
python make_report.py
```

Regenerates the figure and report from the existing audited results.

## Fresh reproduction

Create a separate empty directory. Copy `study.py`, `config.json`,
`F1_A2_Protocol.md`, and the **entire `prior/` directory** there, preserving its
nested paths. Do not copy the top-level `run/` directory. Then:

```bash
python study.py
```

The script refuses to overwrite an existing run. It executes the fixed 1,260
observer candidates and local regression algorithm, without neural retraining.
Timestamps, runtime and minor floating-point differences depend on the platform;
this is a reproduction, not new independent evidence. Copy `audit.py` into the
fresh directory to verify its result.

## Main files

| File or directory | Contents |
| --- | --- |
| `study.py`, `config.json`, `F1_A2_Protocol.md` | Frozen experiment and design |
| `prior/` | Exact F1-A1 source/config/protocol/report and nested F1-A0 checkpoint dependencies |
| `run/manifest.json` | Source/checkpoint hashes and versions recorded before outcomes |
| `run/scenarios.npz` | Fresh scenes and fit/tuning/evaluation IDs |
| `run/calibration_*.npz` | Independent observations and signed residuals |
| `run/diagnostic_*.npz` | Disjoint rollout diagnostic inputs and predictions |
| `run/data_*.npz` | Prefix features, current candidates/costs, U3/U6 plans and true signed benefits |
| `run/observer_*.npz` | All fitted coefficients, normalizers and selected indices |
| `run/controls_*.npz`, `run/expanders_*.npz` | Derangements and fit-only expansion statistics |
| `run/evaluation.npz` | Selected tuning/evaluation predictions and targets |
| `run/candidate_grid.json` | All 1,260 tuning scores |
| `run/result.json` | Complete primary/secondary and per-condition results |
| `run/runtime.json` | Individual runtime repetitions and online replay checks |
| `audit.py`, `run/audit.json` | Reconstruction code and recorded audit |
| `make_report.py` | Post-run presentation, not experimental model selection |
| `F1_A2_Report.md`, `F1_A2_Results.png` | Human-readable findings |
| `package_study.py`, `SHA256SUMS.txt` | Packaging and byte integrity |

P is a privileged simulator diagnostic; B uses only model plus calibration
residuals for its estimate. The underlying exact/half-error model conditions are
laboratory controls requiring true dynamics. Runtime is descriptive. No sealed
T1 seeds, P1 reserve families, new neural fits, F1-A3 study or website deployment
are included. Temporary writes, bytecode and the ZIP itself are excluded.
