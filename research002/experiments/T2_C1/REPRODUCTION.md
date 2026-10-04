# Reproduce T2-C1

Use Python 3.12 or a compatible newer runtime. The original run used Python 3.12.14; package versions are pinned in `requirements.txt`. Small numeric differences across BLAS implementations may occur. Run from the extracted `T2_C1_Sensitivity` directory.

```bash
python -m pip install -r requirements.txt
python verify_methods.py
python verify_results.py
python run_calibration.py --out reproduced_results
python verify_results.py --results reproduced_results
```

`run_calibration.py` refuses an existing output directory. The distributed results remain available for comparison. No Google Drive connection, Transformer checkpoint, GPU, training service, or private data is needed. The 43.51-second original runtime is environment-specific; Colab timing may differ.

`make_figure.py` regenerates the figure from the distributed results. `PACKAGE_SHA256.json` verifies all distributed file bytes except itself; `FROZEN_INPUT_SHA256.json` separately verifies the protocol and two original T2-E1 reference files. The reference runner is used only to extract its two ridge functions for numerical verification; it is not executed as a neural study.

The bundled notebook automates archive validation, installation, verification, a fresh reproduction run, and a comparison of the resulting scores and decisions. Raw X/H/target arrays and probe weights are not archived; deterministic generator seeds and saved hashes support regeneration. All dataset scores, selected α values, panel decisions, and conditional bootstrap panel draws are archived.
