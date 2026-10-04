# Verify T1-U7C without rerunning training

The experiment is complete and failed its all-seed development gate. This
workflow reads the saved outputs and runs inference from existing final
weights. It does not train a model, fit a probe, or instantiate reserved seeds
1103/1201/1301.

Unzip `T1_U7C_PORTABLE_BUNDLE.zip` and use the enclosed
`T1_U7C_Audit_Colab.ipynb` in Google Colab. Upload the same ZIP when prompted.
The notebook verifies every manifest entry before importing the audit code.

For an existing local Python environment, change into the extracted
`t1_transformer_trajectory` directory, install `requirements.txt` if necessary,
and run:

```bash
python audit_t1_u7c.py
```

The recorded environment used Python 3.12.13, NumPy 2.3.5, pandas 2.2.3,
SciPy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8, and PyYAML 6.0.3.
The NumPy Transformer has no PyTorch/TensorFlow dependency. The included
`threadpoolctl` dependency of scikit-learn limits BLAS concurrency during the
audit. No GPU is required.

Verification performs inference for six existing model seeds and reconstructs
their original evaluation sets. Allow several minutes on a CPU. It writes only
the recovery-audit record; it preserves original training outputs. A freshly
regenerated audit record changes its own timestamp, so use the ZIP manifest
to verify the distributed copy before running the audit.

The full-run entrypoint `run_t1_u7c.py` is retained as original source for
reproducibility. It is intentionally not executed by this notebook, since it
replays all training. Final `.npz` checkpoints contain weights only and cannot
resume an interrupted optimizer trajectory.

The original seed-paired T1-U7B/C comparisons for seeds 811/907/1009 use
different evaluation examples because their phase changed. Read the report's
comparison limitation before interpreting those small differences. The
recovery audit reproduces the original T1-U7C measurements; it does not replace
them with a newly chosen evaluation set.
