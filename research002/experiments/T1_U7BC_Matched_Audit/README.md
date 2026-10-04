# T1-U7BC-M1 matched-example audit

Read `T1_U7BC_Matched_Audit_Report.md` first. The audit completed: both model versions pass 3/6 exposed seeds on both original suites. No failing seed was rescued.

## Colab

Open the enclosed `T1_U7BC_Matched_Audit_Colab.ipynb` in Google Colab and upload the ZIP when prompted. The notebook uses CPU inference and preserves distributed results. It verifies all package files before execution.

## Local reproduction

Use Python 3.12, install `requirements.txt`, and run:

```bash
python run_matched_audit.py --out reproduced_results
```

The output directory must be new. Expect several minutes on CPU. The original runners inside `frozen/code` are source dependencies; the audit entrypoint is `run_matched_audit.py`.

## Contents

- `AUDIT_SPEC.json`: procedure frozen before cross-evaluation.
- `SOURCE_PROVENANCE.json`: original archive/checkpoint/trajectory hashes and compact-reference provenance.
- `FROZEN_INPUT_SHA256.json`: exact immutable audit input hashes.
- `frozen/`: original inference/generator code, 12 checkpoints, exact compact native references, configs, and native result tables.
- `results/`: all metrics, paired effects, per-example correctness, example fingerprints, and the completed verification record.
- `execution.log`: the single completed audit execution.
- `NEXT_STUDY_DESIGN.md`: source-grounded design memo; no probe protocol was executed.
- `PACKAGE_SHA256.json`: integrity hashes for distributed files, excluding this manifest itself.

The original evaluation sets are development-exposed. Duplicate suite labels for 503/607/709 are not independent replicates. All per-seed gates remain fixed; population-level repair or history-advantage claims are not supported by this audit.
