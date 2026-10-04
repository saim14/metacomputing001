# P1-A1 v1.1 — runnable research evidence

Read P1_A1_Report.md, P1_A1_Protocol.md and AMENDMENT.md first.
This is development evidence, not confirmation or a test of subjective awareness.

Requirements for the study and audit: Python 3.12 and NumPy (recorded run: 2.3.5).
Matplotlib is only required to rebuild the report's scientific figure.

Verify existing evidence without fitting:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_a1.py panel_run

Optionally reproduce the exposed development panel into a NEW directory:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_a1.py panel --output reproduction_run
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_a1.py reproduction_run

The second workflow fits exactly 648 natural and 432 synthetic-control candidates.
It does not train the underlying arithmetic networks or open reserved evaluation
families. Existing output directories are refused. No network access, GPU,
external model download, or Colab runtime is required.

Input model checkpoints and original NumPy Transformer source are under inputs/.
The initial v1.0 stopped attempt is preserved under stopped_v1/; its identical
inputs are stored once under inputs/ at the archive root. The v1.0 attempt had no
natural or synthetic-control observer fits. Its support audit informed v1.1.

panel_run/FREEZE.json binds prospective v1.1 scientific sources and inputs.
CONTENTS_SHA256.json binds all archive members except itself. Post-run report,
figure-building and verification scripts are separate from the frozen execution
source. Reproduction repeats exposed development conditions; it is not independent
confirmation and may differ slightly with numerical-library or hardware changes.
