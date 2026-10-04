# P1-A0B handoff

Read P1_A0B_Report.md and P1_A0B_Protocol.md first. This is development readiness,
not a history-sufficiency result. All four model runs are included.

Verify the completed evidence without fitting:

    OPENBLAS_NUM_THREADS=1 python verify_a0b.py panel_run

Reproduce the exposed panel into a NEW folder (several CPU minutes):

    OPENBLAS_NUM_THREADS=1 python run_a0b.py panel --output reproduction_run

P1_A0B_Colab.ipynb embeds all required code/protocol and needs NumPy only.
Training is disabled by default. There is no history-probe fitting path.
Old data/weights remain unchanged; reserved examples are never evaluated.

CONTENTS_SHA256.json binds every file in this archive. FREEZE.json inside
panel_run binds the scientific sources prospectively before training.
