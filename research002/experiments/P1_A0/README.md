# P1-A0 handoff

Start with P1_A0_Report.md. The pilot failed competence; no history probes ran.
Upload P1_A0_Colab.ipynb into Colab for standalone method checks or an explicit
reproduction of the exposed pilot. No GPU and no dependencies beyond NumPy.

To verify the original saved data after extracting this archive:

    OPENBLAS_NUM_THREADS=1 python verify_results.py pilot_run

To reproduce into a NEW path, never replacing original results:

    OPENBLAS_NUM_THREADS=1 python experiment.py pilot --output reproduction_run

Reproductions are not independent confirmation. The original source archive and
unchanged vendor implementation are included; no old trained models were loaded.
Read the protocol's next-stage boundary before any additional scientific fitting.
