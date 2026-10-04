"""Build human-readable report, self-contained Colab notebook and evidence ZIP."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "deliverables"


def main():
    result = json.loads((ROOT / "pilot_run/result.json").read_text())
    verification = json.loads((ROOT / "pilot_run/result_verification.json").read_text())
    assert verification["passed"]
    OUT.mkdir(exist_ok=True)
    rows = ["| Training updates | Training accuracy | Unseen-combination accuracy |",
            "|---:|---:|---:|"]
    for checkpoint in result["checkpoints"]:
        rows.append(f"| {checkpoint['update']:,} | {100*checkpoint['train']['accuracy_by_step'][-1]:.2f}% "
                    f"| {100*checkpoint['calibration']['accuracy_by_step'][-1]:.2f}% |")
    changes = ["| Computation step | Changed / total | Flip rate | Distinct families: changed / unchanged | Eligible |",
               "|---:|---:|---:|---:|---|"]
    for row in result["checkpoints"][-1]["calibration"]["decision_change"]:
        n = row["changed_examples"] + row["unchanged_examples"]
        changes.append(f"| {row['step']} | {row['changed_examples']} / {n} | {100*row['flip_prevalence']:.2f}% "
                       f"| {row['changed_families']} / {row['unchanged_families']} | {'Yes' if row['eligible'] else 'No'} |")
    report = f"""# P1-A0 — Arithmetic readiness pilot

Research 002 · 15 September 2026 · protocol v1.0.0  
**Outcome: competence gate failed; no history predictors fitted.**

The new present-state arithmetic branch has started. The implementation checks
passed, but this one fixed training recipe did not learn to answer unseen operand
combinations reliably. The prespecified stop was respected. The present-state
sufficiency and practice hypotheses remain **untested**, not refuted or supported.

## What actually ran

- Task: `(a+b+c) mod 17`, with inputs `[CLS, a, b, c]`.
- One newly initialized model, seed 2003; no prior models or reserved seeds used.
- Original T1 NumPy Transformer block reused for six steps; width 24, two heads,
  feed-forward width 48; {result['parameter_count']:,} trainable parameters.
- Exactly 1,200 Adam updates, batch size 64, learning rate 0.001, final-only loss.
- 2,992 training examples from 581 canonical families; 950 calibration examples
  from 193 different families. All operand permutations share a partition.
- 195 additional families reserved; no reserved examples or outputs evaluated.
- Local execution took {result['elapsed_seconds']:.1f} seconds, including scheduled
  evaluation and saving; Colab runtime may differ. No interactive Colab session
  was executed here. The delivered notebook is syntax-checked and uses the same
  locally executed code with NumPy only.

## Learning across the fixed checkpoints

{chr(10).join(rows)}

The final calibration result was 30 correct answers out of 950, versus the
declared 90% development competence target. Uniform random guessing among 17
classes has expected accuracy 5.88%; this is a mathematical reference, not a
fitted baseline or an independent significance test. Permutations are correlated.

The train/calibration gap is **consistent with fitting training combinations
without learning a reliably generalizing arithmetic rule**. This single short
run does not establish that the architecture cannot learn modular arithmetic,
that longer training would succeed, or that its failures arise from one specific
mechanism. No post-outcome extension or hyperparameter search was run.

## Decision-change target at update 1,200

{chr(10).join(changes)}

All four candidate steps had enough examples and distinct families in both
classes under the declared anti-degeneracy screen. A family can contribute to
both classes through different permutations; the two family counts are not
disjoint sample sizes. The target gate passed, but competence did not, so readiness
failed. Changes in an unreliable network are not evidence of useful self-monitoring.

## Mathematical and implementation checks

Every method check passed: exact labels, deterministic family assignment, split
separation, unique weight-tied parameters, gradients checked by finite differences,
probability normalization, checkpoint round trips, and observer interfaces that
do not use labels or future states.

At all four saved training checkpoints, resuming from any of steps 0..6 reproduced
the final logits for the 16 audited calibration examples with maximum absolute
difference **0.0** in this runtime. This checks exact computational continuation
from the complete current state. It costs the remaining Transformer computation
and does not show that an inexpensive current-only observer predicts equally well.
It is expected for this deterministic architecture, even before training.

Finite binary truth-table controls gave oracle Brier losses 0.25 for an incomplete
current view and 0.0 when the missing bit was revealed; a complete current view
already had oracle loss 0.0. These are mathematical fixtures, not fitted-probe
power estimates and not natural Transformer-history findings.

The separate verification script recomputed all calibration accuracy/flip metrics
from saved arrays, all checkpoint training accuracies from saved weights, source
identity against the original ZIP, all 1,200 update records, and the stopping
decision. The prospective local protocol/source freeze remained unchanged.

## Scientific boundary and next step

P1-A0 is complete with status `competence_failed`. No history advantage, practical
sufficiency, training-induced sufficiency, consciousness or causal metacognition
claim has been established. Earlier T1/T2 findings remain unchanged.

The next step is a separately versioned, bounded **arithmetic competence design**,
before observer fitting. It must justify any task/training change, record that
this development outcome informed the choice, and set a finite budget and fresh
evaluation plan before fitting. The old reserved seeds remain sealed. This report
does not automatically authorize additional runs or use of held-out families.

## Reproduction and files

`P1_A0_Colab.ipynb` is standalone: upload it into Colab. It contains the source,
protocol and recorded report, runs method checks, and defaults to **not** rerunning
training. Set `RUN_REPRODUCTION = True` only to reproduce the same exposed pilot;
it is not an independent replication and may vary slightly across numerical
library versions. No GPU or external model download is needed.

`P1_A0_Study.zip` contains source, original source archive, the exact protocol,
four trained checkpoints, calibration trajectories, development examples, optimizer
moments, logs, local freeze, verification and this report. Existing run directories
are never overwritten by the experiment runner.

### Sources and provenance

- [Original T1 source](https://drive.google.com/file/d/1116fOMjOUzIfyT-oYoI9gKv1bfoX63g9/view).
- [T2-E2 report](https://drive.google.com/file/d/1DyJAbpnvQXhIr_pqDwGJvQTU13VfypXf/view).
- [Observer design constraints](https://drive.google.com/file/d/1MwJIAc5stQ-6_987_ZPyBaa7L5M4ruLD/view).
- [Grokking study](https://arxiv.org/abs/2201.02177): background on algorithmic
  learning/generalization, not evidence for our new hypothesis.
"""
    (OUT / "P1_A0_Report.md").write_text(report)
    (OUT / "P1_A0_Protocol.md").write_text((ROOT / "P1_A0_Protocol.md").read_text())
    source_paths = ["experiment.py", "test_methods.py", "P1_A0_protocol.json", "P1_A0_Protocol.md",
                    "vendor/t1_original/src/__init__.py", "vendor/t1_original/src/tiny_autograd.py",
                    "vendor/t1_original/src/tiny_transformer.py"]
    sources = {p: (ROOT / p).read_text() for p in source_paths}
    source_hashes = {p: hashlib.sha256(v.encode()).hexdigest() for p, v in sources.items()}
    original_zip = base64.b64encode((ROOT / "vendor/T1_source_0.1.0.zip").read_bytes()).decode()
    cells = []
    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
    def code(text):
        ast.parse(text)
        cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(True),
                      "execution_count": None, "outputs": []})
    md("# P1-A0: Present-state sufficiency after arithmetic practice\n\n"
       "Self-contained NumPy notebook for Research 002. This is a **completed development-readiness pilot**, "
       "not a history hypothesis test. The original run failed its competence gate.\n\n"
       "Run cells in order. Default behavior checks the methods and shows the recorded report; "
       "training is explicitly opt-in below. Code ran locally; this notebook has not been executed in an interactive Colab session.")
    code("import os\nos.environ['OPENBLAS_NUM_THREADS'] = '1'\nos.environ['OMP_NUM_THREADS'] = '1'\n"
         "import base64, hashlib, json, subprocess, sys, tempfile, time, zipfile\nfrom pathlib import Path\n"
         "from IPython.display import Markdown, display\n"
         "ROOT = Path(tempfile.mkdtemp(prefix='p1_a0_'))\n"
         f"SOURCES = {sources!r}\nSOURCE_HASHES = {source_hashes!r}\n"
         "for relative, content in SOURCES.items():\n"
         "    assert hashlib.sha256(content.encode()).hexdigest() == SOURCE_HASHES[relative]\n"
         "    destination = ROOT / relative\n    destination.parent.mkdir(parents=True, exist_ok=True)\n"
         "    destination.write_text(content)\n"
         f"(ROOT / 'vendor/T1_source_0.1.0.zip').write_bytes(base64.b64decode({original_zip!r}))\n"
         f"RECORDED_REPORT = {report!r}\n(ROOT / 'P1_A0_Report.md').write_text(RECORDED_REPORT)\n"
         "print('Experiment prepared at', ROOT)\nprint('Only NumPy is required; no GPU or downloads.')")
    md("## 1. Run the implementation checks\n\nThese checks use untrained fixture models and a synthetic scalar optimizer test. They do not train a natural arithmetic model or fit history observers.")
    code("methods_output = ROOT / ('methods_' + str(time.time_ns()))\n"
         "subprocess.run([sys.executable, str(ROOT / 'experiment.py'), 'methods', '--output', str(methods_output)], check=True)\n"
         "checks = json.loads((methods_output / 'method_checks.json').read_text())\n"
         "print('Method checks passed:', checks['passed'])")
    md("## 2. Read the recorded result\n\nThe following is the saved local result, not a new Colab run.")
    code("display(Markdown(RECORDED_REPORT))")
    md("## 3. Optional: reproduce the same fixed pilot\n\nSet the flag to `True` to run the same 1,200-update recipe. "
       "This repeats an exposed development experiment, not confirmation. It will not extend training or launch observer fitting. "
       "Expect CPU/runtime-dependent execution time; the original local run took about 25 seconds.")
    code("RUN_REPRODUCTION = False\n"
         "if RUN_REPRODUCTION:\n"
         "    pilot_output = ROOT / ('reproduction_' + str(time.time_ns()))\n"
         "    subprocess.run([sys.executable, str(ROOT / 'experiment.py'), 'pilot', '--output', str(pilot_output)], check=True)\n"
         "    rerun = json.loads((pilot_output / 'result.json').read_text())\n"
         "    print('Reproduction status:', rerun['status'])\n"
         "    print('Final calibration accuracy:', rerun['final_calibration_accuracy'])\n"
         "    print('History observers fitted:', rerun['history_probes_fitted'])\n"
         "else:\n    print('No new training run requested; recorded result remains unchanged.')")
    md("## 4. Download this notebook session's files\n\nThe full original numerical archive is the separately delivered `P1_A0_Study.zip`. "
       "This cell downloads the sources, method checks and any reproduction outputs created in this session.")
    code("archive_path = ROOT.parent / (ROOT.name + '_outputs.zip')\n"
         "with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:\n"
         "    for path in sorted(ROOT.rglob('*')):\n"
         "        if path.is_file() and '__pycache__' not in path.parts:\n"
         "            archive.write(path, path.relative_to(ROOT))\n"
         "try:\n    from google.colab import files\n    files.download(str(archive_path))\n"
         "except ImportError:\n    print('Session archive:', archive_path)")
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": "P1_A0_Colab.ipynb"}}, "nbformat": 4, "nbformat_minor": 4}
    (OUT / "P1_A0_Colab.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    readme = """# P1-A0 handoff

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
"""
    (ROOT / "README.md").write_text(readme)
    selected = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or "deliverables" in path.parts:
            continue
        if path.name.startswith("."):
            continue
        selected.append((path, str(path.relative_to(ROOT))))
    selected += [(OUT / name, name) for name in ("P1_A0_Report.md", "P1_A0_Colab.ipynb")]
    hashes = {relative: hashlib.sha256(path.read_bytes()).hexdigest() for path, relative in selected}
    (OUT / "CONTENTS_SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    with zipfile.ZipFile(OUT / "P1_A0_Study.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path, relative in sorted(selected, key=lambda item: item[1]):
            archive.write(path, relative)
        archive.write(OUT / "CONTENTS_SHA256.json", "CONTENTS_SHA256.json")
    with zipfile.ZipFile(OUT / "P1_A0_Study.zip") as archive:
        assert archive.testzip() is None
        for name, expected in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    print(json.dumps({"notebook_code_cells_syntax_checked": sum(c['cell_type']=='code' for c in cells),
                      "archive_members_verified": len(hashes),
                      "outputs": {p.name: p.stat().st_size for p in OUT.iterdir()}}, indent=2))


if __name__ == "__main__":
    main()
