"""Create the report, standalone Colab notebook and audited evidence archive."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "deliverables"


def main():
    result = json.loads((ROOT / "panel_run/result.json").read_text())
    audit = json.loads((ROOT / "panel_run/result_verification.json").read_text())
    assert audit["passed"]
    decision = result["decision"]
    selected = decision["selected_development_candidate"]
    status = (f"Development candidate: {selected}; observer fitting still unopened."
              if selected else "No regimen passed the joint readiness gate; no history observers fitted.")
    table = ["| Seed | Regimen | Final training accuracy | Final calibration accuracy | Eligible steps | Competence >=90% |",
             "|---:|---|---:|---:|---|---|"]
    curve = ["| Seed | Regimen | Update 0 | Update 1,200 | Update 3,000 | Update 6,000 |",
             "|---:|---|---:|---:|---:|---:|"]
    regime_table = ["| Regimen | Both seeds competent? | Common eligible steps | Ready for observer design? |",
                    "|---|---|---|---|"]
    yes = lambda v: "Yes" if v else "No"
    for r in result["runs"]:
        table.append(f"| {r['seed']} | {r['arm']} | {r['final_train_accuracy']:.2%} "
                     f"| {r['final_calibration_accuracy']:.2%} | {', '.join(map(str,r['eligible_steps'])) or 'None'} "
                     f"| {yes(r['competence_passed'])} |")
        values = " | ".join(f"{c['calibration']['accuracy_by_step'][-1]:.2%}" for c in r["checkpoints"])
        curve.append(f"| {r['seed']} | {r['arm']} | {values} |")
    for r in decision["regimens"]:
        regime_table.append(f"| {r['arm']} | {yes(r['competence_passed_all_seeds'])} "
                            f"| {', '.join(map(str,r['common_eligible_steps'])) or 'None'} "
                            f"| {yes(r['ready_for_observer_design'])} |")
    paired = "; ".join(f"seed {r['seed']}: {100*r['decay_minus_no_decay_accuracy']:+.2f} percentage points"
                       for r in decision["paired_effects"])
    max_resume = max(e for r in result["runs"] for c in r["checkpoints"] for e in c["resume_errors"].values())
    next_step = (f"The prospectively frozen selection policy identifies `{selected}` as a development candidate. "
                 "This supports designing the next observer protocol, but not automatically running probes "
                 "or interpreting present-state sufficiency. Both seeds and every earlier checkpoint remain "
                 "in the evidence record; a later confirmation study needs untouched evaluation conditions."
                 if selected else
                 "This fixed training-budget/regularization screen is closed without a ready regimen. "
                 "Do not keep extending training or choose only a favorable seed. The next step is a "
                 "separately specified review of arithmetic task/representation and learnability before "
                 "another competence recipe. Any proposed change must acknowledge that these development "
                 "outcomes informed it. This is a next-design recommendation, not a new experiment or diagnosis.")
    report = f"""# P1-A0B — Arithmetic competence revision

Research 002 · 15 September 2026 · protocol v1.0.0  
**{status}**

## What was completed

Four fixed runs tested 6,000 updates with and without matrix-only decoupled weight
decay, for new model seeds 2017 and 2027. All 24,000 updates were completed; no
extra seeds, extensions or best-checkpoint selection were added. Initial weights
and the entire training batch stream were identical within each seed pair.

The task, split and model were unchanged from P1-A0: `(a+b+c) mod 17`, a six-step
weight-tied Transformer, width 24, two heads, feed-forward width 48 and 5,873 trainable
parameters. Both arms used final-only cross entropy, batch size 64, learning rate
0.001, Adam beta1=0.9/beta2=0.999 and gradient clipping at 1.0. Decay was either
zero or 1 on matrices only; biases and LayerNorm vectors were excluded. At learning
rate 0.001, decay 1 multiplies each matrix by 0.999 before the adaptive update.

P1-A0's observed train/calibration gap motivated this development revision.
Longer training and weight decay have precedent in small algorithmic studies,
but that literature uses different architectures and much larger budgets; it
does not guarantee success in our setting. [Grokking](https://arxiv.org/abs/2201.02177),
[decoupled weight decay](https://arxiv.org/abs/1711.05101).

## Final results at the fixed 6,000-update endpoint

{chr(10).join(table)}

The paired accuracy differences, decay minus no-decay, were {paired}.
The arithmetic mean was **{100*decision['mean_paired_accuracy_difference']:+.2f} percentage points**.
These are descriptive differences for two seeds on one exposed development split,
not population estimates or independently confirmed effects. Matrix-decay versus
no-decay is the controlled within-seed contrast; comparisons with P1-A0's seed 2003
are not controlled optimizer comparisons.

## Calibration learning curves, without best-checkpoint selection

{chr(10).join(curve)}

Only update 6,000 determined readiness. Training and calibration accuracies for
every computation step and checkpoint are saved in the numeric outputs. A high
training score cannot substitute for performance on unseen operand combinations.
Uniform random guessing has expected accuracy 1/17=5.88%; this is a mathematical
reference, not a fitted baseline or a significance test.

## Joint readiness decision

{chr(10).join(regime_table)}

Competence requires final calibration accuracy >=90% in **both** seeds. Target
readiness requires at least two common steps among 2..5, each with >=50 examples
and >=25 distinct families in both decision-change classes, and each class at
least 10% of the examples. These are development screening rules, not an observer
power analysis. The decision target is disagreement of current and final argmax,
not mathematical correctness, controller utility, or consciousness.

The selection rule prefers no-decay only if both regimens pass. No single model
or favorable checkpoint can override the all-seed gate. Current status:
`{decision['status']}`.

## Data and interpretation limits

- The models trained on 2,992 examples from 581 canonical operand families.
- Calibration used 950 examples from 193 disjoint families; every permutation of
  an operand triple remains in the same partition.
- These calibration examples were already exposed in P1-A0 and influenced this
  research direction. They remain unseen to gradient-based training, but are
  **development data**, not fresh confirmation data.
- The 195 reserved families were not expanded into examples or evaluated. Old
  reserved model seeds 1103/1201/1301 and all closed T1/T2 weights were untouched.
- A family can contribute to both decision-change classes through different
  permutations. Permutations and computation steps are not independent models.
- This tests only same-modulus, same-length operand combinations. No claim of
  universal arithmetic ability or length extrapolation is established.

No history predictors were fitted. Therefore neither practical present-state
sufficiency nor a training-induced reduction in history advantage was tested.
Exact continuation from a complete state is an architectural property, not a
finding of subjective awareness or inexpensive predictability.

## Verification and provenance

Inherited method checks passed, together with exact zero-decay equivalence to the
original Adam, synthetic closed-form decay/moment updates, the matrix/vector mask,
pairing of initializations and batches, and the all-seed selection logic. The
original Transformer, autodiff and P1-A0 implementation were preserved unchanged.

The separate verification script recomputed all training and calibration logits
for all 16 saved checkpoints from the stored weights and inputs. Maximum absolute
logit difference was **{audit['max_logit_error']:.8g}**. It also rechecked every accuracy,
flip count, family count, gate, paired batch streams and initial weights, and the
final decision. All 24,000 log entries were checked for consecutive update indices
and finite loss/gradient norms; saved optimizer moments were checked for shape,
finiteness and step count. Training updates were not replayed. This is a numerical
audit, not independent model replication.

At every checkpoint, resumption from each full current state was audited on 16
calibration examples at steps 0..6; the maximum recorded discrepancy was
**{max_resume:.8g}**. This resumption pays for the remaining Transformer steps and
does not demonstrate that a cheaper predictor can match a history-aware one.

Scientific sources and protocol were locally frozen before natural training and
remained unchanged. The source-hash record is an audit trail, not an independently
timestamped external registration. The verification script is post-run analysis
code and is identified separately from frozen scientific execution sources.

Local panel execution took **{result['elapsed_seconds']/60:.1f} minutes**. Environment:
Python {result['environment']['python'].split()[0]}, NumPy {result['environment']['numpy']}.
No GPU was used. Colab runtime can differ. An interactive Colab session was not
executed here; notebook code cells were syntax-checked and embed the same locally
executed source.

## Next boundary

{next_step}

Before any observer fit, freeze the predictor families and budgets, strong
current-only baselines, matched/mismatched/permuted/residualized history controls,
train-only preprocessing and eligibility, a useful-prediction criterion and
noninferiority margin, fitted-pipeline sensitivity, family/seed-aware uncertainty,
and untouched confirmation rules. The prior 5% MSE threshold is not automatically
a valid Brier-score margin or deployment threshold. No automatic advancement
was triggered by this panel.

## Files and reproduction

- `P1_A0B_Colab.ipynb`: standalone NumPy notebook; method checks and this recorded
  report by default. Full reproduction requires explicitly enabling its flag.
- `P1_A0B_Protocol.md`: readable prospective specification.
- `P1_A0B_Study.zip`: exact protocol/source, original-code provenance, all model
  checkpoints, optimizer moments, paired batches, predictions, logs and audit.

Reproducing the same four exposed runs is not independent confirmation. Existing
run directories are never overwritten. Every scientific change requires a new
version; do not edit the completed protocol to relabel an observed outcome.
"""
    OUT.mkdir(exist_ok=True)
    (OUT / "P1_A0B_Report.md").write_text(report)
    (OUT / "P1_A0B_Protocol.md").write_text((ROOT / "P1_A0B_Protocol.md").read_text())
    names = ["run_a0b.py", "checks_a0b.py", "verify_a0b.py", "P1_A0B_protocol.json", "P1_A0B_Protocol.md"]
    names += [str(p.relative_to(ROOT)) for p in (ROOT/"prior").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".zip"]
    sources = {name:(ROOT/name).read_text() for name in names}
    source_hashes = {name:hashlib.sha256(value.encode()).hexdigest() for name,value in sources.items()}
    original = base64.b64encode((ROOT/"prior/vendor/T1_source_0.1.0.zip").read_bytes()).decode()
    cells=[]
    def md(text):
        cells.append({"cell_type":"markdown","metadata":{},"source":text.splitlines(True)})
    def code(text):
        ast.parse(text)
        cells.append({"cell_type":"code","metadata":{},"source":text.splitlines(True),"execution_count":None,"outputs":[]})
    md("# P1-A0B: arithmetic competence revision\n\n"
       "Research 002. Four-run, paired NumPy development screen. This is **not a history-sufficiency test**. "
       "Run cells in order: by default they prepare sources, test methods and show the recorded report. "
       "Full 24,000-update reproduction is explicitly opt-in. Code was executed locally; the notebook was not run in an interactive Colab session.")
    code("import os\nos.environ['OPENBLAS_NUM_THREADS']='1'\nos.environ['OMP_NUM_THREADS']='1'\n"
         "import ast, base64, hashlib, json, subprocess, sys, tempfile, time, zipfile\n"
         "from pathlib import Path\nfrom IPython.display import Markdown, display\n"
         "ROOT=Path(tempfile.mkdtemp(prefix='p1_a0b_'))\n"
         f"SOURCES={sources!r}\nSOURCE_HASHES={source_hashes!r}\n"
         "for name,content in SOURCES.items():\n"
         "    assert hashlib.sha256(content.encode()).hexdigest()==SOURCE_HASHES[name]\n"
         "    path=ROOT/name\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_text(content)\n"
         f"(ROOT/'prior/vendor/T1_source_0.1.0.zip').write_bytes(base64.b64decode({original!r}))\n"
         f"RECORDED_REPORT={report!r}\n(ROOT/'P1_A0B_Report.md').write_text(RECORDED_REPORT)\n"
         "print('Prepared:',ROOT)\nprint('NumPy only; no model downloads or GPU required.')")
    md("## 1. Run method checks\n\nUses synthetic optimizer fixtures and untrained networks. Does not train the arithmetic panel or fit history predictors.")
    code("method_output=ROOT/('methods_'+str(time.time_ns()))\n"
         "subprocess.run([sys.executable,str(ROOT/'run_a0b.py'),'methods','--output',str(method_output)],check=True)\n"
         "print('Methods passed:',json.loads((method_output/'method_checks.json').read_text())['passed'])")
    md("## 2. Recorded local results\n\nThese are the completed run's saved results, not a new Colab result.")
    code("display(Markdown(RECORDED_REPORT))")
    md("## 3. Optional full reproduction\n\n"
       "Enabling this flag runs four models × 6,000 updates using the frozen recipe, then verifies the results. "
       "Expect several CPU minutes; hardware affects duration. No extra runs or history probes will launch. "
       "This repeats exposed development work, not independent confirmation.")
    code("RUN_REPRODUCTION=False\n"
         "if RUN_REPRODUCTION:\n"
         "    output=ROOT/('reproduction_'+str(time.time_ns()))\n"
         "    subprocess.run([sys.executable,str(ROOT/'run_a0b.py'),'panel','--output',str(output)],check=True)\n"
         "    subprocess.run([sys.executable,str(ROOT/'verify_a0b.py'),str(output)],check=True)\n"
         "    rerun=json.loads((output/'result.json').read_text())\n"
         "    print(json.dumps(rerun['decision'],indent=2))\n"
         "else:\n    print('No new training requested.')")
    md("## 4. Download this session's files\n\n"
       "The separately delivered P1_A0B_Study.zip holds all original numerical evidence. "
       "This cell saves the sources, checks and any reproduction results created in this session.")
    code("archive_path=ROOT.parent/(ROOT.name+'_outputs.zip')\n"
         "with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:\n"
         "    for path in sorted(ROOT.rglob('*')):\n"
         "        if path.is_file() and '__pycache__' not in path.parts:\n"
         "            archive.write(path,path.relative_to(ROOT))\n"
         "try:\n    from google.colab import files\n    files.download(str(archive_path))\n"
         "except ImportError:\n    print('Outputs:',archive_path)")
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.12"},"colab":{"name":"P1_A0B_Colab.ipynb"}},
              "nbformat":4,"nbformat_minor":4}
    (OUT/"P1_A0B_Colab.ipynb").write_text(json.dumps(notebook,indent=1)+"\n")
    readme="""# P1-A0B handoff

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
"""
    (ROOT/"README.md").write_text(readme)
    selected_files=[]
    for path in ROOT.rglob("*"):
        if path.is_file() and "deliverables" not in path.parts and "__pycache__" not in path.parts and not path.name.startswith("."):
            selected_files.append((path,str(path.relative_to(ROOT))))
    selected_files += [(OUT/name,name) for name in ("P1_A0B_Report.md","P1_A0B_Colab.ipynb")]
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for path,name in selected_files}
    (OUT/"CONTENTS_SHA256.json").write_text(json.dumps(hashes,indent=2)+"\n")
    with zipfile.ZipFile(OUT/"P1_A0B_Study.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for path,name in sorted(selected_files,key=lambda item:item[1]):
            archive.write(path,name)
        archive.write(OUT/"CONTENTS_SHA256.json","CONTENTS_SHA256.json")
    with zipfile.ZipFile(OUT/"P1_A0B_Study.zip") as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for name,expected in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    print(json.dumps({"notebook_code_cells_syntax_checked":sum(c['cell_type']=='code' for c in cells),
                      "archive_files_verified":len(hashes),
                      "outputs":{p.name:p.stat().st_size for p in OUT.iterdir()}},indent=2))


if __name__ == "__main__":
    main()
