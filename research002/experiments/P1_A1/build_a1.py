"""Build readable findings, an exact scientific figure, and runnable evidence."""
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"deliverables"
LABELS={"current":"Current", "expanded_current":"Expanded current", "ordered_history":"Ordered history",
        "residual_history":"Residual history", "mismatched_history":"Mismatched history", "shuffled_history":"Shuffled history"}


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "|"+"|".join(["---"]*len(headers))+"|"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])


def main():
    result=json.loads((ROOT/"panel_run/result.json").read_text())
    audit=json.loads((ROOT/"panel_run/result_verification.json").read_text())
    assert audit["passed"]
    d=result["decision"]
    primary=[r for r in d["natural_cells"] if r["key"][0]==314159]
    assert len(primary)==4
    current_mean=np.mean([r["losses"][r["current_reference"]] for r in primary])
    history_mean=np.mean([r["losses"]["ordered_history"] for r in primary])
    primary_table=table(["Model seed","Step","Current Brier","History Brier","History advantage Δ","One-sided 95% upper bound","Current skill vs constant"],
        [[r["key"][1],r["key"][2],f"{r['losses'][r['current_reference']]:.5f}",
          f"{r['losses']['ordered_history']:.5f}",f"{r['history_advantage']['mean']:+.5f}",
          f"{r['history_advantage']['upper95']:+.5f}",f"{r['current_skill']:.1%}"] for r in primary])
    all_table=table(["Observer split","Model","Step",*LABELS.values()],
        [[*r["key"],*[f"{r['losses'][a]:.5f}" for a in LABELS]] for r in d["natural_cells"]])
    split_table=table(["Observer split","Pooled Δ","Two-sided 95% interval","One-sided 95% upper bound"],
        [[key,f"{value['history_advantage']['mean']:+.5f}",
          f"[{value['history_advantage']['ci95'][0]:+.5f}, {value['history_advantage']['ci95'][1]:+.5f}]",
          f"{value['history_advantage']['upper95']:+.5f}"] for key,value in d["by_observer_split"].items()])
    timing_rows=[]
    ratios=[]
    selected_rows=[]
    for row in primary:
        split,seed,step=row["key"]
        cell=json.loads((ROOT/f"panel_run/split_{split}_model_{seed}_step_{step}/result.json").read_text())
        ref=cell["current_reference"]
        times=cell["timing"]
        cm,hm,fm=[times[a]["median_seconds"]*1000 for a in [ref,"ordered_history","continuation"]]
        timing_rows.append([seed,step,f"{cm:.2f}",f"{hm:.2f}",f"{fm:.2f}",f"{cm/hm:.2f}×"])
        ratios.append(fm/cm)
        for arm in [ref,"ordered_history"]:
            chosen=cell["arms"][arm]["selected"]
            selected_rows.append([seed,step,LABELS[arm],chosen["kind"],chosen["penalty"],chosen["coefficient_count"]])
    timing_table=table(["Model","Step","Current ms","History ms","Full continuation ms","Current / history time"],timing_rows)
    selected_table=table(["Model","Step","Arm","Readout kind","Penalty","Fitted coefficients incl. intercept"],selected_rows)
    synthetic_table=table(["Synthetic seed","Ideal latent gain","Observed Δ","One-sided 95% lower bound"],
        [[r["synthetic_seed"],f"{r['ideal_latent_brier_gain']:.2f}",f"{r['history_advantage']['mean']:+.5f}",
          f"{r['history_advantage']['lower95']:+.5f}"] for r in d["synthetic_cells"]])
    gates=table(["Required condition","Outcome"],[[name.replace('_',' '),"Pass" if passed else "Fail"]
                                               for name,passed in d["sufficiency_checks"].items()])
    ci=d["primary"]["history_advantage"]["ci95"]
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.9),gridspec_kw={"width_ratios":[1.15,1]})
    y=np.arange(4)
    names=[f"Seed {r['key'][1]} · step {r['key'][2]}" for r in primary]
    for offset,arm,color,label in [(-0.17,"expanded_current","#186a65","Current with nonlinear features"),
                                  (0.17,"ordered_history","#6474a4","Current with ordered history")]:
        vals=[r["losses"][arm] for r in primary]
        axes[0].barh(y+offset,vals,height=.30,color=color,label=label)
        for i,v in enumerate(vals): axes[0].text(v+.0015,i+offset,f"{v:.4f}",va="center",fontsize=9)
    axes[0].set_yticks(y,names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0,.19)
    axes[0].set_xlabel("Brier loss · lower is better")
    axes[0].set_title("Prediction quality",loc="left",fontweight="bold")
    axes[0].legend(frameon=False,loc="lower left",bbox_to_anchor=(0,-.34),fontsize=9)
    means=np.array([r["history_advantage"]["mean"] for r in primary])
    upper=np.array([r["history_advantage"]["upper95"] for r in primary])
    axes[1].hlines(y,means,upper,color="#186a65",linewidth=2)
    axes[1].plot(means,y,"o",color="#186a65",markersize=6)
    axes[1].plot(upper,y,"|",color="#186a65",markersize=13)
    axes[1].axvline(0,color="#a9afb5",linewidth=1)
    axes[1].axvline(.01,color="#b76524",linestyle="--",linewidth=1.5)
    axes[1].text(.0094,-.43,"Tolerance 0.01",ha="right",color="#9b551e",fontsize=9)
    axes[1].set_yticks(y,["2017 · 3","2017 · 4","2027 · 3","2027 · 4"])
    axes[1].set_ylim(3.6,-.7)
    axes[1].set_xlim(-.026,.018)
    axes[1].set_xticks([-.02,-.01,0,.01])
    axes[1].set_xlabel("Δ: Brier(current) − Brier(history)")
    axes[1].set_title("Every upper bound is below tolerance",loc="left",fontweight="bold")
    axes[1].text(0,-.25,"Dots: mean effects\nRight whiskers: one-sided 95% upper bounds",transform=axes[1].transAxes,fontsize=9,va="top")
    for ax in axes:
        ax.grid(axis="x",alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("P1-A1 · Current-state sufficiency within a fixed prediction budget",x=.04,ha="left",fontweight="bold",fontsize=13)
    fig.text(.04,.025,"Development evidence: 193 operand families, two fixed models. Family bootstrap; no awareness measurement.",fontsize=9,color="#515b65")
    fig.subplots_adjust(left=.17,right=.97,top=.82,bottom=.27,wspace=.37)
    fig.savefig(OUT/"P1_A1_Results.png",dpi=200,facecolor="white")
    plt.close(fig)
    report=f"""# P1-A1 — Current-state and history prediction

Research 002 · 15 September 2026 · completed development protocol v1.1.0

**Result: the planned criteria for bounded practical current-state sufficiency passed.**
For these two trained arithmetic networks, at steps 3 and 4, the selected
current-state predictors were noninferior to the selected ordered-history
predictors within the declared **0.01 absolute Brier-loss tolerance**. This is a
specific development result about accessible prediction, not subjective awareness.

Mean Brier loss was **{current_mean:.5f} for current state** and **{history_mean:.5f}
for ordered history**. History advantage Δ = current loss minus history loss was
**{d['primary']['history_advantage']['mean']:+.5f}**, with conditional family-bootstrap
95% interval **[{ci[0]:+.5f}, {ci[1]:+.5f}]**. Negative values favor current state.
The conclusion uses the predetermined one-sided bounds and all-cell rules below,
rather than a failure to reject equality.

## What this establishes

The complete current state, combined with a modest learned predictor and fixed
nonlinear features of that same state, was enough to forecast whether the model's
current answer would change by its final computation step, to the stated
tolerance against this ordered-history comparator. Current prediction improved
Brier loss by **{min(r['current_skill'] for r in primary):.1%}–{max(r['current_skill'] for r in primary):.1%}**
relative to a fit-prevalence constant baseline across the four primary cells.

This does not imply that history is useless. Correctly paired history outperformed
mismatched history on average. It also had **lower inference time than the selected
current-state predictor when history was already cached**. The result supports
accuracy sufficiency at the tested budgets, not global computational optimality.

## Transparent amendment before observer fitting

The first v1.0 attempt stopped before any natural or synthetic-control observer
fit: some step-4 fit/tuning partitions had fewer than 10% decision changes.
Every partition nevertheless passed the separately specified event and family
counts. Version 1.1 removed the additional percentage cutoff and retained the
counts, all models, steps, splits, predictors, budgets, metrics and interpretation
thresholds. The smallest step-4 minority supports were 196 fit examples from
54 families and 37 tuning examples from 11 families.

This amendment was informed by observed support counts. It is preserved in
`AMENDMENT.md`, with the original stopped protocol, code and audit under
`stopped_v1/`. It must not be described as an untouched preregistration. No
prediction outcomes informed the amendment; no extra model or observer budget
was consumed by the stopped attempt beyond the separate method fixtures.

## Fixed experiment

- Arithmetic: `(a+b+c) mod 17`, the six-step weight-tied Transformer from P1-A0B.
  Final regularized model seeds 2017 and 2027; no arithmetic-network retraining.
- Target: current argmax differs from the final argmax, at steps 3 and 4. It is
  not mathematical correctness, conscious experience or metacognitive awareness.
- Current observation: all 96 token-state coordinates, 17 current probabilities,
  entropy and probability margin. History contains the same observations from
  steps 0 through t−1. Future values and targets never enter features.
- Fit/tuning: 465/116 disjoint training families, repeated with three fixed group
  splits. Evaluation: the same 193 families/950 examples from P1-A0B. These are
  exposed development conditions; the 195 reserve families remain unevaluated.
- Six arms, nine candidates each: linear ridge and two Gaussian landmark-feature
  ridge alternatives, each at three penalties. Tune on family-weighted Brier;
  choose the current reference between raw and dimension-matched expanded current.
- Exactly **648 natural** and **432 synthetic-control** candidate fits completed.
  All 120 selected pipelines, 1,080 candidate tuning scores and predictions are
  retained. No selection used the evaluation scores.

All four primary current references selected the expanded-current arm: fixed
cosine features of the present state, with no earlier state as input. Their
linear readouts and ordered-history readouts had matching coefficient counts.
This is a stronger current benchmark than testing a raw-state linear readout alone.
All primary readouts selected the weakest penalty in the fixed grid (0.0001).
The grid therefore does not exclude improvements from still weaker regularization
or other predictor families. No extra candidates were added after this observation.

{selected_table}

## Primary results and noninferiority checks

Brier loss is mean squared probability error, with each operand family weighted
equally and each permutation weighted equally within its family. The tolerance
0.01 is a development choice in Brier units, not a percentage point of accuracy.

{primary_table}

All four one-sided 95% upper bounds are below 0.01. The largest is
**{max(r['history_advantage']['upper95'] for r in primary):.5f}**, so the finding is
specific to this tolerance and does not establish arbitrarily close equivalence.
Every current predictor also passed the declared usefulness criterion: at least
10% relative Brier improvement and a positive lower bound against the constant.

{gates}

## Refitting the entire observer pipeline

The other two family splits refit scaling, residualization, nonlinear features,
candidate coefficients and tuning selection. Their pooled upper bounds also
remained below 0.01. These refits examine pipeline variation; they are not new
independent neural-network seeds.

{split_table}

The bootstrap resamples the same family IDs jointly across models, steps and
observer splits. Intervals are conditional on two fixed networks and exposed
development data; they do not quantify generalization to a population of trained
networks or correct for the research program's prior development choices.

## All six arms, all cells

Every entry is evaluation Brier, lower is better. No favorable cell or seed was
removed. The table is descriptive outside the frozen primary comparison.

{all_table}

The primary pooled mismatched-minus-ordered loss was
**{d['primary']['correspondence_advantage']['mean']:+.5f}**, with a positive
one-sided lower bound **{d['primary']['correspondence_advantage']['lower95']:+.5f}**.
Thus correspondence between present and past did matter to the history probes.
For seed 2027 at step 4, mismatched history had a slightly better point score;
the paired effect's interval included zero. That exception remains visible.

Residual history is a linear-residualization diagnostic, not proof of conditional
information beyond the present. Shuffling removes explicit slots but may leave
the computation stage inferable from state values. Neither control establishes a
general absence of useful historical structure.

## Sensitivity through the fitted pipeline

Synthetic controls use family-level current and history signals, noisy observed
features, and Bernoulli targets with known latent history effects. The same
preprocessing, candidate fitting and tuning selection used in the actual study
recovered **4/4 strong effects**, **4/4 near-margin effects**, and produced
**0/4 material false positives in null cases** under the frozen rules.

{synthetic_table}

These are twelve diagnostic datasets, not a power study. Success for this planted
signal does not guarantee detection of every nonlinear or distributed history
effect. It does provide a check that the actual fitted pipeline can register a
history contribution at approximately the declared tolerance in this setting.

## Computation and memory tradeoff

Median milliseconds for the same 950 evaluation examples, after three warm-ups
and over 15 timed repetitions:

{timing_table}

The selected current predictor was **{min(ratios):.1f}–{max(ratios):.1f} times faster
than full network continuation**, but slower than its ordered-history counterpart.
Feature transformations are included. Acquisition of existing state/readout and
recording of history are excluded. Therefore this timing comparison assumes
history is available already; it is not an end-to-end storage/latency benchmark.

The current observation has 115 coordinates. Ordered history adds 345 coordinates
at step 3 or 460 at step 4. Expanded current computes an equal number of nonlinear
features from current data, trading extra computation for avoiding those earlier
observations. Constant model parameters and stored random-feature matrices have
their own memory cost. No fastest or smallest implementation is established.

## Mathematical and philosophical boundary

With fixed weights and a known step count, the complete current state determines
the final state: `S_6 = F^(6−t)(S_t)`. Exact continuation is therefore a property
of this architecture. The empirical result adds that a much less expensive
predictor can extract a useful decision-change estimate from that state, with
the stated tolerance against a bounded history predictor.

The complete state may already carry effects of prior computation and of
arithmetic training. "Current-only" does not mean free of memory or prior causes.
The data support a limited statement: historical influence need not be supplied
again as a separate trajectory for this prediction task. They do not establish
present-moment consciousness, human mindfulness effects, or awareness in a
mathematical neural network. No untrained-versus-trained observer comparison was
performed, so training-induced changes in sufficiency remain untested.

## Verification and reproducibility

The audit reconstructed every selected evaluation and tuning prediction from
saved pipelines, checked all 1,080 tuning scores and selection rules, verified
selected ridge solutions by their first-order optimality equations, regenerated
all neural trajectories, and independently reconstructed family metrics,
bootstrap intervals and decision conditions. Maximum prediction discrepancy:
**{audit['max_prediction_error']:.3g}**; maximum ridge stationarity residual:
**{audit['selected_ridge_stationarity_max_residual']:.3g}**. Frozen inputs and the
stopped v1.0 evidence remained unchanged. This is a numerical audit, not an
independent replication.

One saved intermediate training-trajectory cache was truncated. The damaged
bytes are preserved, and pure forward computation from the frozen weights
reconstructed the cache. The entire damaged file matched the exact byte prefix
of the reconstruction. The audit then verified the reconstructed trajectory and
every saved predictor output; no observer fitting or network training was added.
The cause of truncation is not established. The recovery record and original
bytes are included under `panel_run/recovery/`.

Execution: Python {result['environment']['python'].split()[0]}, NumPy
{result['environment']['numpy']}, CPU. The amended panel took
**{result['elapsed_seconds']:.1f} seconds** in this runtime, excluding design,
artifact preparation and the separate numerical audit. Exact wall time and
floating-point behavior can differ elsewhere.

`P1_A1_Study.zip` includes the unchanged trained model inputs, both protocol
versions and amendment, execution and verification scripts, selected pipelines,
all tuning/evaluation predictions, control maps, trajectories and a SHA-256
content manifest. See `README.md` for verification and optional reproduction.
No Colab or external compute service is required.

## Next research boundary

The next stage should be a separately frozen confirmation design with new model
seeds and evaluation conditions that have not informed this development. Define
the accuracy/storage/latency tradeoff explicitly, since history was faster when
cached. Review the penalty-grid boundary using development conditions before
opening confirmation. Keep the current nonlinear benchmark, all model/step outcomes, and the
same Brier tolerance unless a new tolerance is justified prospectively. Confirm
adequate class support before designating confirmation data; do not tune to its
prediction outcomes. No confirmation evaluation has begun.

## Methodological context

Control tasks help distinguish representation access from what the probe learns;
this motivates our fitted synthetic diagnostics.
[Hewitt and Liang (2019)](https://aclanthology.org/D19-1275/).
Probe complexity and predictive performance should be considered together; this
motivates reporting coefficient counts and measured time.
[Pimentel et al. (2020)](https://arxiv.org/abs/2010.02180).
The data splits, margin and decision rules here are our development choices.
"""
    (OUT/"P1_A1_Report.md").write_text(report)
    (OUT/"P1_A1_Protocol.md").write_text((ROOT/"P1_A1_Protocol.md").read_text())
    (OUT/"P1_A1_Amendment.md").write_text((ROOT/"AMENDMENT.md").read_text())
    readme="""# P1-A1 v1.1 — runnable research evidence

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
"""
    (ROOT/"README.md").write_text(readme)
    selected=[]
    for p in ROOT.rglob('*'):
        if p.is_file() and "deliverables" not in p.parts and "__pycache__" not in p.parts and not p.name.startswith('.'):
            selected.append((p,str(p.relative_to(ROOT))))
    selected += [(OUT/name,name) for name in ["P1_A1_Report.md","P1_A1_Results.png"]]
    hashes={name:hashlib.sha256(p.read_bytes()).hexdigest() for p,name in selected}
    manifest=OUT/"CONTENTS_SHA256.json"
    manifest.write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(OUT/"P1_A1_Study.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p,name in sorted(selected,key=lambda item:item[1]): z.write(p,name)
        z.write(manifest,"CONTENTS_SHA256.json")
    with zipfile.ZipFile(OUT/"P1_A1_Study.zip") as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for name,expected_hash in hashes.items(): assert hashlib.sha256(z.read(name)).hexdigest()==expected_hash
    print(json.dumps({"files":{p.name:p.stat().st_size for p in OUT.iterdir()},
                      "archive_members_verified":len(hashes),"current_brier":current_mean,"history_brier":history_mean},indent=2))


if __name__=="__main__":
    main()
