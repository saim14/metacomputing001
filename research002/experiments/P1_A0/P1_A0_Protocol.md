# P1-A0 — Present-state sufficiency after arithmetic practice

Research 002 · protocol v1.0.0 · 15 September 2026  
Status: a new, bounded development pilot, not a confirmatory history study.

## Purpose and boundary

Begin the approved arithmetic branch while retaining all negative T1/T2 findings.
T2-E2 already used all-token current observations; repeating that comparison on its
exposed panel is not authorized here. Its remaining true-label-loss target is
distinct from this branch's proposed decision-change target. Original reserved
seeds 1103, 1201 and 1301 are not used. No previous trained weights are loaded.

The new hypothesis concerns **usable prediction under a finite observer budget**.
It does not operationalize subjective awareness, and no arithmetic accuracy or
probe result can by itself establish consciousness, causal metacognition, or a
benefit from an early-exit controller.

## Mathematical result before experimentation

For fixed weights theta, a fixed total step count L, no stochasticity, no incoming
new inputs, and complete all-token residual state S_l at a block boundary:

`S_(l+1) = F_theta(S_l)` and `logits_L = readout(F_theta^(L-l)(S_l))`.

Induction gives exact functional dependence on the current state and step. Thus
earlier activation records are unnecessary to **resume the actual computation**.
This holds before training as well as after it. Our numerical resume check is an
implementation audit of that property, not a discovery or a test of cheap
predictability. Resumption pays the cost of all remaining Transformer steps.

The complete state can encode the past. It is not the latest input token alone,
and it does not mean removing learned weights or previous operands. For this
encoder wrapper there are no KV caches, dropout, external inputs, or recurrent
variables outside the saved residual tensor; the weights and step remain fixed.
The statement determines the model's future output, not whether that output is
mathematically correct. True-label losses also require the true label.

The nontrivial future experiment asks whether an **affordable** current-only
observer predicts future decision change accurately, with history's possible
increment smaller than a justified, prospectively fixed margin. Equal performance
from two poor observers and a nonsignificant difference are insufficient evidence.
The further practice hypothesis asks whether that increment declines across
predeclared training checkpoints. This first stage does not fit those observers.

## Why this task and implementation

Use `(a+b+c) mod 17` with three operands in 0..16. Labels are exact and inexpensive;
there are 4,913 ordered triples and 969 canonical sorted-triple families. All
permutations of a triple stay in one partition. A randomly assigned 60% family
training set is separated from a 20% development calibration set and the remaining
reserved families. The reserved families are assigned IDs but not expanded into
examples or evaluated in this stage. This tests unseen operand combinations,
not new lengths, new moduli, or universal arithmetic reasoning.

The implementation reuses the original T1 NumPy autodiff and pre-norm Transformer
block unchanged, wrapped as one block repeated six times. It has width 24, two
heads, feed-forward width 48 and a shared 17-class CLS readout. The readout is
trained at the final step only; intermediate readout alignment is not assumed.
This is a new arithmetic regime, not a claim of numerical identity with U7C.

One development model, seed 2003, receives exactly 1,200 Adam updates at learning
rate 0.001, batch size 64, global gradient clip 1.0, no weight decay and no
intermediate supervision. Checkpoints are 0, 100, 400 and 1,200 updates. No best
checkpoint is selected. The finite budget limits this turn's exploration; failure
to learn within it is not a theorem about the task or architecture.

The canonical machine-readable specification is `P1_A0_protocol.json`. A local
`FREEZE.json` binds the protocol, source and original source archive before the
pilot's first training update. Hashes provide a local audit trail, not an
independently timestamped external preregistration. Reproduction is explicitly a
rerun of the same exposed development study, never a fresh independent replicate.

## Readiness measurements and fixed stopping rules

At every checkpoint, report training and calibration accuracy for steps 0..6.
At candidate steps 2..5, record
`Z_l = 1[current argmax differs from final argmax]`, which measures change rather
than correctness. Report both event classes and the number of distinct families
contributing to each class. Count families, not permutations, when interpreting
independent support. These are descriptive development measurements, not p-values.

Final readiness requires:

- All method tests and numeric checks pass.
- Update-1,200 final calibration accuracy is at least 90%: a declared development
  target, not a validated operational threshold.
- At least two of steps 2..5 have at least 50 examples **and** 25 families in each
  flip class, with each class at least 10% of examples. These anti-degeneracy rules
  are not a power analysis.

Stop after the fixed schedule even if competence fails. A competence failure
means this training recipe is not ready; a target failure means this target is
not ready. Neither supports nor refutes present-state sufficiency. A pass permits
the next **design stage**, not automatic probes or confirmation. Numeric failure
stops the run, preserves diagnostics and invalidates scientific interpretation.

## Mandatory checks

Check exact arithmetic labels; canonical-family disjointness and full coverage;
reserved examples not instantiated; unique tied parameters; finite-difference
gradients through the shared block; probability normalization; input and output
shapes; serialization round trip; and future-logit reproduction from each saved
current state using the same weights and remaining step count.

Check the future observer extraction interface uses only steps at or before l,
keeps the current prefix identical, and is insensitive to changes to future
arrays. Two finite binary truth-table fixtures distinguish full observation
(zero oracle Brier loss) from partial observation (oracle Brier loss 0.25, reduced
to zero by revealing the missing bit). These are mathematical fixtures, not
empirical Transformer evidence and not sensitivity validation of any fitted probe.

## Boundary before P1-A1 observer fitting

P1-A0 outputs candidate observables, not an observer implementation. Before fitting
anything, prospectively freeze a separate protocol specifying:

- The finite development/replication budget and fresh training seeds; do not count
  multiple training checkpoints or steps as independent model replicates.
- Observer splits keeping every canonical family and all its checkpoints/steps
  together. No base-model selection on observer test outcomes.
- CLS-only current (43-D) and all-token current (115-D), each compared with the
  same current prefix plus past CLS/prediction summaries from steps 1..l-1.
- Linear and a stronger nonlinear current-only reference; matched feature-width,
  parameter, optimization, tuning and inference resource accounting.
- Sample-mismatched and train-only-residualized history controls; per-example
  permutation of past slots with the current observation fixed for chronology.
  At least two past slots are required; a fixed column reversal is insufficient.
- Primary proper score (proposed Brier), an absolute useful-prediction criterion,
  an independently justified equivalence/noninferiority margin, train-only
  eligibility rules, multiplicity/aggregation and family-aware uncertainty.
- Null and known-positive calibration through the **actual fitted pipeline**,
  with enough sensitivity around the chosen margin. The exact oracle fixtures
  below cannot substitute for this check.
- Prediction-cost justification: the original T2-E2 5% relative-MSE continuation
  threshold is not inherited as a validated Brier-score or deployment margin.

No scientific rule may be selected using a future history-versus-current test
gain. Changes following development results need a new version and provenance.

## Sources

- [Original T1 source](https://drive.google.com/file/d/1116fOMjOUzIfyT-oYoI9gKv1bfoX63g9/view).
- [T2-E2 completed report](https://drive.google.com/file/d/1DyJAbpnvQXhIr_pqDwGJvQTU13VfypXf/view).
- [Observer design constraints](https://drive.google.com/file/d/1MwJIAc5stQ-6_987_ZPyBaa7L5M4ruLD/view).
- [Power et al., Grokking](https://arxiv.org/abs/2201.02177): precedent for studying
  learning/generalization on algorithmically generated datasets. It does not
  establish our sufficiency hypothesis or prescribe this pilot's recipe.
