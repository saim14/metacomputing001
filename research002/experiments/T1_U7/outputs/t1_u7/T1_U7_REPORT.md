# T1-U7 — Bounded Two-Hop Learnability and Causal-Dependence Gate

## Decision

**Status:** `failed_development_gate_fresh_seeds_unopened`

T1-U7 established a valid shortcut-controlled two-hop computation, but the
fixed T1-U4-D training protocol was not reliable across all exposed development
seeds. Seeds 607 and 709 passed every preregistered gate. Seed 503 retained the
required causal and trajectory properties but missed the clean-accuracy and
paired-counterfactual thresholds. The fresh model seeds `[811, 907, 1009]`
therefore remained unopened, and no trajectory probe was fit.

This is a bounded negative decision with a constructive result: the research
now has a genuine multi-step task and a nondegenerate continuous
value-of-computation target. The remaining obstacle is optimization reliability,
not task validity or inference pacing.

## Research question

Can the fixed six-step, weight-tied Transformer learn a balanced two-hop
retrieval task, demonstrate causal dependence on both required edges, and
retain measurable future computational value before trajectory-history probes
are considered?

The task is

\[
q \rightarrow m, \qquad m \rightarrow v,
\]

where the model observes the query and a randomized collection of mappings and
must predict the binary terminal value \(v\).

## Preregistered protocol

- Exposed development seeds: `503`, `607`, `709`.
- Fresh confirmation seeds: `811`, `907`, `1009`, opened only after a complete
  development pass.
- Five mappings per sequence: two chain edges and three distractors.
- Two occurrences of each binary value token in every sequence.
- Randomized relation order and balanced labels.
- No direct \(q\rightarrow v\) relation.
- Six recurrent refinement steps using one shared Transformer block.
- State width 32, FFN width 64, two attention heads.
- Residual scale 0.5 and the fixed T1-U4-D tapered auxiliary-loss schedule.
- 4,000 updates per seed; no adaptive stopping.
- No trajectory probes.

### Matched causal controls

The counterfactual keeps keys, sources, and relation positions fixed while
complementing every binary value token. This preserves within-sequence token
counts and flips the correct answer.

The first-edge intervention redirects the query to an unused sink key. The
second-edge intervention preserves the token inventory while swapping sources
so the intermediate node points to a fair random value. A valid learned chain
should therefore remain accurate on clean and counterfactual examples but fall
to chance after either edge is broken.

### Continuous computation-value target

For refinement step \(l\), remaining computational value is

\[
V_l = \mathcal{L}_l - \mathcal{L}_L,
\]

where \(\mathcal{L}_L\) is final-step negative log-likelihood. A step counts as
value-bearing when the mean and sample standard deviation of \(V_l\) are each
at least 0.02 nats. Every seed must contain at least two such intermediate
steps. Future prediction flips are recorded only as a secondary diagnostic.

### Joint seed gate

Every exposed seed had to satisfy all conditions:

1. Clean final accuracy at least 0.95.
2. Matched paired-counterfactual accuracy at least 0.90.
3. First-edge corrupted accuracy within `[0.40, 0.60]`.
4. Second-edge corrupted accuracy within `[0.40, 0.60]`.
5. At least two value-bearing intermediate steps.

## Results

| Seed | Clean accuracy | Paired counterfactual | Break first edge | Break second edge | Valid value steps | Passed |
|---:|---:|---:|---:|---:|---|---|
| 503 | 0.93050 | 0.88950 | 0.51100 | 0.49200 | 1, 2, 3, 4 | No |
| 607 | 0.99550 | 0.98950 | 0.49550 | 0.49650 | 1, 2, 3 | Yes |
| 709 | 0.98400 | 0.97050 | 0.48400 | 0.50750 | 1, 2, 3, 4 | Yes |

Mean clean accuracy was 0.97000. Mean paired-counterfactual accuracy was
0.94983. Mean first- and second-edge corrupted accuracies were 0.49683 and
0.49867 respectively. Thus both causal controls were essentially at chance
across all seeds.

Seed 503 missed clean accuracy by 0.01950 and paired-counterfactual accuracy by
0.01050. These are not independent failures: paired correctness is necessarily
bounded by performance on each member of the pair. Its counterfactual
prediction-flip consistency was 0.90700, and both edge controls passed.

## Layer-wise computation

Accuracy rose systematically across refinement steps:

| Seed | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Final |
|---:|---:|---:|---:|---:|---:|---:|
| 503 | 0.50325 | 0.69375 | 0.83625 | 0.90600 | 0.92975 | 0.93050 |
| 607 | 0.51675 | 0.70850 | 0.95025 | 0.99025 | 0.99600 | 0.99550 |
| 709 | 0.52450 | 0.73500 | 0.91650 | 0.96875 | 0.98400 | 0.98400 |

At step 1, mean remaining NLL reduction ranged from 0.79630 to 1.08902 nats.
At step 3 it remained between 0.11589 and 0.21687 nats. This is substantially
stronger and more naturally graded temporal structure than the manufactured
flip windows in the earlier one-hop experiments.

## Training dynamics

All seeds remained near chance for hundreds of updates before entering a sharp,
seed-dependent learning transition. The first logged validation score at or
above 0.95 occurred at update 3,700 for seed 607 and update 3,500 for seed 709.
Seed 503 was still improving at the final logged update but ended at 0.925 on
validation and 0.93050 on the independent development sample.

This pattern rules out a claim that the architecture cannot perform the task:
two seeds learned it almost perfectly. It also rules out treating mean
performance as sufficient: a mean clean accuracy of 0.97 conceals one failed
initialization.

## Scientific interpretation

T1-U7 supports four conclusions.

1. **The two-hop construction is valid.** Breaking either required edge reduced
   every seed to chance, so high clean accuracy cannot be explained by value
   counts, token identity, or direct retrieval.
2. **Transformer depth carries useful computation.** Intermediate accuracy and
   remaining NLL reduction show a graded refinement process across multiple
   recurrent steps.
3. **The continuous value target is viable.** Every seed supplied at least three
   preregistered value-bearing steps, without requiring artificial flip-rate
   manipulation.
4. **Learning remains initialization-sensitive.** The unchanged one-hop-derived
   4,000-update schedule is not reliable enough to authorize confirmatory
   trajectory probing.

The result therefore advances the research object while correctly blocking the
next inferential claim. We have demonstrated a suitable computational setting,
but not yet a sufficiently stable training procedure.

## Next bounded study

Do not proceed directly to T1-U8 probes and do not reopen residual-pacing search.
The next study should be labelled **T1-U7B**, because it repairs the learning
gate within the same two-hop experimental stage.

The strongest single intervention is a preregistered extension using additional
fresh two-hop training streams rather than repeatedly resampling the same finite
stream. Architecture, residual scale, two-hop task, causal controls, and all
evaluation gates should remain fixed. This directly tests whether seed 503's
failure reflects insufficient algorithmic generalization time while avoiding a
new capacity or pacing confound.

Only after all exposed seeds pass T1-U7B should fresh model seeds be opened. A
fresh-seed pass would authorize T1-U8, which tests whether ordered history adds
predictive information about \(V_l\) beyond the current state and current
confidence.

## Integrity boundary

- Fresh confirmation seeds were not trained or evaluated.
- No sealed trajectories or metrics were created.
- No state-history, motion, attention-history, or other trajectory probe was fit.
- No threshold or training parameter was altered after observing results.
