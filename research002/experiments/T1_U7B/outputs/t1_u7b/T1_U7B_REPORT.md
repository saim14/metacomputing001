# T1-U7B — Fresh-Stream Training-Stability Repair

Date: 2026-09-04  
Status: **Passed development; failed fresh confirmation**  
Advancement decision: **Do not fit trajectory probes and do not advance to T1-U8**

## Research question

Does one fixed 2,000-update extension across four new two-hop training streams
repair the seed-level learning failure found in T1-U7 while leaving the task,
model, causal controls, continuous value target, and evaluation gates unchanged?

T1-U7B replayed the exact 4,000-update T1-U7 prefix from initialization so that
the optimizer state was reproduced. It then appended four 500-update,
final-only phases using independent stream IDs 2, 3, 4, and 5. This intervention
jointly changes training duration and fresh-data exposure; it does not isolate a
pure duration effect from a pure stream-diversity effect.

## Frozen gate

A seed passes only if all of the following hold:

- clean accuracy is at least 0.95;
- paired-counterfactual accuracy is at least 0.90;
- corrupting either required edge reduces accuracy into [0.40, 0.60]; and
- at least two intermediate steps have both mean and standard deviation of
  remaining NLL reduction at least 0.02.

The untouched seeds 811, 907, and 1009 could be opened only after every exposed
development seed passed. No trajectory probe was permitted in T1-U7B.

## Results

### Development repair

| Seed | Clean | Paired CF | Break edge 1 | Break edge 2 | Valid value steps | Pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 503 | 0.9925 | 0.9915 | 0.5190 | 0.4950 | 3 | Yes |
| 607 | 1.0000 | 1.0000 | 0.4915 | 0.4975 | 3 | Yes |
| 709 | 0.9980 | 0.9965 | 0.4970 | 0.5095 | 3 | Yes |

All three development seeds passed. Relative to T1-U7, seed 503 improved by
0.0620 in clean accuracy and 0.1020 in paired-counterfactual accuracy. Seeds 607
and 709 retained their earlier success. Edge-corruption accuracy remained near
chance, so the improvement did not arise from bypassing either required chain
link.

### Untouched fresh confirmation

| Seed | Clean | Paired CF | Break edge 1 | Break edge 2 | Valid value steps | Pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 811 | 0.7123 | 0.5660 | 0.4990 | 0.4960 | 4 | No |
| 907 | 0.7290 | 0.6455 | 0.5360 | 0.4955 | 4 | No |
| 1009 | 0.7093 | 0.5895 | 0.4740 | 0.4960 | 3 | No |

No fresh seed passed. Their mean clean accuracy was 0.7168, compared with
0.9968 for the development seeds, a difference of 0.2800. Their mean
paired-counterfactual accuracy was 0.6003, compared with 0.9960 in development,
a difference of 0.3957.

Every fresh seed nevertheless kept both edge-corruption accuracies inside the
chance band and retained at least three valid computation-value steps. The
failure is therefore not evidence of a simple shortcut or disappearance of the
continuous value signal. It is a failure to learn the two-hop task reliably
from these initializations.

## Interpretation

The fresh-stream extension is an effective *development repair* but not a
seed-robust training protocol. It rescued the previously failing development
seed without damaging the two previously passing seeds, yet this success did
not generalize to any untouched initialization.

The learning curves expose two regimes. Development seeds eventually enter a
high-accuracy basin and converge near 1.0. All three fresh seeds instead plateau
near 0.7 despite the extra 2,000 final-only updates and four independent data
streams. More final-only data exposure is therefore not an adequate immediate
solution to the optimization instability.

The result also sharpens an important methodological distinction: a usable
remaining-computation target can exist even when the base model is not competent
enough for a meaningful controller study. Passing the value-availability gate
alone does not authorize trajectory probes.

## Decision and next bounded direction

T1-U7B accepts the null branch of its preregistered decision rule:

- fresh confirmation failed;
- seeds 811, 907, and 1009 are now exposed and cannot be reused as untouched
  confirmation seeds;
- no trajectory probes are fit;
- T1-U8 remains unauthorized.

The most diagnostic next step is a separately preregistered T1-U7C training-
signal intervention, not another unbounded increase in final-only updates. A
clean candidate is to replay the same 6,000-update schedule and fresh streams
while retaining a small intermediate auxiliary loss during the 2,000-update
extension. That changes one factor relative to T1-U7B: whether the extension
continues to supervise intermediate computation. It should first be tested on
the now-exposed six seeds, with a newly reserved seed set opened only after an
all-seed development pass.

## Reproducibility

The output directory contains the preregistered protocol, closed-reference
verification, per-seed training histories, causal metrics, continuous
trajectory-value statistics, gate tables, paired T1-U7 effects, environment
manifest, decision record, and summary figure. The completed run took
4,101.3 seconds in the recorded environment.
