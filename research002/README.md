# Research 002 — Metacognitive Analysis in Attention

## Research map

| Branch | Recovered work | Role |
|---|---|---|
| Transformer computational history | T1 baseline/source, T1.1 diagnostics/source, T1-U, U2–U7, U7B, U7C, matched U7BC audit | Test training readiness and history beyond the current Transformer state |
| Observer controls | T2-E1, T2-C1 sensitivity, T2-E2 | Test how readout choice and stronger controls change apparent history gains |
| Present-state learning | P1-A0, P1-A0B, P1-A1 | Investigate present-state access under frozen development constraints |
| Foresight | F1-A0, F1-A1, F1-A2, F1-A3 | Test geometry, controlled model error and candidate-specific cost bias |
| Proposed next study | F1-A4 | Decision-focused learning versus acquiring extra bias information; not run or frozen |

## Latest completed result

[F1-A3](experiments/F1_A3/main_v1/F1_A3_Report.md) is complete and audited. Eight-candidate correction was faster than the full estimator but did not meet the fixed prediction, accuracy-retention, charged-control or reference-runtime gates. The report preserves the detectable small prediction improvement and the uncertainty around practical usefulness.

[Current working state and research decisions](PROJECT_STATE.md) tracks the branch map, fixed findings and proposed next design. The next question is whether selecting and learning for stop/continue regret improves the use of already-available information. This proposed F1-A4 has not been run. T1 sealed seeds and P1 reserve families remain unused.

F1-A2's prior prediction result remains unchanged; the new-scene F1-A3 result narrows confidence in practical transfer. Read each protocol for its exact scope.

## Reproduce a study

1. Clone the repository, including archive parts.
2. From the repository root, run `python research002/restore_archives.py --study F1_A2` (replace the study ID as appropriate). Omitting `--study` restores all archives.
3. Open the restored study's README. Install its stated dependencies and run its integrity/audit steps before any reproduction that changes run files.
4. Fresh reproductions belong in separate directories. A replay is not independent confirmation.

The archive manifest lists exact byte sizes, SHA-256 hashes, original members and extraction prefixes. `RECOVERY_VERIFICATION.json` records ZIP CRC verification of all 23 recovered archives. Original bytecode is retained inside byte-identical archives; it is not part of the browsable source selection.

The original first-study notebook remains at the repository root. This is a snapshot of recovered research artifacts; it does not claim to contain every conversational message or the website's deployment source. The direction-freeze and observer-design records are preserved under `context/`.

