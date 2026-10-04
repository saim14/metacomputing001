# Research 002 — Metacognitive Analysis in Attention

## Research map

| Branch | Recovered work | Role |
|---|---|---|
| Transformer computational history | T1 baseline/source, T1.1 diagnostics/source, T1-U, U2–U7, U7B, U7C, matched U7BC audit | Test training readiness and history beyond the current Transformer state |
| Observer controls | T2-E1, T2-C1 sensitivity, T2-E2 | Test how readout choice and stronger controls change apparent history gains |
| Present-state learning | P1-A0, P1-A0B, P1-A1 | Investigate present-state access under frozen development constraints |
| Foresight | F1-A0, F1-A1, F1-A2 | Test geometry, controlled model error and candidate-specific cost bias |
| Next methods study | F1-A3 | Started 4 October; engineering pilot and draft protocol only |

## Latest established result

[F1-A2](experiments/F1_A2/F1_A2_Report.md) found useful prediction from distortions of candidate-plan costs. Its accessible estimate passed the fixed prediction criterion, but charged controller improvement did not pass adjusted uncertainty, and runtime increased. History and physical-path geometry added no demonstrated gain on top of the new bias signal.

The current next problem is **preserving useful bias information at lower online cost**. [F1-A3](experiments/F1_A3/README.md) records the first prototype and its limitations. T1 sealed seeds and P1 reserve families remain unused by this continuation.

## Reproduce a study

1. Clone the repository, including archive parts.
2. From the repository root, run `python research002/restore_archives.py --study F1_A2` (replace the study ID as appropriate). Omitting `--study` restores all archives.
3. Open the restored study's README. Install its stated dependencies and run its integrity/audit steps before any reproduction that changes run files.
4. Fresh reproductions belong in separate directories. A replay is not independent confirmation.

The archive manifest lists exact byte sizes, SHA-256 hashes, original members and extraction prefixes. `RECOVERY_VERIFICATION.json` records ZIP CRC verification of all 23 recovered archives. Original bytecode is retained inside byte-identical archives; it is not part of the browsable source selection.

The original first-study notebook remains at the repository root. This is a snapshot of recovered research artifacts; it does not claim to contain every conversational message or the website's deployment source. The direction-freeze and observer-design records are preserved under `context/`.
