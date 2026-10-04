# T2-C1 preflight record

Before the first calibration run, numerical equivalence and leakage checks passed, but the method-verification script initially required the Wilson interval lower boundary at zero successes to equal zero exactly. Floating-point evaluation returned approximately 1.39e-17 and raised `AssertionError` in that boundary check.

The verification assertion was changed to use an absolute tolerance of 1e-15 for that boundary. The interval implementation, scientific protocol, generator, probe code, screening rules, and calibration outputs were unchanged. The full preflight subsequently passed. No calibration panel was run before this correction and no experimental run was repeated.
