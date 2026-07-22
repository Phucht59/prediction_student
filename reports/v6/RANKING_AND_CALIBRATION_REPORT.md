# V6 ranking and calibration report

Risk ranking did not pass the registered compatibility guardrails; Candidate C
remained frozen. Temperature scaling was fitted only on valid Candidate C/W0
inner-OOF predictions from outer-training fold 0.

- Temperature: 1.043603
- Inner NLL: 0.368259 -> 0.368024
- Inner Brier: 0.116544 -> 0.116558
- Inner ECE: 0.024388 -> 0.025915
- Diagnostic slope/intercept: 0.911252 / -0.244636

Outer-fold values are reporting-only and never enter the calibrator fit.
