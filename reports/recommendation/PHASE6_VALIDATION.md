# Phase 6 validation

- Phase scope: behavioral LF finalization, LLM source normalization, diagnostics, and source manifest only.
- API calls: `0`.
- Snorkel execution: `0`.
- Silver-label generation: `0`.
- EBM training: `0`.

## Integrity

- Panel A cases: `500`; Panel B cases: `150`.
- Panel B overlap in canonical LLM rows: `0`.
- Panel B overlap in Behavioral rows: `0`.
- Canonical effective LLM rows: `5000`; expected `5000`.
- Behavioral rows: `2500`; expected `2500`.
- Canonical duplicate grain: `0`.
- Forbidden fields in canonical state/label artifact: `NONE`.
- Source manifest Panel-B overlap: `0`.

## Leakage contract

All Phase 6 labels consume Panel-A Student State or existing frozen LLM artifacts. No final_result, future activity/assessment/unregistration, prediction truth label, or Panel-B row is used for LF fitting, threshold derivation, normalization, diagnostics, or source registration. A4 course_progress is not converted into a progress_gap; Behavioral A4 remains ABSTAIN because course_progress is stage-like.

## Phase 7 gate

The Phase 7 source manifest is complete and variable-LF input is prepared. Phase 7 may start only as a separate user-authorized task; it was not executed here.
