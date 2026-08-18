# Phase 5 labeling infrastructure

- Prompt/rubric: `recommendation_label_v1`; one provider-neutral rubric for Gemma and Gemini.
- Action taxonomy: A1 Assessment Recovery, A2 Re-engagement, A3 Study Planning, A4 Content Review, A5 Retrieval Practice.
- Label domain: `0/1/2/3/ABSTAIN`; infeasible action is `ABSTAIN` with reason `INFEASIBLE`, never numeric zero.
- Panel A coverage: 500 cases; Panel B cases are excluded from all jobs.
- Pilot coverage: 30 cases; pilot is deterministically selected with seed `2026` across stages, folds, and risk bands.
- Default batch size: `10`; supported sizes: 1, 5, 10.
- Models: Gemma `gemma-4-31b-it`, Gemini `gemini-3.5-flash-lite`.
- Jobs: Gemma 50 full / 3 pilot; Gemini 50 full / 3 pilot.
- API calls made during generation: none. API keys are not stored in jobs.

Runners write raw response records only after the user runs them locally. Snorkel, label modeling,
Panel B labeling, recommendation prose, and human evaluation are out of scope for this phase.
