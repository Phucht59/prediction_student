# Phase 5 Gemini robustness preparation

- Main run remains `LF_GEMINI_MAIN`, with its existing jobs/raw artifacts untouched.
- Repeatability experiment: `gemini_repeat_v1`, 150 shared Panel A cases, 15 jobs, seed `2026`, prompt `recommendation_label_v1`.
- Prompt robustness experiment: `gemini_prompt_v1b`, 150 identical cases, 15 jobs, prompt `recommendation_label_v1b`.
- Model: `gemini-3.5-flash-lite`; batch size: `10`.
- Sample coverage: stages 20/35/50/75, folds 0/1/2, risk bands Low/Borderline/High.
- Panel B cases: excluded. API calls made: none.

The comparison script reports LLM self-consistency for main versus repeat and prompt robustness
for main versus v1b. Neither experiment is an independent labeling function and neither is sent
to Snorkel. A4 diagnostics report numeric, ABSTAIN, and available abstain-reason rates; it flags
`A4 lacks observable evidence in current Student State` when all runs are nearly all ABSTAIN.
