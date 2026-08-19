# 09 — Panel C collection

**STATUS: INCOMPLETE — authentic Gemini pass 1 of 2**

Panel C was opened with authentic `gemini-3.5-flash-lite` calls. No fabricated reviews.

| Item | Value |
|---|---|
| Provider | Google Gemini API |
| Model | gemini-3.5-flash-lite |
| Prompt | `panel_c_external_reviewer_v3_c0_blinded_v1` |
| Prompt SHA-256 | `714800f716feff78f431e1dd6d82b24ed0a2c38264db90d87880c7b9b282a7b4` |
| Cases sampled | 632 (150 students) |
| Cases reviewed (pass 1) | 501 |
| Review records (pass 1) | 1910 |
| Provider failures (pass 1) | 131 |
| Model substitution | false |
| Prompt changed after open | false |
| Case replacement | false |

Pass-1 failures are HTTP 429 on the free tier:

- per-minute cap 15
- daily cap 500 `generate_content_free_tier_requests`

Remaining 131 cases were **not** filled with synthetic labels. They will be collected with the same frozen prompt and model after the daily quota resets (00:00 America/Los_Angeles).

`complete_coverage_required = true`, so Panel C is not complete and is not used for a held-out claim until the remaining authentic reviews exist.
