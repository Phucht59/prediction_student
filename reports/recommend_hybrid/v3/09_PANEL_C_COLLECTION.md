# 09 — Panel C collection

**STATUS: COMPLETE**

Authentic Gemini only. No fabricated reviews. No case replacement.

| Item | Value |
|---|---|
| Students | 150 |
| Cases | 632 |
| Review records | 2398 |
| Abstain | 0 |
| Provider failures | 0 |
| Prompt | `panel_c_external_reviewer_v3_c0_blinded_v1` |
| Prompt SHA-256 | `714800f716feff78f431e1dd6d82b24ed0a2c38264db90d87880c7b9b282a7b4` |
| Prompt changed after open | false |

Pass 1 used frozen `gemini-3.5-flash-lite` (1910 records, 501 cases) until the free-tier daily cap of 500 stopped the remainder.

The owner then authorized finishing the remaining 131 cases with `gemini-3.1-flash-lite` (488 records). Same frozen prompt, same case payloads, no silent model swap.

Mixing policy: `EXPLICIT_OWNER_AUTHORIZED_GEMINI_3_5_THEN_3_1_FLASH_LITE`.
