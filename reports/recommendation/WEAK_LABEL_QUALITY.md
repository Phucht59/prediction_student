# Supported-action weak-source comparison

This measures **LLM weak-source agreement**, not human inter-rater agreement. A4 is excluded because it is `UNSUPPORTED_BY_CURRENT_STATE`; A5 remains included and is flagged `REVIEW`.

- Gemma normalized rows: `2000`
- Gemini normalized rows: `2000`
- Supported actions: `A1, A2, A3, A5`

## Overall

- Exact agreement: `1373/2000` (0.686500)
- Disagreement: `627/2000`
- Quadratic weighted Cohen kappa, numeric non-ABSTAIN pairs: `0.706167`
- Gemma ABSTAIN rate: `0.274000`; distribution: `{'0': 228, '1': 511, '2': 443, '3': 270, 'ABSTAIN': 548}`
- Gemini ABSTAIN rate: `0.274000`; distribution: `{'0': 259, '1': 405, '2': 518, '3': 270, 'ABSTAIN': 548}`

## By action

| Action | Exact | Rate | Weighted kappa | Gemma ABSTAIN | Gemini ABSTAIN |
|---|---:|---:|---:|---:|---:|
| A1 | 478/500 | 0.956000 | 0.299774 | 0.718000 | 0.718000 |
| A2 | 294/500 | 0.588000 | 0.769998 | 0.000000 | 0.000000 |
| A3 | 282/500 | 0.564000 | 0.566416 | 0.000000 | 0.000000 |
| A5 | 319/500 | 0.638000 | 0.044278 | 0.378000 | 0.378000 |

### Label distributions by action

- `A1`: Gemma `{'0': 0, '1': 0, '2': 19, '3': 122, 'ABSTAIN': 359}`; Gemini `{'0': 0, '1': 0, '2': 17, '3': 124, 'ABSTAIN': 359}`
- `A2`: Gemma `{'0': 179, '1': 53, '2': 120, '3': 148, 'ABSTAIN': 0}`; Gemini `{'0': 192, '1': 98, '2': 122, '3': 88, 'ABSTAIN': 0}`
- `A3`: Gemma `{'0': 25, '1': 282, '2': 193, '3': 0, 'ABSTAIN': 0}`; Gemini `{'0': 53, '1': 190, '2': 203, '3': 54, 'ABSTAIN': 0}`
- `A5`: Gemma `{'0': 24, '1': 176, '2': 111, '3': 0, 'ABSTAIN': 189}`; Gemini `{'0': 14, '1': 117, '2': 176, '3': 4, 'ABSTAIN': 189}`

## By stage

- `20pct`: exact=`372/532` (0.699248); kappa=`0.663767`; Gemma ABSTAIN=`0.334586`; Gemini ABSTAIN=`0.334586`
- `35pct`: exact=`358/516` (0.693798); kappa=`0.696820`; Gemma ABSTAIN=`0.292636`; Gemini ABSTAIN=`0.292636`
- `50pct`: exact=`314/488` (0.643443); kappa=`0.683929`; Gemma ABSTAIN=`0.235656`; Gemini ABSTAIN=`0.235656`
- `75pct`: exact=`329/464` (0.709052); kappa=`0.769262`; Gemma ABSTAIN=`0.224138`; Gemini ABSTAIN=`0.224138`

## By risk band

- `high`: exact=`328/436` (0.752294); kappa=`0.549227`; Gemma ABSTAIN=`0.178899`; Gemini ABSTAIN=`0.178899`
- `low`: exact=`727/1088` (0.668199); kappa=`0.494275`; Gemma ABSTAIN=`0.304228`; Gemini ABSTAIN=`0.304228`
- `medium`: exact=`318/476` (0.668067); kappa=`0.458854`; Gemma ABSTAIN=`0.292017`; Gemini ABSTAIN=`0.292017`

## Review flag

- `A5`: REVIEW — retain for now; do not remove automatically.
