# Phase 3-4 Validation

| Gate | Result | Evidence |
|---|---|---|
| Authority reconciliation | PASS | AUTHORITY_RECONCILIATION.md |
| Feasibility 5 actions/case | PASS | 500,305 rows; 5 per state case |
| Feasibility/relevance separation | PASS | no risk/engagement relevance rules |
| No invented availability | PASS | A4 UNKNOWN; A5 zero activity UNKNOWN |
| Panel A target | PASS | 500 rows |
| Panel B target | PASS | 150 rows |
| Panel case overlap | PASS | 0 |
| Panel student overlap | PASS | 0 |
| Panel enrollment overlap | PASS | 0 |
| FINAL exclusion | PASS | no FINAL-100 |
| Deterministic seed | PASS | 2026 |
| State source coverage | PASS | all panel case_ids exist in reconciled state |

## Feasibility distribution

| Action | Status | Rows |
|---|---|---:|
| A1 | FEASIBLE | 27099 |
| A1 | INFEASIBLE | 72962 |
| A2 | FEASIBLE | 100060 |
| A2 | INFEASIBLE | 1 |
| A3 | FEASIBLE | 100061 |
| A4 | UNKNOWN | 100061 |
| A5 | FEASIBLE | 65830 |
| A5 | UNKNOWN | 34231 |

## Panel distributions

| Panel | Stage counts | Outer-fold counts | Risk-band counts | Covered strata |
|---|---|---|---|---:|
| Panel A | `{"20pct": 133, "35pct": 129, "50pct": 122, "75pct": 116}` | `{"0": 168, "1": 167, "2": 165}` | `{"Borderline": 119, "High": 109, "Low": 272}` | 36/36 |
| Panel B | `{"20pct": 39, "35pct": 38, "50pct": 37, "75pct": 36}` | `{"0": 49, "1": 51, "2": 50}` | `{"Borderline": 37, "High": 36, "Low": 77}` | 36/36 |

Panel A SHA-256: `8980047221bc83a4a9f7f9ce4c3ecde5be5c275193a1cecd3000c5db5c67b20f`
Panel B SHA-256: `bea69335b089c7a7de365f33176f758cb0bd1fcadff6c8bb6b196d44df77ef84`
Manifest SHA-256: `bb85b6374c037daf12e9e12c74ffc3b15fae512942fde36b6838b52a0ea16645`
