# Final Release Status

- Final source commit: `03987f51662221771a17a721a4b2c9a817c21302`
- Branch: `codex/final-model-evidence`
- Selection run: `nested-full-20260710`
- Scientific DB run: `a2945d79-9845-4979-b148-159f4853eca3`
- Clean-commit verification run: `c719439e-bb88-42ff-bb98-d258c21d204e`
- Selected config SHA-256:
  `cda38460197627ac1d71e764f61d784e4c03cf6f86775339d38787c6890678ad`
- Dataset SHA-256:
  `e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80`
- Final evidence:
  `artifacts/final/final-a2945d79-9845-4979-b148-159f4853eca3/`
- Tests from clean commit: `88 passed, 5 skipped`
- PostgreSQL integrity: passed (316 train, 79 test, 79 predictions, 79 recommendations)
- Reproducibility: exact prediction checksum match and zero metric difference.

## Known limitations

- CNN–BiLSTM final Macro-F1 0.9262 remains below G2 rule 0.9365 and HGB
  locked-test 0.9463; no overall superiority claim is valid.
- Recommendation is rule-based advisory support. Expert review remains pending.
- The sample is UCI Portuguese secondary-school data, not Vietnamese university
  students. G1/G2 are two prior assessments, not a long multi-semester series.

DOCX files were not modified. Next stage: thesis report revision from the frozen
evidence only.
