# Reproducibility and testing

- Final evidence run: `a2945d79-9845-4979-b148-159f4853eca3`.
- Selection run: `nested-full-20260710`.
- Live DB-first verification run: `5a0b5041-5216-4a48-9e46-b0c16ab14866`.
- Prediction checksum: `d5b6f86d50a1a4c90b6a68139ec0eb6f4635e55c572c647d6d9b62d5a31f4a74`.
- Reproduction prediction checksum matches exactly; reported metric delta is 0.
- Selected-config SHA-256:
  `cda38460197627ac1d71e764f61d784e4c03cf6f86775339d38787c6890678ad`.
- Current main source commit before context generation: `50d7696`.
- Tests: 62 passed, 0 skipped. All five PostgreSQL integration tests ran with
  `POSTGRES_TEST_DSN` and `POSTGRES_TEST_APP_DSN`.

Test groups cover unit behavior, leakage/preprocessing, deterministic splits,
artifact metric recomputation, recommendation structure and PostgreSQL lineage
contracts. Evidence verifier recomputes locked-test metrics and validates hashes.
Live PostgreSQL migration and DB-first verification are complete. Independent
expert review of recommendation cases is not complete.
