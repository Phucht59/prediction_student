# Neural Sanity Ablation V2.2 preflight

- Run: `neural-sanity-v2-2-20260714`
- Source revision: `87509a3aa4f9c87af969153225d6e181a15afcf4`
- Source tree was clean at run creation.
- Dataset checksum: `e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80`
- Shared fold checksum: `bf5e5cbd8d09679f5d34900486ba23cc5ac57c93b28aa38ac3f3ce2578307ce1`
- Legacy-79 exclusion: enforced by the development-manifest runner guard before data preparation.
- Scenario/feature contract: `late_stage`, ordered `[G1, G2]`; G3 is excluded from model features.
- Matrix: fixed S0–S5 only; no Optuna search, no architecture search, no additional experiments.
- Expected-job contract was written before the first job: 150 jobs, 9,480 prediction rows.

Pre-run test command used the disposable PostgreSQL schema and completed with 125 passed, 0 failed, 0 skipped. The smoke run `neural-sanity-v2-2-smoke-20260714b` then validated 6/6 jobs before the full run was launched.
