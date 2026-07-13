# Benchmark V2 probability-fix preflight

Status: **passed for smoke execution**.

- Shared manifest checksum: `bf5e5cbd8d09679f5d34900486ba23cc5ac57c93b28aa38ac3f3ce2578307ce1`.
- Dataset checksum: `e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80`.
- The development-only manifest is used; `assert_no_legacy_records` is executed by the runner before any fold construction.
- Late-stage and early-warning allowlists are checked before execution. G3 is not an input feature.
- The fallback contract is deterministic one-hot in official Low/Medium/High order. The central strict validator rejects non-finite values, values outside `[0,1]`, sums outside `1e-6`, and label/argmax disagreement.
- `80 passed, 5 skipped` unit/protocol tests. The five skips are environment-blocked PostgreSQL lineage integrations; no DSN or `psql` was available in this process.
- No split, feature, target, search-space, seed, tuning budget, early-stopping, resampling, class-weight, loss, or architecture setting changed.

The checked-out Git commit is `6526065b69f9bd5027274c898fc76b9208f930f2` and the worktree is dirty because the probability fix and the earlier Protocol V2 implementation have not been committed. A new full scientific run must record a frozen source revision rather than present this dirty worktree as a new commit.
