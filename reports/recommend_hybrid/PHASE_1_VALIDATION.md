# recommend_hybrid Phase 1 authority validation

## Gate

`PHASE_1_PASS`. `scripts/recommend_hybrid/validate_phase1_authority.py` returned `RECOMMEND_HYBRID_PHASE1_AUTHORITY_PASS` (legacy gate alias: `RECOMMENDATION_V2_PHASE1_AUTHORITY_PASS`). All 30 real checkpoint files passed path, SHA-256, payload, architecture, parameter, fold, seed, stage mapping and read-only immutability checks.

## Dedicated validator scope

The validator checks the authority YAML, manifest and source training authority; exact stage policy; exclusion of historical recommendation and 65-checkpoint authorities; 30 checkpoint paths; file SHA-256; payload architecture hash; payload/state-dict parameter count; seed; config-hash-supported fold; unique expanded stage/fold/seed mappings; CPU read-only load; and before/after checkpoint hashes. It never loads a dataset or runs inference/training.

Expected inventory: 15 shared intervention files + 15 dedicated evaluation files = 30 files and 75 unique stage/fold/seed mappings. Expected missing/invalid: zero.

Actual inventory: expected 30, found 30, missing 0, invalid 0; 75 unique mappings. Architecture hash and 160,492 parameters matched both manifest and every loaded payload/state dict. Checkpoint hashes were identical before and after loading. Detailed output is stored in `reports/recommend_hybrid/logs/phase1_authority_validation.log`.

## Other validation

- JSON/YAML/CSV schema and naming checks are required.
- Import/compile smoke applies only to the new validator.
- No Ruff findings are allowed for the validator.
- The legacy repository validator remains outside this authority. Its seven existing report checksum mismatches are `PRE_EXISTING_LEGACY_VALIDATOR_FAILURE` and `NOT_RECOMMEND_HYBRID_AUTHORITY`.
- No full evaluation, GPU job, training or hyperparameter search is authorized.

Executed checks: authority validator PASS; Python compile smoke PASS; YAML/JSON/CSV schema PASS; Ruff PASS; naming/path validation PASS. The working tree scope check found only Phase 1 authority/config/manifest/validator/documentation changes. The old repository-wide validator was not reclassified: its seven report checksum mismatches remain `PRE_EXISTING_LEGACY_VALIDATOR_FAILURE` and `NOT_RECOMMEND_HYBRID_AUTHORITY`.

## Phase 2 invariance gate design

Hash checkpoint bytes before and after; clone and compare every state-dict name/shape/dtype/value; run same-device/same-dtype eval twice; compare logits, probabilities and classes before and after adapter attachment. Require exact equality when reusing the same forward outputs. If a different numerical execution path is unavoidable, derive tolerances from dtype epsilon, device kernels and repeated-run evidence, and record the rationale.
