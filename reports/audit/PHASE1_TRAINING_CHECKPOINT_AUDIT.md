# Phase 1 — Training and Checkpoint Audit

## Finding: why `selected_epoch = 1`

Status: **CONFIRMED BUG (metadata), NOT a one-epoch training run**.

All 45 unified OULAD deep checkpoints—CNN-only, BiLSTM-only, and
CNN-BiLSTM; 3 folds × 5 seeds—contain:

- `selected_epoch = 1`;
- `config.max_epochs = 4`;
- a manifest SHA-256 that matches the checkpoint file.

The source establishes the execution path:

1. `train()` sets `selected_epoch` to the configured `max_epochs` whenever
   inner-trial rows exist (`src/pipelines/oulad.py:960`), which yields `4`.
2. `_fit_deep(... selected_epoch=4)` sets `fixed=4` and executes
   `for epoch in range(1, fixed + 1)` (`:788-789`).
3. In fixed-epoch mode, validation and best-state updates are bypassed by
   `continue` (`:804`).
4. `best_epoch` remains its initialization value `1` (`:788`).
5. The payload serializes and returns this unchanged value (`:811-812`).

Therefore the checkpoints are last-state four-epoch refits mislabeled as epoch
1. There is no state-dict evidence that an epoch-1 best checkpoint was restored.

## Unified OULAD training behavior

| Item | Actual behavior |
| --- | --- |
| Train loss | Weighted binary BCE; CNN-BiLSTM adds `0.15 × survival BCE + 0.15 × outcome CE` |
| Positive weight | `(negative_count / positive_count)` on the fit partition |
| Stage balancing | Per-record weights equalize total contribution over available stage views |
| Sampler | Shuffled DataLoader; no balanced sampler |
| Batch size | 256 |
| Optimizer | AdamW |
| Learning rate | 0.0005 |
| Weight decay | 0.00001 |
| Gradient clipping | Norm 1.0 |
| Scheduler | None |
| Inner max epochs / patience | 4 / 2 |
| Inner validation frequency | Once per epoch |
| Inner early-stop metric | Mean-stage binary Macro-F1 at threshold 0.5 |
| Improvement mode / delta | Maximize; strict `> best + 1e-8` |
| Outer refit | Exactly 4 epochs, no validation checkpoint selection |
| Restore best checkpoint | Inner only; outer fixed refit uses last state |
| Epoch numbering | One-based |

CNN-only and BiLSTM-only use the same risk loss and training loop; only
CNN-BiLSTM activates survival/outcome auxiliary losses.

## Confirmed objective mismatch

Status: **CONFIRMED DESIGN MISMATCH**.

Inner epoch selection monitors Macro-F1 at threshold `0.5`
(`src/pipelines/oulad.py:805-808`). Final unified evaluation uses a separate
stage/fold threshold chosen from pooled inner-OOF data
(`:838-875`, `:1050-1052`). An epoch that is best at `0.5` need not be best
after operational threshold selection.

The mismatch does not directly select the current outer checkpoint, because
the outer refit ignores inner best epochs and always runs four epochs. It does,
however, make the inner training evidence unsuitable as a checkpoint-selection
proxy for the reported operational objective.

## Fixed-epoch propagation issue

Status: **CONFIRMED DESIGN ISSUE**.

Unlike unified UCI, which records inner best epochs and refits for their median,
unified OULAD discards the inner best epoch and uses `max_epochs=4` for every
fold/model. Patience is consequently irrelevant to outer refit. This is a
plausible training limitation, but its metric impact is not estimated in Phase
1.

## Checkpoint and run identity

- Checkpoint file hashes: **PASS**, 0/45 deep mismatches.
- Four stages map to one path and SHA per fold/seed: **PASS**, 0/45 failures.
- Payload/manifest training run ID: **FAIL**, 45/45 mismatches.

The run-ID mismatch is deterministic: the checkpoint payload ID is built from
`{"dataset","model","outer","seed","config"}`, whereas the manifest ID is built
from `{"dataset","model","outer","seed","config_hash"}` at
`src/pipelines/oulad.py:989` and `:995`.

## Diagnostic decision

No new training was required to explain `selected_epoch=1`: source control flow,
all 45 checkpoint payloads, manifest metadata, and hashes agree on the metadata
bug. A controlled inner-only learning-curve diagnostic remains recommended
before changing the training budget, because no retained unified curve logs
show whether four epochs underfit, plateau, or overfit.
