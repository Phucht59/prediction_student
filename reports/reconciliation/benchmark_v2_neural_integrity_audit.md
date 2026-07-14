# Benchmark V2 neural integrity audit

Static audit of the frozen V2 source and valid run found: a new model and Adam optimizer are constructed inside each fold invocation; BatchNorm belongs to that fresh model; `train_epoch` calls `model.train()` and validation/inference call `model.eval()`; LSTM hidden state is not retained across batches; DataLoader training shuffle is true and scoring shuffle false; labels are 0/1/2 in Low/Medium/High order; CNN receives `[N,C,L]` through transpose and LSTM receives `[N,L,F]`; final hidden concatenation is `hidden[-2] || hidden[-1]`.

Early stopping uses an internal stratified outer-train split. The scoring fold is transformed but is not passed to `train_model`; epoch is selected internally and V2 then refits a fresh model/preprocessor/selector on all outer-train for fixed epochs. Checkpoints are saved per fold/seed and prediction rows carry record identity/fold/seed/config checksum. Existing 85 tests pass, including protocol, record-order, probability, fold, leakage, and PostgreSQL integration tests.

No evidence of a V2 implementation bug that would invalidate `benchmark-v2-full-20260713c` was found. This is a static/code-and-artifact audit; unavailable historical checkpoints prevent bitwise historical reproduction.
