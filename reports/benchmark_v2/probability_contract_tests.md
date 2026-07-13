# Probability contract tests

The central validator requires vectors in Low/Medium/High order, finite values in `[0,1]`, row sum within `1e-6`, and argmax equal to the stored predicted label. The tolerance is for native float32 neural softmax numerical precision after serialization to float64; it still rejects the historical `0.001` error by three orders of magnitude.

Regression coverage includes all three labels, a multi-record batch, JSON serialization/deserialization, acceptance of valid one-hot vectors, acceptance of a float32-softmax-level numerical residual, and rejection of the historical invalid vector `[0.999, 0.001, 0.001]`. The full suite result after this change is **80 passed, 5 skipped**. The five skips are the PostgreSQL lineage integration tests in `tests/test_postgres_source_ml_integration.py`, blocked because this environment exposes neither `POSTGRES_TEST_DSN`/`POSTGRES_TEST_APP_DSN` nor `psql`.
