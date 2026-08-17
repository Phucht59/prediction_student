# Hybrid Scientific Pipeline (`src/hybrid`)

## 1. Scope and Isolation

`src/hybrid` is the isolated candidate pipeline for binary academic-risk prediction across:
1. **UCI Combined** (Student-Mat and Student-Por concatenated as subject-records with global student grouping).
2. **OULAD** (Cutoff-filtered longitudinal weekly student activity tracking).

### Technical Identity
- **Public Technical Name**: `Hybrid`
- **Model ID**: `hybrid`
- **Planned Architecture**: Parallel Residual CNN + BiLSTM with light context projection and binary classification head.

## 2. Boundaries and Integrity Rules

- **Candidate Status**: Hybrid is an experimental candidate pipeline. It makes no claims of SOTA, universal superiority, or production finality in this phase.
- **Isolation from Frozen Release**: No code in `src/hybrid` may import from or depend on `artifacts/final/` for training. Frozen current results serve as reference and provenance only.
- **Domain Independence**: UCI and OULAD maintain separate data adapters and preprocessing logic due to differing data structures, while sharing the conceptual parallel CNN + BiLSTM core.
- **Recommendation Deferred**: Recommendation policies and intervention generation are strictly out of scope until predictive model benchmarks pass all required gates.

## 3. Planned Target Namespace Architecture

The planned tree to be built incrementally in subsequent phases is:

```text
src/hybrid/
    __init__.py
    contracts.py
    provenance.py

    data/
        __init__.py
        common.py
        splits.py
        preprocessing.py
        uci.py
        oulad.py
        tabular.py

    baselines/
        __init__.py
        registry.py
        tuning.py

    models/
        __init__.py
        components.py
        hybrid.py

    training/
        __init__.py
        trainer.py
        losses.py
        tuning.py

    evaluation/
        __init__.py
        metrics.py
        bootstrap.py
        calibration.py
        sequence_length.py
```

*Note: Directories and modules will only be instantiated during their respective implementation phases to avoid empty placeholder scaffolding.*
