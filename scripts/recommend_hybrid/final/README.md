# Final recommendation scripts

Only release-safe verification utilities live here.

`verify_release.py` verifies that the canonical final evidence is byte-identical
to the frozen scientific lineage and checks the already-recorded final metrics.
It does **not** call Gemini, rerun Panel B, retrain models, or tune thresholds.

Historical development, annotation, audit, and freeze scripts remain under
`scripts/recommend_hybrid/explainable_v2/` as provenance. They are not the
production execution surface.
