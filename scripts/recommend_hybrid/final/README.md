# Final recommendation release utilities

`verify_release.py` verifies frozen Recommendation evidence and hashes already recorded
at the scientific release. It does not train a model, call Gemini, rerun Panel B, or
change any threshold/configuration.

Development and experiment scripts are intentionally left on branch `Module_recomend`
instead of being promoted into the clean production surface on `main`.
