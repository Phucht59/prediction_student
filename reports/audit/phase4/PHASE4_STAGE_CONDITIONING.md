# Phase 4 — Stage Conditioning

**EXPLICIT_STAGE_CONDITIONING_REDUNDANT.** The authoritative aggregate branch already consumes `progress_fraction`, `observed_week_count`, `weeks_remaining`, `assessment_available_fraction`. These variables contain only observation availability known at prediction time. Adding B1 would duplicate equivalent information, so it was validly skipped.
