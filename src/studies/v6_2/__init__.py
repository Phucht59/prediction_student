"""V6.2 scientific validation for the recommendation layer.

This namespace is evaluation-only.  It must never train or select a prediction
model, open the Future OULAD cohort, or rewrite frozen V5--V6.1 evidence.
"""

from .contract import ARTIFACT_ROOT, REPORT_ROOT, SCHEMA_VERSION

__all__ = ["ARTIFACT_ROOT", "REPORT_ROOT", "SCHEMA_VERSION"]
