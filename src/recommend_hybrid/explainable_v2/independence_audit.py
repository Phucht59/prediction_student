"""Independent LLM Source Independence Auditor."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def compute_source_independence_audit(df_accepted: pd.DataFrame) -> dict:
    """Compute verified independent source counts across provider, model, endpoint, and run identities."""
    if df_accepted.empty:
        return {
            "unique_provider_count": 0,
            "unique_model_family_count": 0,
            "unique_endpoint_count": 0,
            "unique_generation_run_count": 0,
            "verified_independent_source_count": 0,
            "same_provider_dependency": True,
            "same_model_family_dependency": True,
            "status": "BLOCKED_NO_VERIFIED_RAW_RESPONSES",
        }

    provs = df_accepted["provider"].nunique()
    models = df_accepted["model_name"].nunique() if "model_name" in df_accepted.columns else 0
    endpoints = df_accepted["reviewer_configuration_id"].nunique() if "reviewer_configuration_id" in df_accepted.columns else 0
    runs = df_accepted["request_id"].nunique() if "request_id" in df_accepted.columns else 0

    indep_count = min(provs, models)

    return {
        "unique_provider_count": provs,
        "unique_model_family_count": models,
        "unique_endpoint_count": endpoints,
        "unique_generation_run_count": runs,
        "verified_independent_source_count": indep_count,
        "same_provider_dependency": provs < 2,
        "same_model_family_dependency": models < 2,
        "status": "PASS" if indep_count >= 2 else "BLOCKED_INSUFFICIENT_INDEPENDENT_SOURCES",
    }
