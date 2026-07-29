"""Checkpoint compatibility shim for the former OULAD module path.

New code must import :mod:`src.pipelines.oulad`. Existing immutable joblib
checkpoints retain this module name in their pickle metadata.
"""

from src.pipelines.oulad import *  # noqa: F403
from src.pipelines.oulad import Bundle, StageData, _DeepPreprocessor

__all__ = ["Bundle", "StageData", "_DeepPreprocessor"]
