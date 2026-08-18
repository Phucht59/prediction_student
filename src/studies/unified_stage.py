"""Checkpoint compatibility shim for the former UCI module path.

New code must import :mod:`src.pipelines.uci`. Existing immutable joblib
checkpoints retain this module name in their pickle metadata.
"""

from src.pipelines.uci import *  # noqa: F403
