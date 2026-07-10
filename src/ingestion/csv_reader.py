"""The only production wrapper around pandas CSV parsing.

Training and inference must use PostgreSQL loaders; this wrapper exists for the
one-time seed ingestion and legacy split compatibility tests only.
"""

from pathlib import Path

import pandas as pd


def read_csv(path: str | Path, *, sep: str = ",") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)
