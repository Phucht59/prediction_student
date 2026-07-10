"""Small, dependency-free reproducibility helpers.

Raw CSV ingestion and split materialisation were retired from this module.  The
final pipeline obtains rows and split membership from PostgreSQL; this module
only retains the hash primitive used by manifests and database provenance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
