from __future__ import annotations

import hashlib

import pytest

from src.recommend_hybrid.checkpoint_authority import resolve_checkpoint_path


def _write(path, value: bytes) -> str:
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def test_resolver_prefers_declared_local_authority(tmp_path):
    local = tmp_path / "local.pt"
    release = tmp_path / "release.pt"
    expected = _write(local, b"local-authority")
    _write(release, b"release-authority")

    result = resolve_checkpoint_path(
        local,
        release,
        expected_sha256=expected,
        structural_validator=lambda path: path.read_bytes() == b"local-authority",
    )

    assert result["resolved_checkpoint_source"] == "local_authority"
    assert result["checkpoint_path"] == str(local)


def test_resolver_falls_back_only_when_local_is_missing(tmp_path):
    local = tmp_path / "missing-local.pt"
    release = tmp_path / "release.pt"
    expected = _write(release, b"release-authority")

    result = resolve_checkpoint_path(
        local,
        release,
        expected_sha256=expected,
        structural_validator=lambda path: path.read_bytes() == b"release-authority",
    )

    assert result["resolved_checkpoint_source"] == "release_lfs_fallback"
    assert result["checkpoint_path"] == str(release)


def test_resolver_fails_closed_on_present_local_sha_mismatch(tmp_path):
    local = tmp_path / "local.pt"
    release = tmp_path / "release.pt"
    _write(local, b"tampered-local")
    _write(release, b"valid-release")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_checkpoint_path(
            local,
            release,
            expected_sha256=hashlib.sha256(b"valid-release").hexdigest(),
            structural_validator=lambda path: True,
        )


def test_resolver_fails_closed_on_structural_mismatch(tmp_path):
    local = tmp_path / "missing-local.pt"
    release = tmp_path / "release.pt"
    expected = _write(release, b"valid-release")

    with pytest.raises(ValueError, match="structural authority mismatch"):
        resolve_checkpoint_path(
            local,
            release,
            expected_sha256=expected,
            structural_validator=lambda path: False,
        )
