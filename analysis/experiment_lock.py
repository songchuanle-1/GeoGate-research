"""Build and verify frozen hashes for critical experiment inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCHEMA = "geogate-experiment-lock-v1"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lock_key(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_lock(
    root: Path,
    experiment: str,
    paths: list[Path],
    metadata: dict | None = None,
) -> dict:
    root = Path(root).resolve()
    files = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        key = lock_key(path, root)
        files[key] = {
            "bytes": path.stat().st_size,
            "sha256": digest_file(path),
        }
    return {
        "schema": SCHEMA,
        "experiment": experiment,
        "files": files,
        "metadata": metadata or {},
    }


def verify_lock(lock_path: Path, root: Path) -> dict:
    lock_path = Path(lock_path)
    root = Path(root).resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != SCHEMA:
        raise ValueError("Unexpected experiment-lock schema")
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Experiment lock contains no files")

    issues = []
    for key, expected in sorted(files.items()):
        path = Path(key)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            issues.append({"path": key, "issue": "missing"})
            continue
        actual_bytes = path.stat().st_size
        actual_sha256 = digest_file(path)
        if actual_bytes != int(expected["bytes"]):
            issues.append({
                "path": key,
                "issue": "size_mismatch",
                "expected": int(expected["bytes"]),
                "actual": actual_bytes,
            })
        if actual_sha256 != str(expected["sha256"]):
            issues.append({
                "path": key,
                "issue": "sha256_mismatch",
                "expected": str(expected["sha256"]),
                "actual": actual_sha256,
            })

    return {
        "schema": "geogate-experiment-lock-audit-v1",
        "state": "passed" if not issues else "failed",
        "experiment": str(lock.get("experiment", "unknown")),
        "lock": str(lock_path.resolve()),
        "lock_sha256": digest_file(lock_path),
        "metadata": lock.get("metadata", {}),
        "files_checked": len(files),
        "issues": issues,
    }


def require_lock(
    lock_path: Path,
    root: Path,
    expected_experiment: str,
) -> dict:
    """Verify a frozen launch lock or stop before model/output work begins."""
    lock_path = Path(lock_path)
    if not lock_path.is_file():
        raise RuntimeError(f"Experiment lock is missing: {lock_path}")
    audit = verify_lock(lock_path, root)
    if audit["state"] != "passed":
        raise RuntimeError(
            "Experiment lock verification failed: "
            + json.dumps(audit["issues"][:5], sort_keys=True)
        )
    if audit["experiment"] != expected_experiment:
        raise RuntimeError(
            "Experiment lock names "
            f"{audit['experiment']!r}, expected {expected_experiment!r}"
        )
    return audit


def lock_reference(audit: dict) -> dict:
    return {
        "schema": str(audit["schema"]),
        "experiment": str(audit["experiment"]),
        "path": str(audit["lock"]),
        "sha256": str(audit["lock_sha256"]),
        "files_checked": int(audit["files_checked"]),
        "metadata": dict(audit.get("metadata", {})),
    }
