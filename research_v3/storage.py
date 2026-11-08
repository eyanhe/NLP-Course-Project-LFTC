"""Atomic, bounded writes and content identity for preparation artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def bounded_path(path: Path, root: Path) -> Path:
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Output escapes its declared root: {path}')
    return path


def atomic_write_json(path: Path, payload: dict[str, Any], root: Path) -> None:
    path = bounded_path(path, root)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))
