"""Immutable JSON records and explicit output boundaries."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite JSON literal: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def write_new(path: Path, value: Any) -> None:
    """Atomic create, never overwrite an existing successful or failed record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)  # Fails atomically if destination already exists.
    finally:
        Path(tmp).unlink(missing_ok=True)


def external_directory(path: Path, case: Path) -> Path:
    path = path.resolve()
    if path.is_relative_to(case.resolve()) or path.exists():
        raise ValueError("Choose a NEW output directory outside the source case")
    path.mkdir(parents=True)
    return path


class Ledger:
    """One immutable file per explicit cell ID; no silent duplicate merges."""
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def path(self, cell_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", cell_id):
            raise ValueError("Unsafe cell ID")
        return self.root / (cell_id + ".json")

    def get(self, cell_id: str, identity: dict) -> dict | None:
        path = self.path(cell_id)
        if not path.exists():
            return None
        record = load(path)
        if record.get("identity") != identity:
            raise ValueError("Existing cell has a different input/config/source identity")
        return record

    def put(self, cell_id: str, identity: dict, payload: dict) -> None:
        write_new(self.path(cell_id), {"identity": identity, "payload": payload})
