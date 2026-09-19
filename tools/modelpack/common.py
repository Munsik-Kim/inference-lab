"""Small, explicit storage and integrity primitives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "cases/008-build-reconstruct-reload"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path: Path, value: Any) -> None:
    require(not path.exists(), "Refusing to overwrite " + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(blob)


def new_external(path: Path) -> Path:
    path = path.resolve()
    require(not path.is_relative_to(ROOT), "Artifacts and execution outputs must be outside repository")
    require(not path.exists(), "Output already exists")
    path.mkdir(parents=True)
    return path


def safe_name(name: str) -> str:
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and ".." not in p.parts and "\\" not in name
            and ":" not in name and str(p) == name, "Unsafe artifact path")
    return name


def inventory(root: Path, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Artifact symlinks are not allowed")
        if path.is_file():
            name = safe_name(path.relative_to(root).as_posix())
            if name not in exclude:
                result[name] = sha(path)
    return result


def checksums(root: Path) -> None:
    records = inventory(root, ("checksums.sha256",))
    with (root / "checksums.sha256").open("x") as stream:
        stream.write("".join(f"{value}  {name}\n" for name, value in records.items()))


def verify_checksums(root: Path) -> dict[str, str]:
    entries = {}
    for line in (root / "checksums.sha256").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        safe_name(name)
        require(name != "checksums.sha256" and name not in entries, "Duplicate/self checksum")
        require(len(checksum) == 64 and all(c in "0123456789abcdef" for c in checksum), "Bad SHA256")
        entries[name] = checksum
    require(entries == inventory(root, ("checksums.sha256",)), "Artifact inventory/hash mismatch")
    return entries
