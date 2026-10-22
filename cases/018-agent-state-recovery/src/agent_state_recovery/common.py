"""Serialization and owned-output helpers. No SDK, subprocess or network imports."""
import hashlib
import json
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parents[2]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def fresh_dir(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError("symlink output root is forbidden")
    path.mkdir(parents=True, exist_ok=False)
    return path


def strict_equal(left, right):
    return canonical(left) == canonical(right)


def source_manifest():
    files = sorted(p for folder in ("src", "tests", "specs", "data/dev")
                   for p in (CASE_ROOT / folder).rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    rows = [{"path": str(p.relative_to(CASE_ROOT)), "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    return {"files": rows, "working_source_sha256": digest(rows),
            "identity_kind": "UNCOMMITTED_WORKING_SOURCE_BUNDLE",
            "base_commit": read_json(CASE_ROOT / "specs/protocol.json")["base_commit"]}
