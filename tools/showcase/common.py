"""Small read-only utilities shared by the presentation tools."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
C6 = 'cases/006-attention-decision-stability'
C7 = 'cases/007-interaction-aware-mlp-pruning'
SUP = C6 + '/supplemental/readout-ties-v1'

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result

def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))

def script_json(value) -> str:
    return json_text(value).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')

def safe_name(name: str) -> str:
    p = PurePosixPath(name)
    require(bool(name) and str(p) == name and not p.is_absolute()
            and all(s not in ('.', '..') for s in p.parts)
            and not any(c in name for c in '\\:\n\r\0'), 'Unsafe relative path')
    return name

def source_manifest(root: Path) -> dict:
    return read(root/'presentation/source_manifest.json')

def checked_read(root: Path, name: str, manifest: dict):
    safe_name(name)
    path = root/name
    require(path.is_file() and not path.is_symlink(), 'Missing/unsafe source: ' + name)
    require(name in manifest['files'] and sha(path) == manifest['files'][name], 'Changed source: ' + name)
    return read(path)

def new_output(root: Path, output: Path) -> Path:
    output = output.resolve()
    require(not output.is_relative_to(root.resolve()), 'Output must be outside the repository')
    require(not output.exists(), 'Output already exists; choose a new path')
    return output
