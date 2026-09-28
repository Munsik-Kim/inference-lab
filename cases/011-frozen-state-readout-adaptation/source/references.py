"""Read-only, namespaced access to the exact Case 010 v2 implementation.

The legacy code uses top-level ``codec`` imports. Temporary aliases exist only
while its unchanged modules load; a pre-existing ``codec`` module and sys.path
are restored. Classes retain unique module identities after that boundary.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

CASE010_NAME = "010-ckda-finite-precision-memory-horizon"
SNAPSHOT_MANIFEST_SHA256 = "9fdd342a9d8da6b216fb06e2458ac1ba79d6fe4b56c13f7511df7346d49d436e"
_LOCK = threading.RLock()
_CACHE = {}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def default_case010_root():
    return Path(__file__).resolve().parents[2] / CASE010_NAME


def _package(name, folder):
    module = ModuleType(name)
    module.__path__ = [str(folder)]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _load(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_reference_files(case010_root=None):
    root = Path(case010_root or default_case010_root()).resolve(strict=True)
    raw = (root / "provenance/snapshot_manifest.json").read_bytes()
    if _digest(raw) != SNAPSHOT_MANIFEST_SHA256:
        raise ValueError("Case010 snapshot manifest identity differs from the frozen integration")
    entries = json.loads(raw)["files"]
    selected = [item for item in entries if item["path"].startswith((
        "versions/v2/source/", "versions/v2/inputs/v1_calibration/",
        "versions/v2/inputs/codecs/"))]
    if not selected:
        raise ValueError("Missing Case010 source/input identity entries")
    for item in selected:
        path = root / item["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Unsafe Case010 reference path")
        data = path.read_bytes()
        if len(data) != item["bytes"] or _digest(data) != item["sha256"]:
            raise ValueError("Case010 source/input changed: " + item["path"])
    return root, {item["path"]: item["sha256"] for item in selected}


def load_references(case010_root=None):
    root = Path(case010_root or default_case010_root()).resolve(strict=True)
    with _LOCK:
        if root in _CACHE:
            return _CACHE[root]
        root, hashes = verify_reference_files(root)
        source = root / "versions/v2/source"
        reference = source / "v1_reference"
        name = "_diova_case011_case010_" + _digest(str(root).encode())[:16]
        old_path = list(sys.path)
        old_codec = {key: value for key, value in sys.modules.items()
                     if key == "codec" or key.startswith("codec.")}
        try:
            for key in old_codec:
                del sys.modules[key]
            _package(name, source)
            codec = _load(name + ".codec", reference / "codec/__init__.py", package=True)
            for short in ("packed", "online", "groups", "survival", "learned"):
                qualified = name + ".codec." + short
                if qualified not in sys.modules:
                    _load(qualified, reference / "codec" / (short + ".py"))
                setattr(codec, short, sys.modules[qualified])
            sys.modules["codec"] = codec
            for short in ("packed", "online", "groups", "survival", "learned"):
                sys.modules["codec." + short] = getattr(codec, short)
            ref = _load(name + ".reference", source / "reference.py")
            online = _load(name + ".online_v2", source / "online_v2.py")
            common = _load(name + ".common", source / "common.py")
            evaluation = _load(name + ".evaluation", source / "evaluation.py")
            result = SimpleNamespace(root=root, v2_root=root / "versions/v2",
                v1=ref.v1, learned=ref.learned, groups=ref.groups,
                packed=ref.packed, online_v2=online, common=common,
                evaluation=evaluation, source_identity=hashes,
                snapshot_manifest_sha256=SNAPSHOT_MANIFEST_SHA256)
        finally:
            sys.path[:] = old_path
            for key in list(sys.modules):
                if key == "codec" or key.startswith("codec."):
                    del sys.modules[key]
            sys.modules.update(old_codec)
        _CACHE[root] = result
        return result


def load_verified_model(upstream, checkpoint, seed, case010_root=None):
    """Verify the original checkpoint SHA and exact retained FP32 token table."""
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("Expected one of the three original model seeds")
    ref = load_references(case010_root)
    return ref.common.load_model(upstream, checkpoint, seed)


def load_storage(storage, seed=0, case010_root=None):
    if storage not in ("NATIVE_FP32", "UNIFORM_8"):
        raise ValueError("Case011 permits only frozen NATIVE_FP32 and UNIFORM_8")
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("Expected original model seed 0, 1, or 2")
    ref = load_references(case010_root)
    folder = ref.v2_root / "inputs/codecs" / ("seed" + str(seed))
    receipt = json.loads((folder / "selection.json").read_text())
    item = receipt["codec_files"][storage]
    config = (folder / item["config"]).read_bytes()
    basis = (folder / item["basis"]).read_bytes()
    if _digest(config) != item["config_sha256"] or _digest(basis) != item["basis_sha256"]:
        raise ValueError("Frozen storage identity mismatch")
    return ref.online_v2.OnlineAdapter.from_shared(config, basis)
