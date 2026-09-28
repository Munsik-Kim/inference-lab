"""Exact two-tensor FP32 head patches, tied to a required base checkpoint hash."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np
import torch


SCHEMA = "diova-case011-final-linear-patch-v1"
SPECS = {"mlp.2.weight": (6, 192), "mlp.2.bias": (6,)}
HASH_FIELDS = ("base_checkpoint_sha256", "fit_input_sha256", "dev_input_sha256", "code_sha256")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("patch metadata must be an object")
    for key in HASH_FIELDS:
        if not isinstance(metadata.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", metadata[key]):
            raise ValueError(f"metadata {key} must be a lowercase SHA256")
    if not isinstance(metadata.get("feature_boundary"), str) or not metadata["feature_boundary"]:
        raise ValueError("metadata feature_boundary is required")
    if not isinstance(metadata.get("solver"), dict):
        raise ValueError("metadata solver log is required")
    if metadata.get("regularization") not in (1e-4, 1e-2, 1.0) or isinstance(metadata.get("regularization"), bool):
        raise ValueError("metadata regularization must be one of the declared lambdas")
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("patch metadata must be finite JSON") from exc


def _payload(value, name):
    t = torch.as_tensor(value).detach().cpu()
    if t.dtype != torch.float32 or tuple(t.shape) != SPECS[name]:
        raise ValueError(f"{name} requires float32 shape {SPECS[name]}")
    if not torch.isfinite(t).all():
        raise ValueError(f"{name} is nonfinite")
    return t.contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")


def _json_without_duplicates(text):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError(f"duplicate JSON key {k}")
            result[k] = v
        return result
    def no_constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")
    return json.loads(text, object_pairs_hook=unique, parse_constant=no_constant)


def save_patch(directory, weight, bias, metadata):
    """Create a new directory via validated sibling .partial + atomic rename.

    The patch requires the original checkpoint; it is not a standalone model.
    No claim of power-loss durability is made for the directory rename.
    """
    destination = Path(directory)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite patch destination: {destination.name}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError("patch parent must be an existing real directory")
    _metadata(metadata)
    payloads = {name: _payload(t, name) for name, t in zip(SPECS, (weight, bias))}
    partial = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    partial.mkdir()
    entries = []
    for name, data in payloads.items():
        filename = name + ".f32le"
        (partial / filename).write_bytes(data)
        entries.append({"name": name, "shape": list(SPECS[name]), "dtype": "float32",
                        "endianness": "little", "order": "C", "file": filename,
                        "bytes": len(data), "sha256": _digest(data)})
    manifest = {"schema": SCHEMA, "requires_base_checkpoint": True,
                "metadata": metadata, "tensors": entries,
                "payload_bytes": sum(map(len, payloads.values())),
                "trainable_parameters": 1158, "additional_recurrent_state_bytes": 0}
    (partial / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                       sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    load_patch(partial, expected_base_sha256=metadata["base_checkpoint_sha256"])
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("patch destination appeared before commit; .partial retained")
    os.rename(partial, destination)
    return {"manifest_sha256": _digest((destination / "manifest.json").read_bytes()),
            "payload_bytes": manifest["payload_bytes"],
            "manifest_bytes": (destination / "manifest.json").stat().st_size,
            "total_file_bytes": sum(p.stat().st_size for p in destination.iterdir()),
            "tensor_names": list(SPECS)}


def load_patch(directory, *, expected_base_sha256):
    path = Path(directory)
    if not path.is_dir() or path.is_symlink():
        raise ValueError("patch must be a real directory")
    if not isinstance(expected_base_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_base_sha256):
        raise ValueError("expected base checkpoint SHA256 is required")
    inventory = {p.name for p in path.iterdir()}
    wanted = {"manifest.json", *(name + ".f32le" for name in SPECS)}
    if inventory != wanted or any(not p.is_file() or p.is_symlink() for p in path.iterdir()):
        raise ValueError("patch inventory must contain exactly manifest and the two tensor files")
    manifest = _json_without_duplicates((path / "manifest.json").read_text(encoding="utf-8"))
    expected_keys = {"schema", "requires_base_checkpoint", "metadata", "tensors", "payload_bytes",
                     "trainable_parameters", "additional_recurrent_state_bytes"}
    if not isinstance(manifest, dict) or set(manifest) != expected_keys or manifest.get("schema") != SCHEMA:
        raise ValueError("unknown patch schema or manifest keys")
    if manifest["requires_base_checkpoint"] is not True or manifest["trainable_parameters"] != 1158 or manifest["additional_recurrent_state_bytes"] != 0:
        raise ValueError("patch model/parameter/storage contract mismatch")
    _metadata(manifest["metadata"])
    if manifest["metadata"]["base_checkpoint_sha256"] != expected_base_sha256:
        raise ValueError("patch base checkpoint SHA256 mismatch")
    entries = manifest["tensors"]
    if not isinstance(entries, list) or len(entries) != 2 or {e.get("name") for e in entries if isinstance(e, dict)} != set(SPECS):
        raise ValueError("patch must specify exactly the final weight and bias")
    tensors = {}
    for entry in entries:
        name = entry["name"]
        expected_file = name + ".f32le"
        if (set(entry) != {"name", "shape", "dtype", "endianness", "order", "file", "bytes", "sha256"}
                or entry["file"] != expected_file or entry["shape"] != list(SPECS[name])
                or entry["dtype"] != "float32" or entry["endianness"] != "little" or entry["order"] != "C"):
            raise ValueError(f"tensor schema/shape/dtype/path mismatch for {name}")
        data = (path / expected_file).read_bytes()
        nbytes = 4 * int(np.prod(SPECS[name]))
        if len(data) != nbytes or entry["bytes"] != nbytes or _digest(data) != entry["sha256"]:
            raise ValueError(f"tensor length/checksum mismatch for {name}")
        array = np.frombuffer(data, dtype="<f4").reshape(SPECS[name]).astype(np.float32, copy=True)
        tensor = torch.from_numpy(array)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"nonfinite stored tensor {name}")
        tensors[name] = tensor
    if manifest["payload_bytes"] != sum(t.numel() * t.element_size() for t in tensors.values()):
        raise ValueError("payload byte total mismatch")
    return {"manifest": manifest, "tensors": tensors,
            "manifest_sha256": _digest((path / "manifest.json").read_bytes())}


def apply_patch(model, patch, *, expected_base_sha256):
    """Copy values into existing Parameters; reject every non-final tensor change."""
    if isinstance(patch, (str, Path)):
        patch = load_patch(patch, expected_base_sha256=expected_base_sha256)
    if (not isinstance(patch, dict) or not isinstance(patch.get("manifest"), dict)
            or patch["manifest"].get("schema") != SCHEMA
            or patch["manifest"].get("metadata", {}).get("base_checkpoint_sha256") != expected_base_sha256
            or set(patch.get("tensors", {})) != set(SPECS)):
        raise ValueError("loaded patch identity or tensor set mismatch")
    parameters = dict(model.named_parameters())
    _metadata(patch["manifest"]["metadata"])
    entries = patch["manifest"].get("tensors", [])
    if not isinstance(entries, list) or len(entries) != 2 or {e.get("name") for e in entries if isinstance(e, dict)} != set(SPECS):
        raise ValueError("loaded patch tensor manifest mismatch")
    indexed = {e["name"]: e for e in entries}
    for name, shape in SPECS.items():
        if name not in parameters or tuple(parameters[name].shape) != shape or parameters[name].dtype != torch.float32:
            raise ValueError(f"model {name} does not match FP32 {shape}")
        data = _payload(patch["tensors"][name], name)
        if _digest(data) != indexed[name].get("sha256") or len(data) != indexed[name].get("bytes"):
            raise ValueError(f"loaded patch tensor checksum mismatch: {name}")
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    ids = {name: id(p) for name, p in parameters.items()}
    with torch.no_grad():
        for name in SPECS:
            parameters[name].copy_(patch["tensors"][name].to(parameters[name].device))
    after = model.state_dict()
    changed = [name for name in before if not torch.equal(before[name], after[name].detach().cpu())]
    if set(changed) - set(SPECS):
        raise RuntimeError("non-final model tensor changed while applying a head patch")
    if ids != {name: id(p) for name, p in model.named_parameters()}:
        raise RuntimeError("Parameter identities changed while copying patch values")
    return {"changed_tensors": sorted(changed), "allowed_tensors": sorted(SPECS),
            "other_tensor_changes": 0, "parameter_objects_preserved": True,
            "patch_requires_original_checkpoint": True}
