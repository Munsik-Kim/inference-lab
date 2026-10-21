"""Build and verify a local review candidate, with no Git/network/model actions.

The archive contains Case017, five read-only Case016 import dependencies and the
root license. Model/environment/cache binaries are never archive candidates.
Candidate bytes are scanned before copying. A finding stops packaging; this
module does not silently sanitize raw research evidence or alter its hashes.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

from .common import CASE, REPO


UPSTREAM_FILES = (
    "cases/016-kv-agent-bench/followups/nvfp4-technical-r1/src/nvfp4_technical/__init__.py",
    "cases/016-kv-agent-bench/followups/nvfp4-technical-r1/src/nvfp4_technical/worker.py",
    "cases/016-kv-agent-bench/src/kv_agent_bench/__init__.py",
    "cases/016-kv-agent-bench/src/kv_agent_bench/profiler_extension.py",
    "cases/016-kv-agent-bench/src/kv_agent_bench/worker_inspection.py",
)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".svg", ".html", ".css", ".js", ".cjs", ".log", ".bib", ".toml", ".yaml", ".yml", ".partial"}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
FORBIDDEN_PARTS = {".git", ".venv", "venv", "site-packages", "node_modules", "engine_cache", "flashinfer_cache", "snapshots"}
FORBIDDEN_SUFFIXES = {".safetensors", ".pt", ".pth", ".rpi4", ".bin", ".so", ".dll", ".exe", ".zip", ".whl"}
PATTERNS = {
    "personal_linux_home": re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    "personal_windows_home": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9_. -]+[\\/]"),
    "personal_wsl_windows_home": re.compile(r"/mnt/[a-z]/Users/[^/\s]+/"),
    "github_token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})"),
    "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{25,}"),
    "aws_access_key": re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_url": re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            value.update(part)
    return value.hexdigest()


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _scan(path, max_file_bytes):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Only regular nonsymlink candidate files are allowed")
    if path.stat().st_size > max_file_bytes:
        raise ValueError("Candidate exceeds the registered packaging file-size ceiling")
    if path.suffix in BINARY_SUFFIXES:
        return
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            data = stream.read(max_file_bytes + 1)
        if len(data) > max_file_bytes:
            raise ValueError("Decompressed candidate exceeds scan ceiling")
        text = data.decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    for label, pattern in PATTERNS.items():
        if pattern.search(text):
            # Do not echo a token or private path into the public console log.
            raise ValueError("Candidate privacy scan blocked: " + label)


def _candidates(case, repo):
    result, excluded = [], []
    for path in sorted(case.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(repo)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            excluded.append({"path": relative.as_posix(), "reason": "generated Python bytecode"})
            continue
        if any(part in FORBIDDEN_PARTS for part in relative.parts) or path.suffix in FORBIDDEN_SUFFIXES:
            raise ValueError("Forbidden model/environment/archive candidate: " + relative.as_posix())
        if path.suffix not in TEXT_SUFFIXES | BINARY_SUFFIXES | {".gz"} and path.name not in {"LICENSE", "NOTICE"}:
            raise ValueError("Unreviewed candidate file type: " + relative.as_posix())
        if "audit" in relative.parts and "fixtures" in relative.parts:
            excluded.append({"path": relative.as_posix(), "reason": "regenerable CPU replay fixture state"})
            continue
        result.append((path, relative))
    for name in UPSTREAM_FILES:
        result.append((repo / name, Path(name)))
    result.append((repo / "LICENSE", Path("LICENSE")))
    return result, excluded


def verify_archive(archive, extraction_root, *, max_uncompressed_bytes=512 * 1024 * 1024):
    extraction_root = Path(extraction_root)
    extraction_root.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member")
        if sum(info.file_size for info in bundle.infolist()) > max_uncompressed_bytes:
            raise ValueError("Archive exceeds uncompressed verification ceiling")
        for info in bundle.infolist():
            name = PurePosixPath(info.filename)
            if "\\" in info.filename or name.is_absolute() or any(part in ("", ".", "..") for part in name.parts) or re.match(r"^[A-Za-z]:", info.filename):
                raise ValueError("Unsafe archive member path")
            if stat.S_ISLNK(info.external_attr >> 16) or info.is_dir():
                raise ValueError("Symlinks and directory-only archive entries are not expected")
        crc_failure = bundle.testzip()
        if crc_failure:
            raise ValueError("Archive CRC failure")
        for info in bundle.infolist():
            target = extraction_root.joinpath(*PurePosixPath(info.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(bundle.read(info))
    project = extraction_root / "inference-lab"
    manifest_path = project / "REVIEW_CONTENT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {row["path"] for row in manifest["files"]} | {"REVIEW_CONTENT_MANIFEST.json"}
    actual = {path.relative_to(project).as_posix() for path in project.rglob("*") if path.is_file()}
    if expected != actual:
        raise ValueError("Closed archive file set differs from content manifest")
    for row in manifest["files"]:
        path = project / row["path"]
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Extracted file hash or size differs")
    return {"project": project, "members": len(expected), "manifest_files": len(manifest["files"]),
            "CRC": "PASS", "safe_paths": "PASS", "duplicates": "PASS", "symlinks": "PASS", "hashes": "PASS"}


def _cpu_verify(project, output):
    case = project / "cases/017-agent-contract-cost-bench"
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(case / "src"), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    plans = [("unit", ["-m", "unittest", "discover", "-s", str(case / "tests"), "-v"]),
             ("postprocess-unit", ["-m", "unittest", "discover", "-s", str(case / "postprocess/tests"), "-v"]),
             ("validate", ["-m", "agent_contract_cost", "validate", "--output", str(output / "validate")]),
             ("oracle-check", ["-m", "agent_contract_cost", "oracle-check", "--output", str(output / "oracle-check")])]
    stages = []
    for stage in ("P1", "P2"):
        root = case / "results" / (stage.lower() + "_r1")
        if list((root / "cells").glob("*.json")) or list((root / "cells").glob("*.json.gz")):
            plans.append((stage + "-audit", ["-m", "agent_contract_cost", "audit", "--stage", stage, "--output", str(output / (stage + "-audit"))]))
            if (root / "audit/audited_rows.json").exists():
                plans.append((stage + "-analyze", ["-m", "agent_contract_cost", "analyze", "--stage", stage, "--output", str(output / (stage + "-analyze"))]))
            stages.append(stage)
            plans.append((stage + "-inputs", [str(case / "postprocess/verify_recorded_inputs.py"), "--stage", stage,
                                              "--output", str(output / (stage + "-inputs"))]))
            if (root / "completion.json").exists() and json.loads((root / "completion.json").read_text())["status"] == "COMPLETED":
                plans.append((stage + "-annotations", [str(case / "postprocess/inspect_contract_boundaries.py"), "--stage", stage,
                                                      "--output", str(output / (stage + "-annotations"))]))
    if stages:
        plans.append(("terminal-independence", [str(case / "postprocess/check_terminal_independence.py"),
                                              "--output", str(output / "terminal-independence")]))
        plans.append(("display-report", ["-m", "agent_contract_cost.reporting", "--output", str(output / "display-report")]))
    commands = []
    for label, args in plans:
        result = subprocess.run([sys.executable, "-B", *args], cwd=project, env=env,
                                capture_output=True, text=True, timeout=600)
        (output / (label + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
        commands.append({"label": label, "exit_code": result.returncode})
        if result.returncode:
            _json(output / "commands.json", commands)
            raise ValueError("Extracted public CPU command failed: " + label)
        if label.endswith("-audit"):
            receipt = json.loads((output / label / "audit_receipt.json").read_text())
            if not receipt["audit_pass"]:
                raise ValueError("Extracted raw replay reported audit mismatches: " + label)
            saved = case / "results" / (label[:2].lower() + "_r1") / "audit/audited_rows.json"
            if saved.exists() and json.loads(saved.read_text()) != json.loads((output / label / "audited_rows.json").read_text()):
                raise ValueError("Fresh replay rows differ from retained audited rows")
        if label.endswith("-analyze"):
            saved = case / "results" / (label[:2].lower() + "_r1") / "analysis/summary.json"
            if saved.exists() and json.loads(saved.read_text()) != json.loads((output / label / "summary.json").read_text()):
                raise ValueError("Fresh summary differs from retained analysis")
        if label == "display-report":
            for name in ("REPORT.md", "REPORT.ko.md", "README.md", "README.ko.md"):
                saved = case / name
                if saved.exists() and saved.read_bytes() != (output / label / name).read_bytes():
                    raise ValueError("Regenerated report differs from retained display")
    receipt = {"commands": commands, "stored_trace_stages": stages, "gpu_inference": "NOT_RUN",
               "new_model_download": False, "scope": "CPU unit/contracts/oracle and available saved-episode replay/analysis"}
    _json(output / "commands.json", receipt)
    return receipt


def build_review(output_root, *, case=CASE, repo=REPO, cpu=True,
                 archive_name="Case017_agent_contract_cost_review_v1.zip",
                 max_file_bytes=16 * 1024 * 1024, max_archive_bytes=20 * 1024 * 1024):
    case, repo, output = Path(case).resolve(), Path(repo).resolve(), Path(output_root).resolve()
    if output == case or case in output.parents:
        raise ValueError("Review outputs must be outside the research case")
    if Path(archive_name).name != archive_name or not archive_name.endswith(".zip"):
        raise ValueError("Archive name must be one ZIP filename")
    output.mkdir(parents=True, exist_ok=False)
    project = output / "payload/inference-lab"
    project.mkdir(parents=True)
    candidates, excluded = _candidates(case, repo)
    copied = []
    for source, relative in candidates:
        _scan(source, max_file_bytes)
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append({"path": relative.as_posix(), "bytes": destination.stat().st_size, "sha256": sha(destination),
                       "source_bytes_preserved": sha(source) == sha(destination)})
    scope = {"kind": "local_review_candidate", "case": "017", "source_case_files": len(candidates) - 6,
             "readonly_upstream_source_files": 5, "root_license": True, "excluded": excluded,
             "model_snapshot_weights_KV_environment_credentials": "NOT_INCLUDED",
             "gpu_reexecution": "NOT_RUN", "CPU": "unit/contracts/oracle, stored raw replay and analysis when present",
             "cross_case_reference_links": "Some historical documentation links require the full public repository; runtime dependency source allowlist is included.",
             "privacy_scan": "private-home paths and credential signatures scanned; fail closed, no silent redaction",
             "content_manifest_excludes_itself": True, "zip_hash_receipt_outside_archive": True}
    scope["archive_compression"] = "ZIP_LZMA, lossless Python stdlib zipfile; use Python zipfile/7-Zip to extract"
    _json(project / "BUNDLE_SCOPE.json", scope)
    copied.append({"path": "BUNDLE_SCOPE.json", "bytes": (project / "BUNDLE_SCOPE.json").stat().st_size, "sha256": sha(project / "BUNDLE_SCOPE.json")})
    manifest = {"schema": 1, "files": sorted(copied, key=lambda row: row["path"]),
                "exclusions": excluded, "self_hash": "EXCLUDED_TO_AVOID_RECURSION"}
    _json(project / "REVIEW_CONTENT_MANIFEST.json", manifest)
    archive = output / archive_name
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_LZMA) as bundle:
        for path in sorted(project.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(project.parent).as_posix())
    if archive.stat().st_size > max_archive_bytes:
        raise ValueError("Completed archive exceeds review-size ceiling; retained for coverage review")
    verified = verify_archive(archive, output / "extracted")
    cpu_receipt = _cpu_verify(verified["project"], output / "cpu-verification") if cpu else {"status": "NOT_RUN_REQUESTED"}
    receipt = {"archive_name": archive.name, "bytes": archive.stat().st_size, "sha256": sha(archive),
               "archive_verification": {key: value for key, value in verified.items() if key != "project"},
               "CPU_verification": cpu_receipt, "new_model_requests": 0,
               "source_model_environment_changed": False, "Git_remote_operations": "NOT_RUN"}
    _json(output / "review_receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a local Case017 review candidate and recheck its extracted CPU paths")
    parser.add_argument("--output", required=True, help="New external review directory; never inside Case017")
    parser.add_argument("--name", default="Case017_agent_contract_cost_review_v1.zip")
    parser.add_argument("--no-cpu", action="store_true", help="Explicitly leave extracted CPU validation NOT_RUN")
    args = parser.parse_args(argv)
    print(json.dumps(build_review(args.output, archive_name=args.name, cpu=not args.no_cpu), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
