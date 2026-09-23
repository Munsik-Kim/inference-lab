"""Standard-library exact inventory and identity checks for a publication tree.

Create: python scripts/verify_publication.py --root CASE --create
Verify: python scripts/verify_publication.py --root CASE

The default manifest is publication-manifest.json relative to CASE. It is the
only excluded file. Do not place a checksum of this manifest inside CASE:
that would introduce a self-referential hash cycle. Scientific record replay
is a separate audit_records.py check; this tool verifies retained file bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


SCHEMA = "case010-exact-publication-inventory-v1"


class PublicationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise PublicationError(message)


def safe_name(name):
    require(isinstance(name,str) and bool(name) and "\\" not in name and "\x00" not in name,
            "unsafe publication path")
    path=PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and ":" not in name and path.as_posix()==name
            and name != ".", "unsafe or noncanonical publication path")
    return name


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def json_file(path, label):
    try:
        return json.loads(path.read_bytes())
    except (OSError,UnicodeDecodeError,json.JSONDecodeError) as exc:
        raise PublicationError(f"{label}: unreadable JSON") from exc


def inventory(root, manifest_name):
    root=Path(root)
    manifest_name=safe_name(manifest_name)
    require(root.is_dir(),"publication root is not a directory")
    files=[]
    folded=set()
    for path in sorted(root.rglob("*")):
        name=safe_name(path.relative_to(root).as_posix())
        require(not path.is_symlink(),f"{name}: symlinks are not portable publication artifacts")
        if path.is_dir():
            continue
        require(path.is_file(),f"{name}: unsupported filesystem entry")
        if name == manifest_name:
            continue
        require(name.casefold() not in folded,f"{name}: duplicate path under case-insensitive extraction")
        folded.add(name.casefold())
        raw=path.read_bytes()
        files.append(dict(path=name,bytes=len(raw),sha256=sha(raw)))
    return files


def identity_links(root, files, manifest_name):
    """Check hash sidecars and explicit learned index/correctness file references."""
    root=Path(root)
    available={item["path"] for item in files}
    identities=[]
    def target_from(parent,reference):
        reference=safe_name(reference)
        target=(PurePosixPath(parent).parent/reference).as_posix()
        return safe_name(target)
    def check(source,target,expected,kind="file_sha256"):
        safe_name(source);safe_name(target)
        require(target != manifest_name,"manifest self-hash is forbidden; keep detached checksums outside publication root")
        require(target in available,f"{source}: referenced artifact missing: {target}")
        require(isinstance(expected,str) and re.fullmatch(r"[0-9a-f]{64}",expected),f"{source}: invalid hash identity")
        raw=(root/target).read_bytes()
        actual=sha(raw) if kind == "file_sha256" else sha(canonical(json_file(root/target,target)))
        require(actual==expected,f"{source}: referenced hash mismatch for {target}")
        identities.append(dict(source=source,target=target,kind=kind,sha256=expected))
    for item in files:
        name=item["path"]
        path=PurePosixPath(name)
        if path.suffix == ".sha256":
            try:
                lines=(root/name).read_text(encoding="utf-8").strip().splitlines()
            except (OSError,UnicodeDecodeError) as exc:
                raise PublicationError(f"{name}: unreadable checksum sidecar") from exc
            require(len(lines)==1 and bool(lines[0].split()),f"{name}: expected one checksum record")
            expected=lines[0].split()[0]
            candidates=[path.with_suffix("").as_posix(),path.with_suffix(".json").as_posix()]
            candidates=list(dict.fromkeys(candidate for candidate in candidates if candidate in available or candidate==manifest_name))
            require(len(candidates)==1,f"{name}: checksum target missing or ambiguous")
            target=candidates[0]
            require(target != manifest_name,"manifest self-hash is forbidden; keep detached checksums outside publication root")
            raw=(root/target).read_bytes()
            kind="file_sha256"
            if sha(raw) != expected and target.endswith(".json"):
                kind="canonical_json_sha256"
            check(name,target,expected,kind)
        if path.suffix != ".json":
            continue
        record=json_file(root/name,name)
        if not isinstance(record,dict):
            continue
        if path.name == "index.json" and isinstance(record.get("splits"),dict):
            for records in record["splits"].values():
                require(isinstance(records,dict),f"{name}: invalid split reference map")
                for entry in records.values():
                    require(isinstance(entry,dict) and "file" in entry and "sha256" in entry,f"{name}: incomplete result reference")
                    check(name,target_from(name,entry["file"]),entry["sha256"])
        artifact=record.get("correctness_artifact")
        if artifact is not None:
            require(isinstance(artifact,dict) and "file" in artifact and "sha256" in artifact,f"{name}: incomplete correctness reference")
            check(name,target_from(name,artifact["file"]),artifact["sha256"])
        if path.name == "manifest.json" and record.get("schema") in (
                "case010-learned-evaluation-v1", "case010-learned-evaluation-v2-canonical-token-torch"):
            for field,target_name in {"protocol_sha256":"protocol.json","runtime_sha256":"evaluation_runtime.json",
                                      "calibration_sha256":"calibration.npz","token_table_config_sha256":"token_coefficients.json",
                                      "token_table_data_sha256":"token_coefficients.bin"}.items():
                require(field in record,f"{name}: required identity missing: {field}")
                check(name,target_from(name,target_name),record[field])
    return identities


def build_manifest(root,manifest_name="publication-manifest.json"):
    manifest_name=safe_name(manifest_name)
    files=inventory(root,manifest_name)
    return dict(schema=SCHEMA,root_scope=".",manifest_file=manifest_name,
                excluded_files=[manifest_name],self_hash_policy="manifest is excluded; no manifest hash is stored inside it",
                file_count=len(files),total_file_bytes=sum(item["bytes"] for item in files),files=files,
                identities=identity_links(root,files,manifest_name))


def verify_manifest(root,manifest_name="publication-manifest.json"):
    manifest_name=safe_name(manifest_name)
    root=Path(root)
    manifest=json_file(root/manifest_name,manifest_name)
    require(isinstance(manifest,dict) and manifest.get("schema")==SCHEMA,"unsupported publication manifest")
    require(not any(key in manifest for key in ("sha256","self_sha256","manifest_sha256")),"manifest self-hash field is forbidden")
    require(manifest.get("manifest_file")==manifest_name and manifest.get("excluded_files")==[manifest_name],
            "manifest exclusion policy mismatch")
    expected=manifest.get("files")
    require(isinstance(expected,list),"manifest file list missing")
    seen=set()
    for item in expected:
        require(isinstance(item,dict),"invalid file entry")
        name=safe_name(item.get("path"))
        require(name != manifest_name,"manifest must not inventory itself")
        require(name.casefold() not in seen,f"{name}: duplicate manifest path")
        seen.add(name.casefold())
        require(type(item.get("bytes")) is int and item["bytes"]>=0,"invalid file byte count")
        require(isinstance(item.get("sha256"),str) and re.fullmatch(r"[0-9a-f]{64}",item["sha256"]),"invalid file hash")
    actual=inventory(root,manifest_name)
    expected_map={entry["path"]:entry for entry in expected}
    actual_map={entry["path"]:entry for entry in actual}
    require(set(expected_map)==set(actual_map),
            f"inventory mismatch; missing={sorted(set(expected_map)-set(actual_map))}; extra={sorted(set(actual_map)-set(expected_map))}")
    for name in expected_map:
        require(expected_map[name]==actual_map[name],f"{name}: retained file bytes/hash changed")
    require(manifest.get("file_count")==len(actual) and manifest.get("total_file_bytes")==sum(item["bytes"] for item in actual),
            "manifest totals mismatch")
    identities=identity_links(root,actual,manifest_name)
    require(manifest.get("identities")==identities,"retained identity reference set changed")
    return dict(status="PASS",file_count=len(actual),total_file_bytes=manifest["total_file_bytes"],
                identity_references=len(identities),manifest_sha256=sha((root/manifest_name).read_bytes()))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--manifest",default="publication-manifest.json",help="safe path relative to --root")
    parser.add_argument("--create",action="store_true",help="write a new manifest exclusively, then verify it")
    args=parser.parse_args(argv)
    safe_name(args.manifest)
    if args.create:
        target=args.root/args.manifest
        require(not target.exists(),"manifest already exists; creation never overwrites it")
        manifest=build_manifest(args.root,args.manifest)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("x",encoding="utf-8") as output:
            json.dump(manifest,output,indent=2,allow_nan=False)
            output.write("\n")
    print(json.dumps(verify_manifest(args.root,args.manifest),sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PublicationError as error:
        print(json.dumps(dict(status="FAIL",error=str(error))),file=sys.stderr)
        raise SystemExit(2)
