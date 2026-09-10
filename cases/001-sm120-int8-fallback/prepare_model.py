"""Download the tested public snapshot, or verify an existing local copy."""
import argparse
import hashlib
import json
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent


def verify_model(path):
    manifest = json.loads((BUNDLE / "model-files.json").read_text())
    checked = []
    for item in manifest["files"]:
        file = path / item["file"]
        if file.stat().st_size != item["size"]:
            raise ValueError(f"File size mismatch: {item['file']}")
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"SHA256 mismatch: {item['file']}")
        checked.append(item["file"])
    return {"model_id": manifest["model_id"], "revision": manifest["revision"],
            "files_verified": checked, "path": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cache-dir", type=Path, help="Download pinned files into this cache")
    group.add_argument("--model-dir", type=Path, help="Verify only; no network or downloads")
    args = parser.parse_args()
    if args.model_dir:
        path = args.model_dir.resolve()
    else:
        from huggingface_hub import snapshot_download
        manifest = json.loads((BUNDLE / "model-files.json").read_text())
        path = Path(snapshot_download(
            repo_id=manifest["model_id"], revision=manifest["revision"],
            allow_patterns=[x["file"] for x in manifest["files"]],
            cache_dir=str(args.cache_dir.resolve()), token=False, max_workers=2,
        ))
    print(json.dumps(verify_model(path), indent=2))


if __name__ == "__main__":
    main()
