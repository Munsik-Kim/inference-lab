"""Package the combined, verified Case006 tree; never overwrite a historical ZIP."""
from __future__ import annotations
import argparse, json, stat, zipfile
from pathlib import Path
from verify_publication import verify, safe, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args(); case = args.case.resolve()
    for path in (args.zip, args.record):
        if path.exists() or path.resolve().is_relative_to(case):
            parser.error('Archive and record must be new files outside the case')
    checked = verify(case)
    files = sorted(p for p in case.rglob('*') if p.is_file())
    with zipfile.ZipFile(args.zip, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            name = '006-attention-decision-stability/' + path.relative_to(case).as_posix()
            if not safe(name): raise ValueError('Unsafe archive path')
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(args.zip) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) != len(files) or archive.testzip() is not None:
            raise ValueError('Archive count/CRC/duplicate failure')
        for member in archive.infolist():
            if not safe(member.filename) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Unsafe member')
            rel = member.filename.split('/', 1)[1]
            if archive.read(member) != (case / rel).read_bytes():
                raise ValueError('Repository/archive byte mismatch')
    result = {'filename': args.zip.name, 'files': len(files), 'bytes': args.zip.stat().st_size,
              'sha256': sha(args.zip), 'crc': 'PASS', 'safe_paths': 'PASS', 'symlinks': 0,
              'duplicates': 0, 'byte_correspondence': 'PASS', 'publication_verification': checked,
              'publication_manifest_sha256': sha(case / 'PUBLICATION_SHA256SUMS')}
    with args.record.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
