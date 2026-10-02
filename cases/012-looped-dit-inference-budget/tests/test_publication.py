"""Historical restoration rejects invalid overlays and never overwrites output."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('case012_restore', ROOT / 'publication/restore_original.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root):
    p = root / 'publication/original_docs'
    p.mkdir(parents=True)
    (p / 'README.md.txt').write_bytes(b'original introduction\n')
    (root / 'README.md').write_bytes(b'new introduction\n')
    (root / 'scalar.json').write_bytes(b'{"value": 7}\n')
    blobs = {'README.md': b'original introduction\n', 'scalar.json': b'{"value": 7}\n'}
    inventory = {k: {'bytes': len(v), 'sha256': hashlib.sha256(v).hexdigest()} for k, v in blobs.items()}
    (root / 'publication/original_inventory.json').write_text(json.dumps({'files': inventory}))
    return blobs


def test_restore_uses_original_document_and_preserves_scalar(tmp_path):
    root = tmp_path / 'current'
    expected = fixture(root)
    output = tmp_path / 'restored'
    assert module.restore(root, output)['original_files'] == 2
    for name, blob in expected.items():
        assert (output / name).read_bytes() == blob
    assert (root / 'README.md').read_bytes() == b'new introduction\n'


def test_restore_rejects_corrupt_original_before_writing(tmp_path):
    root = tmp_path / 'current'
    fixture(root)
    (root / 'publication/original_docs/README.md.txt').write_bytes(b'changed original\n')
    output = tmp_path / 'restored'
    with pytest.raises(ValueError, match='identity mismatch'):
        module.restore(root, output)
    assert not output.exists()
    assert not (tmp_path / 'restored.partial').exists()


def test_restore_rejects_existing_output_without_overwrite(tmp_path):
    root = tmp_path / 'current'
    fixture(root)
    output = tmp_path / 'existing'
    output.mkdir()
    (output / 'user-file').write_text('preserve')
    with pytest.raises(ValueError, match='new directory'):
        module.restore(root, output)
    assert (output / 'user-file').read_text() == 'preserve'
