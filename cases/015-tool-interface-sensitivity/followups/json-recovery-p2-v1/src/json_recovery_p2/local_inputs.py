"""Read-only CPU file hashing. Never instantiate or deserialize a model."""
import sys
import time
from pathlib import Path
from .io import read, write
from .upstream import CASE


def check(runtime_file, output):
    from interface_sensitivity.common import sha
    from interface_sensitivity.model_adapter import verify_saved_artifacts
    started = time.monotonic(); runtime = read(runtime_file)
    identity = read(CASE / 'provenance/model_identity.json'); model = Path(runtime['model']['path'])
    files = []
    missing = [name for name, e in identity['files'].items() if not (model / name).is_file()]
    if missing:
        raise ValueError('missing local snapshot files: ' + ', '.join(missing))
    for name, entry in identity['files'].items():
        path = model / name
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError('local snapshot SHA/bytes mismatch: ' + name)
        files.append({'label': 'registered_snapshot/' + name, **entry})
    artifacts = verify_saved_artifacts(runtime, CASE)
    if 'torch' in sys.modules or 'transformers' in sys.modules:
        raise RuntimeError('CPU hashing imported a forbidden framework')
    value = {'status': 'PASS', 'evidence_kind': 'CPU_READ_ONLY_FILE_HASH', 'model_revision': identity['revision'],
        'local_identity_only_not_official_external_digest': True, 'snapshot_files': files,
        'full_stored_weight_shards_hashed': True, 'saved_artifacts': artifacts,
        'runtime_configuration_sha256': sha(runtime_file), 'model_objects_instantiated': 0, 'payloads_deserialized': 0,
        'actual_model_module_coverage': 'NOT_RUN; metadata252 verified, future loaded-module comparison still required',
        'D_origin_required': False, 'artifact_writes': 0, 'model_forwards': 0,
        'torch_imported': False, 'transformers_imported': False, 'CPU_wall_seconds': time.monotonic() - started}
    write(output, value)
    return value

