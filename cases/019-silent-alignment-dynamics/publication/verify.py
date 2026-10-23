"""Closed small-payload manifest and bounded exact-byte evidence restoration.

No numerical SDK, network, pickle, implicit downloads or original workspace.
"""
from pathlib import Path,PurePosixPath
from contextlib import ExitStack
import hashlib,json,tarfile,re,lzma

CASE=Path(__file__).resolve().parents[1]
HASH=re.compile(r'^[0-9a-f]{64}$')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def safe(name):
    p=PurePosixPath(name)
    if not isinstance(name,str) or not name or p.is_absolute() or any(x in ('','.','..') for x in name.split('/')) or any(x in name for x in '\\:\0\r\n'):
        raise ValueError('UNSAFE_RELATIVE_PATH')
    return p
def file_rows(root,exclude=()):
    return [{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
            for p in sorted(Path(root).rglob('*')) if p.is_file() and p.relative_to(root).as_posix() not in exclude]

def verify(case=CASE):
    case=Path(case);manifest=read(case/'publication/PUBLIC_MANIFEST.json')
    rows=manifest['files'];paths=[r['path'] for r in rows]
    actual={p.relative_to(case).as_posix() for p in case.rglob('*') if p.is_file()}
    if len(paths)!=len(set(paths)) or actual!=set(paths)|{'publication/PUBLIC_MANIFEST.json'}:raise ValueError('PUBLIC_COVERAGE')
    if hashlib.sha256(canonical(rows)).hexdigest()!=manifest['public_bundle_sha256']:raise ValueError('PUBLIC_MANIFEST_HASH')
    for r in rows:
        p=case/safe(r['path'])
        if p.is_symlink() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('PUBLIC_BYTES: '+r['path'])
    mapping=read(case/'publication/provenance/original_mapping.json');objects,logical=validate_mapping(mapping)
    distribution=read(case/'publication/distribution.json')
    validate_distribution(mapping,distribution)
    # Public scientific source must have exactly the historic frozen bytes.
    for phase in ['p0','p1','p2']:
        frozen=read(case/f'publication/identity/{phase}/execution_freeze.json')
        if hashlib.sha256(canonical(frozen['files'])).hexdigest()!=frozen['bundle_sha256']:raise ValueError('SCIENCE_BUNDLE_IDENTITY')
        for r in frozen['files']:
            p=case/safe(r['path'])
            if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('FROZEN_SOURCE_CHANGED')
    return {'status':'PASS','public_files':len(actual),'public_bytes':sum(r['bytes'] for r in rows),
            'public_bundle_sha256':manifest['public_bundle_sha256'],'logical_evidence_files':len(logical),
            'science_identity':'historical phase bundles, not the Git base','full_replay':'REQUIRES_EXPLICIT_EVIDENCE','self_excluded':True,
            'release_url':distribution['asset']['url']}

def validate_distribution(mapping,distribution):
    """Keep archive bytes frozen while declaring an explicit Release location.

    Location metadata is not proof of upload or a network availability check.
    No URL is fetched by verification, import or restoration.
    """
    asset=distribution['asset'];archived=mapping['archive']
    location={'url','tag','upload_status'}
    if {k:v for k,v in asset.items() if k not in location}!={k:v for k,v in archived.items() if k not in location}:
        raise ValueError('EVIDENCE_IDENTITY_CHANGED')
    tag=asset.get('tag')
    if not isinstance(tag,str) or not re.fullmatch(r'case019-evidence-v\d+\.\d+\.\d+',tag):
        raise ValueError('CASE019_RELEASE_TAG')
    expected='https://github.com/Munsik-Kim/inference-lab/releases/download/'+tag+'/'+asset['filename']
    if asset.get('url')!=expected:raise ValueError('CASE019_RELEASE_URL')

def validate_mapping(mapping):
    objects={};logical=set()
    for r in mapping['files']:
        safe(r['path']);safe(r['member'])
        if r['path'] in logical or 'alias' in r or 'alias_of' in r:raise ValueError('DUPLICATE_PATH_OR_ALIAS_CHAIN')
        logical.add(r['path']);h=r['sha256']
        if not HASH.fullmatch(h) or r['member']!='objects/'+h or type(r['bytes']) is not int or r['bytes']<0:
            raise ValueError('OBJECT_IDENTITY')
        pair=(r['bytes'],h);previous=objects.setdefault(r['member'],pair)
        if previous!=pair:raise ValueError('ALIAS_IDENTITY_CONFLICT')
        if r['retention']!='EXACT_BYTES' or r['transforms']:raise ValueError('UNDECLARED_TRANSFORMATION')
    a=mapping['archive']
    if len(logical)!=a['logical_files'] or len(objects)!=a['unique_objects'] or sum(r['bytes'] for r in mapping['files'])!=a['restore_bytes']:
        raise ValueError('EVIDENCE_COVERAGE_COUNTS')
    return objects,logical

def restore(case,evidence,output):
    """Stream each regular object into its explicitly mapped regular files.

    A fresh output root is mandatory. No extractall, links, alias chains, or
    follow-up network operations. XZ CRC and TAR termination are consumed.
    """
    case,evidence=Path(case),Path(evidence);output=Path(output).resolve()
    if output.exists() or output.is_relative_to(case.resolve()):raise ValueError('FRESH_EXTERNAL_RESTORE_REQUIRED')
    mapping=read(case/'publication/provenance/original_mapping.json');objects,logical=validate_mapping(mapping);a=mapping['archive']
    if evidence.is_symlink() or evidence.stat().st_size!=a['bytes'] or sha(evidence)!=a['sha256']:raise ValueError('EVIDENCE_ARCHIVE_IDENTITY')
    with evidence.open('rb') as f:
        if f.read(6)!=b'\xfd7zXZ\x00':raise ValueError('XZ_SIGNATURE')
    # Consume to EOF explicitly so XZ CRC/truncation is checked independently
    # of tar's end-of-archive zero blocks. Bound the decompressed tar stream.
    inflated=0;limit=a['unique_uncompressed_bytes']+len(objects)*1024+10240
    with lzma.open(evidence,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            inflated+=len(chunk)
            if inflated>limit:raise ValueError('XZ_EXPANSION_BOUND')
    destinations={}
    for r in mapping['files']:destinations.setdefault(r['member'],[]).append(r['path'])
    output.mkdir(parents=True,exist_ok=False);seen=set();restored=0
    with tarfile.open(evidence,mode='r|xz') as tar:
        for member in tar:
            safe(member.name)
            if not member.isfile() or member.name in seen or member.name not in objects:raise ValueError('UNSAFE_OR_UNEXPECTED_MEMBER')
            size,h=objects[member.name]
            if member.size!=size or size>a['max_object_bytes']:raise ValueError('MEMBER_SIZE')
            with ExitStack() as stack:
                sinks=[]
                for relative in destinations[member.name]:
                    p=output/safe(relative)
                    if not p.resolve().is_relative_to(output):raise ValueError('RESTORE_ESCAPE')
                    p.parent.mkdir(parents=True,exist_ok=True);sinks.append(stack.enter_context(p.open('xb')))
                source=tar.extractfile(member);digest=hashlib.sha256();count=0
                while True:
                    b=source.read(min(1024*1024,size-count+1))
                    if not b:break
                    count+=len(b)
                    if count>size:raise ValueError('MEMBER_OVERFLOW')
                    digest.update(b)
                    for sink in sinks:sink.write(b)
                if count!=size or digest.hexdigest()!=h:raise ValueError('MEMBER_HASH')
            seen.add(member.name);restored+=len(destinations[member.name])
    if seen!=set(objects) or restored!=len(logical):raise ValueError('ARCHIVE_MEMBER_COVERAGE')
    for r in mapping['files']:
        p=output/r['path']
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('RESTORED_ALIAS_HASH')
    return {'status':'PASS','logical_files':restored,'unique_objects':len(seen),'bytes_restored':a['restore_bytes'],
            'archive_sha256':a['sha256'],'output':str(output),'exact_bytes':True}

def verify_restored(case,directory):
    """Reuse one complete restored root without making another large copy."""
    directory=Path(directory).resolve();mapping=read(Path(case)/'publication/provenance/original_mapping.json')
    objects,logical=validate_mapping(mapping)
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual!=logical:raise ValueError('RESTORED_DIRECTORY_COVERAGE')
    for r in mapping['files']:
        p=directory/safe(r['path'])
        if p.is_symlink() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('RESTORED_DIRECTORY_BYTES')
    return {'status':'PASS','logical_files':len(logical),'unique_objects':len(objects),'bytes_restored':mapping['archive']['restore_bytes'],
            'output':str(directory),'exact_bytes':True,'source':'existing exact restored directory; archive CRC was checked when first restored'}
