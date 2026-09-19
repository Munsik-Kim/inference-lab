"""Local recorded freezes. These are not external preregistration."""
from datetime import datetime, timezone
from .common import ROOT, CASE, read, write, sha, require


def freeze_eval(track, dependencies):
    require(track in ('Q','R'),'Unknown track')
    build=read(CASE/'configs/build_freeze.json')
    for name,value in build['input_files'].items():
        require(sha(CASE/name)==value,'Input changed after build freeze')
    result={'track':track,'frozen_at':datetime.now(timezone.utc).isoformat(),
       'design_sha256':sha(CASE/'configs/build_freeze.json'),'dependencies':dependencies,
       'input_files':build['input_files'],
       'code':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'tools/modelpack').glob('*.py'))
               if p.name not in ({'r_study.py'} if track=='Q' else {'q_runtime.py','quantized.py','qartifact.py'})},
       'heldout_candidate_outputs_observed':False,'scope':'local freeze after DEV; original test inputs not used for fitting'}
    write(CASE/f'configs/{track}_evaluation_freeze.json',result)
    return result


def verify_eval(track):
    f=read(CASE/f'configs/{track}_evaluation_freeze.json')
    require(f['track']==track and f['heldout_candidate_outputs_observed'] is False,'Bad evaluation freeze')
    require(sha(CASE/'configs/build_freeze.json')==f['design_sha256'],'Changed design')
    for name,value in f['input_files'].items():require(sha(CASE/name)==value,'Changed evaluation inputs')
    for name,value in f['code'].items():require(sha(ROOT/name)==value,'Changed frozen implementation')
    return f
