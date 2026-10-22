"""Run immutable independent endpoint code in a separate CPU-only phase process."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

class CPUOnly:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','vllm','transformers','xgrammar','flashinfer'}:
            raise RuntimeError('CPU publication replay forbids model SDK import: '+fullname)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--view',type=Path,required=True);p.add_argument('--phase',choices=['P0','P1','P2'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sys.meta_path.insert(0,CPUOnly());t=time.perf_counter();cpu=time.process_time()
    case=a.view/'case';frozen=a.view/(a.phase.lower()+'-frozen')
    sys.path.insert(0,str(frozen/'src'))
    from agent_state_recovery.audit import audit_episode
    from agent_state_recovery.common import read_json, digest
    run=case/'results'/dict(P0='p0_r2',P1='p1_r5',P2='p2_r1')[a.phase]
    manifest=read_json(run/'run_manifest.json')
    if a.phase!='P0':
        freeze=read_json(run/'execution_freeze.json')
        expected_own={'P1':'e19f15776ee4e74a37849e9fa6293e37a3d520bfd0221eb8b3545556f0e9af1d',
                      'P2':'46b2c1dcd90fd368e4a83afff1f5b06954b5e4e699bc856f8c0589ac24b351bd'}[a.phase]
        if digest(freeze['own']['files'])!=expected_own or freeze['own']['working_source_sha256']!=expected_own:
            raise ValueError('Historical own-source bundle hash differs')
        mapping=read_json(Path(__file__).parent/'provenance/original_mapping.json')
        mapped={r['path']:r for r in mapping['files']}
        for item in freeze['own']['files']:
            key=a.phase.lower()+'-frozen/'+item['path'];m=mapped[key];f=frozen/item['path']
            if (m['original_sha256'],m['original_bytes'])!=(item['sha256'],item['bytes']):
                raise ValueError('Frozen file mapping differs from original source bundle')
            if hashlib.sha256(f.read_bytes()).hexdigest()!=m['sha256']:raise ValueError('Restored source file differs')
            if f.suffix=='.py' and m['transforms']:raise ValueError('Frozen Python science must be exact original bytes')
    entries=manifest.get('episodes',[])
    rows=[]
    for entry in entries:
        f=run/entry['path'];raw=f.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Raw bytes differ')
        row=json.loads(raw)
        task=read_json((run/'inputs'/(row['task_id']+'.json')) if a.phase!='P0' else frozen/'data/dev'/(row['task_id']+'.json'))
        result=audit_episode(task,row)
        result.pop('reconstructed_terminal_snapshot')
        rows.append({**result,'task_id':row['task_id'],'family':row['family'],'kv':row.get('kv'),
                     'policy':row['policy'],'phase':a.phase})
    expected=dict(P0=26,P1=24,P2=96)[a.phase]
    if len(rows)!=expected or len({r['episode_id'] for r in rows})!=expected:raise ValueError('Phase coverage differs')
    if a.phase!='P0':
        freeze=read_json(run/'execution_freeze.json');initial=read_json(run/'initial_inputs.json')
        if digest(initial)!=freeze['initial_inputs_sha256']:raise ValueError('Initial inputs differ')
        for entry in entries:
            row=read_json(run/entry['path'])
            if (row['execution_source_sha256']!=freeze['execution_source_sha256'] or row['engine_schema']!=freeze['grammar_schema']
                    or row['requests'][0]['input_token_ids']!=initial[row['task_id']]['token_ids']):
                raise ValueError('Source/schema/all-four initial token identity differs')
    report={'status':'PASS','phase':a.phase,'episodes':len(rows),'endpoint_differences':0,'rows':rows,
            'cost':{'wall_seconds':time.perf_counter()-t,'CPU_seconds':time.process_time()-cpu,
                    'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
            'new_model_requests':0,'shared':['strict parser','serialization','public builder','wire rendering'],
            'independent':['ReferenceState transitions','protection','freshness','continuations','SSE assembly','budgets'],
            'retokenization':'NOT_RUN; original tokenizer receipts are separately retained'}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(a.phase+': '+str(len(rows))+' independent endpoints PASS')

if __name__=='__main__':main()
