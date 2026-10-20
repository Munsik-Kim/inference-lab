import argparse
import pathlib
import tempfile
from .common import load,save,digest
from .fixture_generator import oracle,tools,short_messages

def main():
    p=argparse.ArgumentParser(description='Case016: CPU contracts/replay and explicitly configured vLLM runs')
    sub=p.add_subparsers(dest='cmd',required=True)
    v=sub.add_parser('validate');v.add_argument('--case',type=pathlib.Path,required=True)
    o=sub.add_parser('oracle-check');o.add_argument('--case',type=pathlib.Path,required=True);o.add_argument('--output',type=pathlib.Path,required=True)
    a=sub.add_parser('audit');a.add_argument('--tasks',required=True);a.add_argument('--results',required=True);a.add_argument('--output',required=True)
    t=sub.add_parser('prepare-tokens');t.add_argument('--case',type=pathlib.Path,required=True);t.add_argument('--snapshot',required=True);t.add_argument('--output',type=pathlib.Path,required=True)
    r=sub.add_parser('run');r.add_argument('--case',type=pathlib.Path,required=True);r.add_argument('--runtime',required=True);r.add_argument('--stage',choices=['p0','p1'],required=True)
    re=sub.add_parser('render');re.add_argument('--case',type=pathlib.Path,required=True)
    args=p.parse_args()
    if args.cmd=='validate':
        tasks=load(args.case/'data/tasks.json');protocol=load(args.case/'specs/protocol.json')
        assert len(tasks)==68 and len({t['task_id'] for t in tasks})==68
        assert [len([t for t in tasks if t['split']==s]) for s in ('DEV','SMOKE','EVAL')]==[12,8,48]
        for t in tasks:assert t['oracle']==oracle(t)
        assert tools(8)==tools(32)[:8]
        print({'status':'PASS','tasks':len(tasks),'registered_p1_episodes':96,'protocol_sha256':digest(protocol),'model_loaded':False})
    elif args.cmd=='oracle-check':
        from .tools import Fixture
        from .scorer import assess
        from .audit import independent_endpoint
        if args.output.exists():raise ValueError('fresh output required')
        args.output.mkdir(parents=True);rows=[]
        for t in load(args.case/'data/tasks.json'):
            g=oracle(t);fx=Fixture(t,args.output/'fixtures'/t['task_id']);seen=[]
            calls=[('read_source',{'source_id':g['source_id']}),('calculate_amount',{k:g[k] for k in ('quantity','unit_price_minor','discount_percent')}),('update_state',{k:g[k] for k in ('record_id','total_minor','source_version','status')}),('validate_state',{'record_id':g['record_id']})]
            for i,(n,a) in enumerate(calls):
                e=fx.execute(n,a,i,0,tuple(seen));seen.append(e['event_id'])
            final={k:g[k] for k in ('record_id','total_minor','source_version','status')};final['validated']=True;rs=[{'available_result_event_ids':seen}]
            s=assess(t,fx.events,rs,final,fx.state,fx.version,'FINAL');i=independent_endpoint(t,fx.events,final,fx.state,fx.version,'FINAL',rs)
            assert s['VERIFIED_WORKFLOW_SUCCESS'] and i
            rows.append({'task_id':t['task_id'],'status':'PASS','evidence_kind':'synthetic_contract_test','model_calls':0,'tool_calls':4})
        save(args.output/'oracle.json',{'rows':rows,'passed':len(rows),'model_forwards':0});print({'oracle_fixtures_passed':len(rows),'evidence_kind':'synthetic_contract_test'})
    elif args.cmd=='audit':
        from .audit import audit_directory
        receipt=audit_directory(args.tasks,args.results,args.output);print({k:v for k,v in receipt.items() if k!='rows'})
    elif args.cmd=='prepare-tokens':
        from .tokenization import get_tokenizer,prepare_input,render,token_hash
        tok=get_tokenizer(args.snapshot)
        if args.output.exists():raise ValueError('fresh prepared root required')
        args.output.mkdir(parents=True);manifest=[]
        for task in load(args.case/'data/tasks.json'):
            for menu in (8,32):
                for length in ((8192,16384) if task['split']!='SMOKE' else (0,)):
                    if length:p=prepare_input(tok,task,menu,length)
                    else:
                        messages=short_messages(task);ids=tok.encode(render(tok,messages,tools(menu)),add_special_tokens=False)
                        p={'task_id':task['task_id'],'menu':menu,'length':len(ids),'messages':messages,'tools':tools(menu),'input_token_ids':ids,'input_token_sha256':token_hash(ids),'surface_hash':digest({'messages':messages,'tools':tools(menu)})}
                    name=f'{task["task_id"]}_m{menu}_l{length}.json';save(args.output/name,p)
                    manifest.append({k:v for k,v in p.items() if k not in ('messages','tools','input_token_ids')})
        save(args.case/'data/token_manifest.json',manifest);print({'prepared_inputs':len(manifest),'model_forwards':0})
    elif args.cmd=='run':
        from .workload_runner import run
        run(args.case,args.runtime,args.stage)
    elif args.cmd=='render':
        from .report import render_demo
        render_demo(args.case)

if __name__=='__main__':main()
