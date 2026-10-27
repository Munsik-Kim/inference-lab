"""Replay saved evidence on CPU only into a new external output."""
import argparse,json,subprocess,sys
from pathlib import Path
from common import read,write,identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--label',required=True);a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    if out.exists() or out.is_relative_to(root):raise ValueError('OUTPUT_MUST_BE_NEW_AND_EXTERNAL')
    if not a.label.replace('-','').isalnum():raise ValueError('LABEL')
    out.mkdir(parents=True,exist_ok=False)
    py=read(root/'references/locations.json')['cpu_python'];runs=[]
    def run(suffix,script,args,timeout,freeze=None):
        command=[sys.executable,'-B',str(root/'source/supervise.py'),'--root',str(root),'--kind','CPU','--label',a.label+'-'+suffix,'--timeout',str(timeout)]
        if freeze:command+=['--freeze',str(root/'records'/freeze)]
        command+=['--',py,'-B',str(root/'source'/script)]+list(map(str,args));subprocess.run(command,check=True,cwd=root);runs.append(command)
    run('A','p4a_v2.py',['--root',root,'--output',out/'A'],260,'freeze_AB_v2.json')
    run('B','p4b.py',['--root',root,'--output',out/'B'],30,'freeze_AB.json')
    run('C','p4c_analyze.py',['--root',root,'--output',out/'C'],90,'freeze_C_final.json')
    run('auditB','audit_B.py',['--root',root,'--summary',out/'B','--output',out/'auditB'],30)
    run('figures','present_v2.py',['--a',out/'A','--b',out/'B','--c',out/'C','--output',out/'figures'],30)
    comparisons=[]
    for old,new in [('outputs/p4a_v2','A'),('outputs/p4b_v1','B'),('outputs/p4c_v1','C')]:
        for f in (root/old).glob('*.csv'):
            dest=out/new/f.name
            if identity(f)!=identity(dest):raise ValueError('CSV_REPLAY_MISMATCH:'+f.name)
            comparisons.append({'original':str(f),'replay':str(dest),'identity':identity(f)})
        facts=next((root/old).glob('*_facts.json'))
        if read(facts)!=read(out/new/facts.name):raise ValueError('FACTS_REPLAY_MISMATCH')
    write(out/'replay_receipt.json',{'status':'COMPLETE','GPU_calls':0,'commands':runs,'CSV_exact_byte_matches':comparisons,'facts_equal':True,'scope':'Local saved raw replay, not new inference or external-machine reproduction'})
    print(json.dumps({'status':'COMPLETE','CSV_files':len(comparisons),'output':str(out)}))
if __name__=='__main__':main()
