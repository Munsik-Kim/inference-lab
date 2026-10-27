"""Explicit local private-evidence replay. Never loads Torch or starts a GPU worker."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root.resolve();out=a.output.resolve()
    if out.exists() or out.is_relative_to(r):raise ValueError('NEW_EXTERNAL_OUTPUT_REQUIRED')
    out.mkdir(parents=True);results=[]
    for script,reference,dest in [('p5a_v2.py','outputs/p5a','A'),('analyze_B.py','outputs/B','B'),('analyze_C.py','outputs/C','C')]:
        subprocess.run([sys.executable,'-B',str(r/'source'/script),'--root',str(r),'--output',str(out/dest)],check=True)
        for old in sorted((r/reference).iterdir()):
            if not old.is_file():continue
            new=out/dest/old.name;assert old.read_bytes()==new.read_bytes(),str(old)
            results.append({'relative_path':dest+'/'+old.name,'sha256':digest(new),'bytes':new.stat().st_size,'exact_bytes':True})
    assert 'torch' not in sys.modules
    (out/'replay_receipt.json').write_text(json.dumps({'scope':'Same-machine saved private data replay with the frozen analysis implementations, exact scalar cross-checks included; not external review or new GPU inference','files':results,'GPU_calls':0,'Torch_imported':False},indent=2)+'\n')
    print(json.dumps({'output':str(out),'compared_files':len(results),'GPU_calls':0}))
if __name__=='__main__':main()
