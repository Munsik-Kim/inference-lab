"""Run the unchanged P2 verifier on an exact, temporary original-manifest view. CPU only."""
import argparse,pathlib,tempfile,json,hashlib,shutil,subprocess,sys,os
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tmp-parent',type=pathlib.Path,required=True);p.add_argument('--receipt',type=pathlib.Path);a=p.parse_args()
 case=ROOT.parents[1];inv=json.loads((ROOT/'provenance/baseline_inventory.json').read_text());a.tmp_parent.mkdir(parents=True,exist_ok=True)
 if a.tmp_parent.resolve().is_relative_to(case.resolve()):raise ValueError('Temporary view must be outside the Case')
 with tempfile.TemporaryDirectory(prefix='historical-scope-',dir=a.tmp_parent) as d:
  dest=pathlib.Path(d)
  for rel,e in inv['files'].items():
   src=case/rel;b=src.read_bytes()
   if len(b)!=e['bytes'] or hashlib.sha256(b).hexdigest()!=e['sha256']:raise ValueError('Original identity mismatch: '+rel)
   q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
  cmd=[sys.executable,'-B',str(dest/'src/audit_v2.py'),'verify','--root',str(dest)]
  r=subprocess.run(cmd,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
  receipt={'copied_files':len(inv['files']),'original_verifier_modified':False,'expected_coverage_modified':False,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'temporary_copy_removed_on_exit':True}
 if a.receipt:a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt,indent=2));sys.exit(r.returncode)
if __name__=='__main__':main()
