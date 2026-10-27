"""Semantic tamper and output guards on a disposable public aggregate copy."""
import argparse,csv,hashlib,json,shutil,subprocess,sys,tempfile
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='p5-public-') as tmp:
        t=Path(tmp);r=t/'candidate';shutil.copytree(a.root,r);script=r/'source/public_cpu.py'
        def call(name,expect=0):
            out=t/name;result=subprocess.run([sys.executable,'-B',str(script),'--root',str(r),'--output',str(out),'--quick'],capture_output=True,text=True)
            assert (result.returncode==0)==(expect==0),(name,result.stderr)
            return out
        call('valid');call('valid',2)
        f=r/'summary/candidate_fit.csv';saved=f.read_bytes();f.write_bytes(saved+b'\n');call('hash_tamper',2);f.write_bytes(saved)
        # Rehash a semantic mutation: hash verification alone must not certify the numbers.
        rows=list(csv.DictReader(f.open()));fields=list(rows[0]);row=next(x for x in rows if x['scope']=='ALL_VALID' and x['method']=='P');row['A_only']=str(int(row['A_only'])+1)
        with f.open('w',newline='') as stream:w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
        manifest=r/'PUBLIC_MANIFEST.json';original_manifest=manifest.read_bytes();m=json.loads(original_manifest);m['files']['summary/candidate_fit.csv']={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()};manifest.write_text(json.dumps(m));call('semantic_tamper',2)
        f.write_bytes(saved);manifest.write_bytes(original_manifest);(r/'unexpected.txt').write_text('unexpected');call('extra_file',2);(r/'unexpected.txt').unlink()
        result=subprocess.run([sys.executable,'-B',str(script),'--help'],capture_output=True,text=True);assert result.returncode==0
        receipt={'cases':['valid public quick','existing output rejected','raw byte mutation rejected','rehashed numeric mutation rejected','unexpected file rejected','help works'],'GPU_calls':0,'scope':'Public summaries and synthetic primitive checker, not private GPU reproduction'}
        (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)
if __name__=='__main__':main()
