"""Freeze the new public inventory and make one deterministic Case014 download."""
import argparse,hashlib,json,zipfile
from pathlib import Path

NAME='case014_compensation_unified_publication_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args();case=a.case.resolve();scratch=a.scratch.resolve();scratch.mkdir(parents=True,exist_ok=False)
    relmanifest='publication/provenance/PUBLICATION_MANIFEST.json';mf=case/relmanifest
    files={str(p.relative_to(case)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(case.rglob('*')) if p.is_file() and str(p.relative_to(case))!=relmanifest and '__pycache__' not in p.parts}
    data={'version':1,'case':'014','files':files,'closed_file_set_excluding_self':True,'source_of_truth':'historical records with explicit retained/sanitized path mapping','new_model_forwards':0,'private_artifacts':'excluded; identities and historical receipts only'}
    mf.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    archive=scratch/(NAME+'.zip');prefix=NAME+'/cases/014-quantization-compensation/'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        paths=[case/k for k in files]+[mf]
        for p in sorted(paths):
            info=zipfile.ZipInfo(prefix+str(p.relative_to(case)),date_time=(2026,10,15,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,p.read_bytes())
        info=zipfile.ZipInfo(NAME+'/START_HERE.md',date_time=(2026,10,15,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
        z.writestr(info,'# Case014 unified publication v1\n\n[한국어](cases/014-quantization-compensation/README.ko.md) · [English](cases/014-quantization-compensation/README.md) · [CPU reproduction](cases/014-quantization-compensation/REPRODUCE.md)\n\nThis archive includes the Case014 source/result tree once. Original stage manifests and restored-view mapping are included. Model/K/INT4 weights/large activations/logits/text/tokens and the three old ZIPs are excluded. Full portfolio build needs the complete repository.\n')
    dest=case.parents[1]/'downloads';dest.mkdir(exist_ok=True);published=dest/archive.name
    if published.exists():
        previous=scratch/'previous-download.zip';previous.write_bytes(published.read_bytes())
    published.write_bytes(archive.read_bytes())
    meta={'filename':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),'case_manifest_sha256':sha(mf),'case_files':len(files)+1,'uncompressed_case_bytes':sum(x['bytes'] for x in files.values())+mf.stat().st_size,'archive_version':'v1','historical_run_ids':['local_r1','p1_r2','p1b_r2','p2_r1'],'scope':'CPU-verifiable recorded P0–P2, source, small synthetic fixtures, scalar results and reports; no model/private payload','GPU_inference_during_publication':0}
    (dest/(NAME+'.json')).write_text(json.dumps(meta,indent=2)+'\n');(scratch/'archive.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
