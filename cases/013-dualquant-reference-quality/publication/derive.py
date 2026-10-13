"""Source-linked display projection from the independent recorded-results auditor."""
from pathlib import Path
import argparse, hashlib, importlib.util, json
CASE=Path(__file__).resolve().parents[1]

def derive(case=CASE):
    case=Path(case)
    spec=importlib.util.spec_from_file_location('case013_scalar_audit',case/'analysis/audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.audit(case)
    manifest=module.read(case/'records/experiment_manifest.json')
    result['models']={key:{'id':row['id'],'revision':row['revision']} for key,row in manifest['models'].items()}
    result['method_labels']={'BF16':'Original BF16','RTN':'Group64 RTN W4A16',
                             'D16':'Dual-scale INT4 reference (D16)','D4':'D16 + activation INT4 (D4 variant)'}
    result['interpretation']='Two-model reference quality / small paired DEV; not established serving benefit'
    result['comparisons']=[]
    for model in ('06b','4b'):
        rows={row['method']:row for row in result['quality'] if row['model']==model}
        a,b=rows['RTN'],rows['D16']
        result['comparisons'].append({'model':model,'baseline':'RTN','candidate':'D16',
            'weight_mse_change_percent':100*(b['weight_normalized_mse']/a['weight_normalized_mse']-1),
            'ppl_change_percent':100*(b['ppl']/a['ppl']-1),
            'artifact_extra_percent':100*(b['artifact_payload_bytes']/a['artifact_payload_bytes']-1)})
    paths=['analysis/audit.py','publication/derive.py','provenance/export_map.json',
           'provenance/publication_identity.json','records/experiment_manifest.json','records/reference_contract.json',
           'records/quality_06b.csv','records/quality_4b.csv','records/paired_dev_summary.json',
           'records/diagnostics/packed_precision_aligned_03.json',
           *[f'records/artifact_metadata/{key}-{method}.json' for key in ('06b','4b') for method in ('RTN','D16')]]
    result['source_identity']={p:hashlib.sha256((case/p).read_bytes()).hexdigest() for p in paths}
    result['unmeasured']=['TTFT','TPOT','system GPU peak','formal serving speedup','formal noninferiority']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=CASE);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Choose a new output; do not overwrite original data')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(derive(a.case),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Source-linked display summary generated; no model execution')
