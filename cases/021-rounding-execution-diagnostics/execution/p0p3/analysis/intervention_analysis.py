"""CPU analysis and final DEV selection for one-boundary sensitivity."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from common import CASE,read,write,file_row,freeze_files,utc
from primitives import kl,paired_bootstrap
from pipeline import load_logits,csvwrite

def analyze(private,split,out):
    receipt=read(private/'raw'/('p2-intervention-'+split.lower())/'intervention_receipt.json');out.mkdir(parents=True,exist_ok=False);rows=[];positions=[]
    for r in receipt['rows']:
        p=Path(r['logits']['path'])
        if file_row(p)!={k:r['logits'][k] for k in ['bytes','sha256']}:raise ValueError('INTERVENTION_RAW_HASH')
        doc=r['document_id'];ref=load_logits(private/'main'/split/'D1_R0'/f'{doc}_n512_r0_logits.npy');base=load_logits(private/'main'/split/'P_R0'/f'{doc}_n512_r0_logits.npy');actual=load_logits(p)
        before=kl(ref,base);after=kl(ref,actual);change=after-before
        rows.append({'document_id':doc,'split':split,'layer':r['layer'],'boundary':r['boundary'],'baseline_KL':float(before.mean()),'replaced_KL':float(after.mean()),'KL_change':float(change.mean()),'positions':16})
        for j,pos in enumerate(r['positions']):positions.append({'document_id':doc,'split':split,'layer':r['layer'],'boundary':r['boundary'],'position':pos,'baseline_KL':float(before[j]),'replaced_KL':float(after[j]),'KL_change':float(change[j])})
    candidates=[]
    for candidate in receipt['candidates']:
        sub=[r for r in rows if r['layer']==candidate['layer'] and r['boundary']==candidate['boundary']]
        candidates.append({**candidate,'documents':len(sub),'mean_KL_change':float(np.mean([r['KL_change'] for r in sub])),'mean_baseline_KL':float(np.mean([r['baseline_KL'] for r in sub])),'mean_replaced_KL':float(np.mean([r['replaced_KL'] for r in sub])),'improved_documents':sum(r['KL_change']<0 for r in sub),'worsened_documents':sum(r['KL_change']>0 for r in sub)})
    result={'split':split,'full_forwards':len(rows),'candidates':candidates,'positive_change':'More discrepancy; negative change means less discrepancy','scope':'One-boundary downstream sensitivity; reference value recomputed from the current P block input'}
    if split=='DEV':
        if len(candidates)!=4 or any(r['documents']!=8 for r in candidates):raise ValueError('DEV_INTERVENTION_DENOMINATOR')
        ranked=sorted(candidates,key=lambda r:(-abs(r['mean_KL_change']),r['layer'],0 if r['boundary']=='post_RoPE_Q' else 1));result['selected']={'layer':ranked[0]['layer'],'boundary':ranked[0]['boundary']}
    else:
        if len(candidates)!=1 or candidates[0]['documents']!=24:raise ValueError('EVAL_INTERVENTION_DENOMINATOR')
        result['secondary_document_bootstrap']=paired_bootstrap([r['KL_change'] for r in rows])
    csvwrite(out/'intervention_metrics.csv',rows);csvwrite(out/'intervention_positions.csv',positions);write(out/'summary.json',result);print(result)

def finalize(private,dev,out):
    provisional=read(CASE/'provenance/dev_layers_freeze.json');candidate=read(dev/'summary.json');capture=read(private/'raw/p0-actual-capture-validation/capture_validation.json');preflight=read(private/'raw/p2-intervention-preflight/intervention_preflight.json');local=read(private/'raw/p2-local-dev/local_metrics.json')
    if local['block_replay_count']!=96 or len(preflight['rows'])!=4 or any(r['noop']['SSE']!=0 or r['auxiliary_D1_same_input']['SSE']!=0 for r in preflight['rows']):raise ValueError('INTERVENTION_IMPLEMENTATION_GATE')
    if (private/'main/EVAL').exists():raise ValueError('EVAL_ALREADY_GENERATED')
    for r in capture['rows']:
        for key in ['native_vs_R0_noop','R0_actual_CPU_copy_hook','R1_actual_CPU_copy_hook','R0_after_R1_restoration']:
            if r[key]['SSE']!=0:raise ValueError('MEASUREMENT_INTERFERENCE')
    provisional.update(utc=utc(),dev_complete=True,role='Final selection and analysis freeze before any EVAL output',intervention={'status':'SELECTED','candidate':candidate['selected'],'selection_rule':'largest absolute DEV mean KL change; ties earlier layer then post_RoPE_Q','dev_summary':file_row(dev/'summary.json'),'preflight':file_row(private/'raw/p2-intervention-preflight/intervention_preflight.json'),'source_bundle_sha256':preflight['source_bundle_sha256']},analysis_freeze=freeze_files(list((CASE/'analysis').glob('*.py'))+[CASE/'protocol.json',CASE/'inputs/manifest.json']),eval_observed=False,dev_local_receipt=file_row(private/'raw/p2-local-dev/local_metrics.json'))
    write(out,provisional);print({'EVAL_gate':'READY','layers':provisional['layers'],'intervention':provisional['intervention']['candidate'],'tau_abs':provisional['tau_abs']})

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['analyze','finalize']);p.add_argument('--private',type=Path,required=True);p.add_argument('--split',choices=['DEV','EVAL']);p.add_argument('--output',type=Path,required=True);p.add_argument('--dev-analysis',type=Path);a=p.parse_args()
    if a.command=='analyze':analyze(a.private,a.split,a.output)
    else:finalize(a.private,a.dev_analysis,a.output)

if __name__=='__main__':main()
