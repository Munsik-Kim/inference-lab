"""CPU presentation audit; recompute simple facts from CSV, check links/help."""
import argparse,csv,importlib,math,re,subprocess,sys
from pathlib import Path
from common import read,write,identity

def load(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;out=a.output;out.mkdir(parents=True,exist_ok=False);pub=r/'public_candidate';facts=read(pub/'facts.json')
    ad=load(r/'outputs/p4a_v2/cross_precision_document.csv');checks=0
    for group in facts['A']['groups']:
        own=[x for x in ad if x['analysis_split']==group['split'] and int(x['length'])==group['length']];assert len(own)==group['documents']
        for cell in ['00','01','10','11']:
            calc=math.fsum(float(x['K'+cell]) for x in own)/len(own);assert math.isclose(calc,group['cells'][cell],rel_tol=1e-12,abs_tol=1e-15);checks+=1
        for mode in ['0','1']:
            diff=math.fsum(float(x['K'+mode+'0'])-float(x['K'+mode+'1']) for x in own)/len(own);assert math.isclose(diff,group['delta_P_given_D1R'+mode]['mean'],rel_tol=1e-12,abs_tol=1e-15);checks+=1
    b=load(r/'outputs/p4b_v1/rounding_cancellation_pairs.csv')
    for group in facts['B']['groups']:
        own=[x for x in b if x['analysis_split']==group['split']]
        pattern=[x for x in own if float(x['R1_individual_error_sum'])<float(x['R0_individual_error_sum']) and float(x['R1_pair_gap_SSE'])>float(x['R0_pair_gap_SSE'])]
        count=sum(float(x['R0_cross'])<0 and float(x['R0_cross'])<float(x['R1_cross']) for x in pattern)
        assert len(pattern)==group['individual_improves_pair_gap_worsens'] and count==group['pattern_and_R0_more_negative_cross'];checks+=2
    c=load(r/'outputs/p4c_v1/operator_metrics.csv')
    for layer in [0,25,27]:
        gaps=[x for x in c if int(x['layer'])==layer and x['comparison']=='P_vs_D1'];assert len(gaps)==15
        assert all(float(x['SSE'])==0 for x in gaps if x['kind']!='ACTUAL_DEV')
        assert all(float(x['SSE'])>0 for x in gaps if x['kind']=='ACTUAL_DEV');checks+=2
    links=[];missing=[]
    # Final metadata is written after this worker exits so the ledger debit is complete.
    deferred={str((r/'records/budget.json').resolve()),str((pub/'decision.json').resolve())}
    for md in pub.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',md.read_text()):
            if target.startswith(('https://','http://','#')):continue
            path=(md.parent/target.split('#')[0]).resolve();links.append({'document':md.name,'target':target,'exists':path.exists(),'final_metadata':str(path) in deferred})
            if not path.exists() and str(path) not in deferred:missing.append(str(path))
    assert not missing,missing
    help_checks=[]
    for filename in ['p4a_v2.py','p4b.py','p4c_analyze.py','audit_B.py','present_v2.py','replay.py','gpu_worker.py']:
        q=subprocess.run([sys.executable,'-B',str(r/'source'/filename),'--help'],capture_output=True,text=True,check=True);assert 'usage:' in q.stdout;help_checks.append(filename)
    # Import probes execute no main() and verify the lazy Torch boundary.
    for name in ['p4a_v2','p4b','p4c_analyze','audit_B','present_v2','replay','gpu_worker']:importlib.import_module(name)
    assert 'torch' not in sys.modules
    before=identity(r/'records/worker_ledger.jsonl')
    q=subprocess.run([sys.executable,'-B',str(r/'source/replay.py'),'--root',str(r),'--output',str(r),'--label','must-refuse'],capture_output=True,text=True)
    assert q.returncode!=0 and 'OUTPUT_MUST_BE_NEW_AND_EXTERNAL' in q.stderr and identity(r/'records/worker_ledger.jsonl')==before
    write(out/'verification.json',{'status':'COMPLETE','CSV_fact_checks':checks,'local_links':links,'missing_links':missing,'deferred_final_metadata':[str(x) for x in sorted(deferred)],'help_paths':help_checks,'import_Torch_absent':True,'existing_output_rejected_without_worker':True,'original_verdict_preserved':facts['original_primary_verdict']=='INCONCLUSIVE','shared_Korean_English_tables':read(pub/'report_facts_check.json'),'scope':'Presentation and orchestration checks; not additional model experiments or unique scientific samples'})
    print({'status':'COMPLETE','CSV_checks':checks,'links':len(links)})
if __name__=='__main__':main()
