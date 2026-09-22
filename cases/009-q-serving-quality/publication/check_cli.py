"""Run an installed CLI outside the checkout and check official retained scalars."""
import argparse,json,math,subprocess,sys
from pathlib import Path

def run(root,python,output):
    output.mkdir(parents=True,exist_ok=False);case=root/'cases/009-q-serving-quality'
    subprocess.run([str(python),'-B',str(root/'packages/diova-compare/examples/export_case009.py'),'--case',str(case),'--output',str(output/'inputs')],check=True,cwd=output)
    expected=json.loads((case/'results/derived/quality_summary.json').read_text());checked=[]
    for baseline in sorted((output/'inputs').glob('*-BF16.jsonl')):
        name=baseline.name.removesuffix('-BF16.jsonl');candidate=baseline.with_name(name+'-W4.jsonl');dest=output/name
        subprocess.run([str(python),'-B','-m','diova_compare.cli','compare','--baseline',str(baseline),'--candidate',str(candidate),'--output',str(dest)],check=True,cwd=output)
        report=json.loads((dest/'report.json').read_text(),parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        task=next(t for t in expected['tasks'] if name.startswith(t['task']+'-'));fil='flexible-extract' if 'flexible-extract' in name else 'strict-match' if 'strict-match' in name else 'none'
        for group,value in report['tasks'].items():
            ex=task['groups'][group.split('@')[0]+'/'+fil]
            if '-acc_norm' in name:ex=ex['acc_norm']
            for key in ['baseline_correct','candidate_correct','both_correct','both_wrong','all_answer_disagreement','correct_to_wrong','wrong_to_correct']:assert value['counts'][key]==ex[key],(name,key)
            assert value['counts']['wrong_to_wrong']==ex['wrong_to_different_wrong']
            for arm,side in [('BF16','baseline'),('W4','candidate')]:assert (value['source_process_exits'][side]==[0])==(task['process_status'][arm]=='COMPLETE')
            if 'choice_score_diagnostic' in ex:
                for metric,key in [('delta_gold_nll','mean_gold_NLL_delta_nats'),('delta_brier','mean_Brier_delta')]:assert abs(value['metrics'][metric]['mean']-ex['choice_score_diagnostic'][key])<2e-12
        # The former formula is safe on these homogeneous measured inputs. Compare
        # it only here, never use it in the fixed runtime core or edge fixtures.
        old_diff=0.
        br=[json.loads(x) for x in baseline.read_text().splitlines()];cr=[json.loads(x) for x in candidate.read_text().splitlines()];index={(x['task'],x['task_version'],x['sample_id']):x for x in report['pairs']}
        for b,c in zip(br,cr):
            assert (b['task'],b['sample_id'])==(c['task'],c['sample_id'])
            if 'distribution' in b:
                pp=b['distribution']['probabilities'];qq=c['distribution']['probabilities'];old=math.fsum(x*math.log(x/y) for x,y in zip(pp,qq) if x>0)
                new=index[(b['task'],b['task_version'],b['sample_id'])]['choice_kl_B_to_C'];old_diff=max(old_diff,abs(new-old))
        checked.append({'metric':name,'pairs':report['n_pairs'],'task_groups':len(report['tasks']),'maximum_KL_rounding_difference':old_diff})
    result={'status':'PASS','version':'0.1.1','metrics':checked,'scope':'Exact counts/IDs/source exits and FP64 point metrics. CLI item bootstrap is not original subject-cluster inference. Filter views are not independent samples.'}
    (output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--python',type=Path,default=Path(sys.executable));p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root.resolve(),a.python.absolute(),a.output.resolve())
if __name__=='__main__':main()
