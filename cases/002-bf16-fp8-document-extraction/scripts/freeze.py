"""Freeze a dev-validated configuration before any evaluation request."""
import argparse,datetime,hashlib,json,pathlib
from data import read_jsonl,dump
from score import score

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);p.add_argument('--dev-runs',type=pathlib.Path,nargs=2,required=True);a=p.parse_args();root=a.root
    target=root/'configs/experiment_spec.json';assert not target.exists(),'Existing spec is immutable; use a new version for a new experiment.'
    config=json.loads((root/'configs/common.json').read_text());cfg_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    gold={g['id']:g['values'] for g in read_jsonl(root/'data/dev_gold.jsonl')};dev={}
    for folder in a.dev_runs:
        m=json.loads((folder/'server.json').read_text());assert m.get('measurement_complete') and not m.get('error')
        assert m['common_config_sha256']==cfg_hash
        rows=[r for r in read_jsonl(folder/'requests.jsonl') if r['phase']=='measured']
        assert len(rows)==20 and len({r['case_id'] for r in rows})==20
        assert all(not r['error'] and r['input_tokens']==r['expected_input_tokens'] and r['raw_output'] for r in rows)
        dev[m['model']]={'run_id':m['run_id'],'requests':len(rows),'correct':sum(score(r['raw_output'],gold[r['case_id']])['document_correct'] for r in rows),'runner_sha256':m['runner_sha256']}
    assert set(dev)=={'bf16','fp8'}
    assert dev['bf16']['runner_sha256']==dev['fp8']['runner_sha256']==hashlib.sha256((root/'scripts/run.py').read_bytes()).hexdigest()
    # Accuracy is reported, never used as a gate to discard either deployment.
    files=[p for directory in ['data','scripts','provenance'] for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [root/'configs/prompt.json',root/'configs/common.json']
    spec={'experiment_id':'case002-v1-20260910','frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'synthetic':True,'human_review':False,
          'primary_question':'Memory, streaming latency, and strict gold accuracy for official BF16 vs FP8 Qwen3-4B-Instruct-2507 on synthetic Korean document extraction.',
          'common_config':config,'common_config_sha256':cfg_hash,'dev_results':dev,'prompt_revisions_after_dev':1,
          'evaluation_schedule':[[1,['bf16','fp8']],[2,['fp8','bf16']],[3,['bf16','fp8']]],'document_order':'random.Random(20260910 + round).shuffle(eval_inputs); identical order for both models in each round.',
          'primary_quality_sample':{'round':1,'n':100,'short':50,'long':50},'other_rounds':'Latency variability and repeated-output stability only; not 300 independent documents.',
          'bootstrap':{'method':'paired stratified document bootstrap; keep short/long weights 0.5 each','resamples':10000,'seed':20260910,'interval_percentiles':[2.5,97.5]},
          'sampling_measurements':{'ttft':'HTTP send start to first nonempty generated content chunk','total':'HTTP send start to SSE [DONE]','memory':'NVML entire device every 100 ms; sampled peak; includes background usage','decode_approximation':'(total - TTFT)/(output_tokens - 1), with transport/finish overhead; streaming chunks are not tokens'},
          'frozen_files_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
    dump(target,spec);print(json.dumps({'experiment_spec_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'dev':dev},indent=2))

if __name__=='__main__':main()
