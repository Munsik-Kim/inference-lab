"""Own-cache greedy and common-reference-prefix diagnostics on frozen scenarios."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model_runtime import Runtime,ARMS,array
from src.intervention import ScopedAttention
from src.storage import load,write_new,file_hash,Ledger,digest
from src.metrics import kl_logits
from src.trajectory import alignment,parse_structured
from scripts.run_case006 import gpu_gate
from scripts.measure_scores import read_inputs,read_inputs_gold


def generate(rt,tokens,arm,eos,reference_tokens=None,reference_logits=None,diagnostic=True):
    torch=rt.torch;generated=[];vectors=[];steps=[];hooks=[];flags=[];decode_routes=[]
    def finite_hook(_m,_a,out):
        if isinstance(out,torch.Tensor):flags.append(torch.isfinite(out).all())
    def observe(when,layer,q,k,v,out,selected):
        if when=='after':
            flags.append(torch.isfinite(out).all())
            if q.shape[-2]==1:decode_routes.append(selected=='native_BF16')
    if diagnostic:
        for layer in rt.model.model.layers:hooks.append(layer.register_forward_hook(finite_hook))
        hooks.append(rt.model.model.norm.register_forward_hook(finite_hook))
    try:
        with torch.inference_mode(),ScopedAttention(rt.registry,arm,observe if diagnostic else None,record_calls=False):
            result=rt.model(input_ids=rt.ids(tokens),use_cache=True,logits_to_keep=1)
            for i in range(64 if reference_tokens is None else len(reference_tokens)):
                if diagnostic:
                    flags.append(torch.isfinite(result.logits).all())
                    if not bool(torch.stack(flags).all()):raise ValueError('BLOCKED_NUMERICAL_VALIDITY in generation full outputs')
                    flags.clear()
                z=array(result.logits[0,-1]) if diagnostic else None
                next_token=int(result.logits[0,-1].argmax())
                if reference_tokens is None:
                    token=next_token
                    if diagnostic:vectors.append(z)
                else:
                    token=reference_tokens[i] # Only supplied AFTER scoring this position.
                    steps.append({'position':i,'same_prior_B_token_count':i,'B_argmax':int(reference_logits[i].argmax()),
                                  'candidate_argmax':next_token,'argmax_agreement':next_token==int(reference_logits[i].argmax()),
                                  'full_vocab_kl':kl_logits(reference_logits[i],z)})
                generated.append(token)
                if token in eos or i==63 or (reference_tokens is not None and i+1==len(reference_tokens)):break
                result=rt.model(input_ids=rt.ids([token]),past_key_values=result.past_key_values,use_cache=True,logits_to_keep=1)
    finally:
        for hook in hooks:hook.remove()
    if diagnostic and not all(decode_routes):raise ValueError('Low precision reached a decode step')
    return generated,vectors,steps,{'full_output_finite':True if diagnostic else None,'all_decode_BF16':all(decode_routes) if diagnostic else None,
                                  'decode_attention_calls_checked':len(decode_routes),'diagnostic':diagnostic}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--gold',nargs='+',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--work-dir',type=Path,required=True);a=p.parse_args()
    case=Path(__file__).resolve().parents[1];freeze=load(a.freeze)
    if freeze['kind']!='EVAL_MANIFEST_FREEZE':raise ValueError('Evaluation freeze required')
    for name,sha in freeze['code_hashes'].items():
        if file_hash(case/name)!=sha:raise ValueError('Frozen code changed')
    inputs=read_inputs(a.inputs);gold=read_inputs_gold(a.gold)
    if any(not r['secondary'] or freeze['input_hashes'].get(r['item_id'])!=r['token_hash'] for r in inputs):raise ValueError('Wrong secondary prompt')
    a.work_dir.mkdir(parents=True,exist_ok=True);gate=gpu_gate();write_new(a.work_dir/f'gate-{time.time_ns()}.json',gate)
    if gate['status']!='RESOURCE_GATE_PASSED':return 20
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True,trust_remote_code=False);rt=Runtime(a.snapshot)
    eos=rt.model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
    ledger=Ledger(a.work_dir/'cells')
    for item in inputs:
        identity={'token_hash':item['token_hash'],'freeze_hash':file_hash(a.freeze)}
        if ledger.get(item['item_id'],identity) is not None:continue
        g=gold[item['base_id']];rows={};baseline_tokens=None;baseline_vectors=None
        for arm in ARMS:
            tokens,vectors,_,valid=generate(rt,item['token_ids'],arm,eos)
            text=tok.decode(tokens,skip_special_tokens=True)
            row={'arm':arm,'tokens':tokens,'text':text,'length':len(tokens),'ended_eos':bool(tokens and tokens[-1] in eos),
                 'parsed':parse_structured(text,g['gold'],g['support_ids']),'validity':valid}
            if arm=='B':baseline_tokens=tokens;baseline_vectors=vectors
            else:
                _,_,steps,common_valid=generate(rt,item['token_ids'],arm,eos,baseline_tokens,baseline_vectors)
                row['common_prefix']={'steps':steps,'first_argmax_disagreement':next((s['position'] for s in steps if not s['argmax_agreement']),None),'validity':common_valid}
                row['free_running_alignment']=alignment(baseline_tokens,tokens,eos)
            # Separate uninstrumented end-to-end greedy request, own cache reset.
            rt.torch.cuda.synchronize();start=time.perf_counter()
            timed_tokens,_,_,_=generate(rt,item['token_ids'],arm,eos,diagnostic=False)
            rt.torch.cuda.synchronize();elapsed=time.perf_counter()-start
            row['request_timing']={'seconds':elapsed,'output_tokens':len(timed_tokens),'tokens':timed_tokens,
                                  'matches_diagnostic_tokens':timed_tokens==tokens,'scope':'Single warmed uninstrumented greedy request; includes prefill, own-cache decode, Python loop and scalar token selection; excludes tokenizer and diagnostic scans','service_TTFT':False}
            rows[arm]=row
        result={k:item[k] for k in ('item_id','base_id','task','split','length','token_hash')}
        result.update({'evidence_kind':'SECONDARY_GENERATION','arms':rows,'gold':g['gold'],'support_ids':g['support_ids'],'eos_ids':sorted(eos),
                       'peak_allocated_bytes':rt.torch.cuda.max_memory_allocated(),'peak_reserved_bytes':rt.torch.cuda.max_memory_reserved()})
        ledger.put(item['item_id'],identity,result)
        print(json.dumps({'item':item['item_id'],'lengths':{arm:rows[arm]['length'] for arm in ARMS},'complete':True}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
