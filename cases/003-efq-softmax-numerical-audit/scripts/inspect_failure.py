"""Inspect fixed worst-case selection from saved results; no model forward."""
import argparse
import csv
import json
import math
import torch
from prepare import CASE, write_json
from audit import load_trace, clean
from numerics import dense, online, metrics, affine_codes, nearest_codes


def main():
    p=argparse.ArgumentParser();p.add_argument('--work-dir',required=True);args=p.parse_args()
    from pathlib import Path
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    summary=json.loads((CASE/'results/aggregate.json').read_text())
    spec=json.loads((CASE/'configs/experiment_spec.json').read_text())
    chosen=spec['calibrated_parameters']
    units=[json.loads(line) for line in (CASE/'results/eval_units.jsonl').read_text().splitlines()]
    # Additional joint factor table uses exactly the prespecified unit/error.
    from analyze import grouped
    table=grouped([u for u in units if u['block']==32],['length','layer','head','method'])
    write_json(CASE/'results/layer_head_length.json',table)
    details=[]
    with torch.inference_mode():
        for rep in summary['representative_units']:
            path=Path(args.work_dir)/'traces/eval'/f"{rep['document_id']}-n{rep['length']}-l{rep['layer']}.pt"
            s,v,meta=load_trace(path)
            hi=meta['heads'].index(rep['head'])
            ref=dense(s,v)
            result=online(s,v,rep['method'],128,32,**chosen)
            nearest=online(s,v,'nearest_efq_scale',128,32)
            actual=((result['out'][hi]-ref['out'][hi]).double().norm()/ref['out'][hi].double().norm()).item()
            assert abs(actual-rep['relative_error']) < 2e-6, 'Inspection must reproduce the recorded unit.'
            d=metrics(s,v,ref,result)
            row=int(d['relative_error'][hi].argmax().item())
            n=meta['positions'][row]+1
            diagnostic={k:clean(a[hi,row].item()) for k,a in d.items()}
            # Cancellation ratio: norm(weighted mean V) / weighted mean(norm V).
            vnorm=v[hi].norm(dim=-1)
            ratio=(ref['out'][hi,row].norm()/(ref['probs'][hi,row]*vnorm).sum()).item()
            keys=ref['probs'][hi,row,:n].argsort(descending=True,stable=True)[:16].tolist()
            records=[]
            for key in keys:
                tile_start=key//128*128; tile_end=min(tile_start+128,s.shape[-1])
                running=s[hi,row,:tile_end].max()
                block_start=key//32*32; block_end=min(block_start+32,tile_end)
                local=(s[hi,row,block_start:block_end]-running).max()
                exponent=torch.floor((local+math.log(2/9))/math.log(2)).clamp(-127,127)
                x=s[hi,row,key]-running
                z=x-exponent*math.log(2)-math.log(6)
                tau,h=(-3.06,2.3) if rep['method']=='efq_mean' else (chosen['tau'],chosen['h'])
                records.append({'key_position':key,'logit':s[hi,row,key].item(),'reference_probability':ref['probs'][hi,row,key].item(),
                    'method_effective_probability':result['probs'][hi,row,key].item(),
                    'nearest_same_scale_probability':nearest['probs'][hi,row,key].item(),
                    'scale_exponent':exponent.item(),'residual':z.item(),
                    'method_code':affine_codes(z,tau,h).item(),
                    'nearest_code':nearest_codes(x.exp()/torch.ldexp(torch.ones_like(exponent),exponent.int())).item()})
            details.append({'selection':rep['selection'],'unit':rep,'largest_row_query_position':meta['positions'][row],
                            'row_diagnostics':diagnostic,'reference_output_cancellation_ratio':ratio,
                            'cancellation_definition':'norm(sum(p*V))/sum(p*norm(V)); exploratory diagnostic, not a failure threshold',
                            'top16_reference_keys':records})
    write_json(CASE/'results/representative_details.json',details)
    print(json.dumps(clean(details),indent=2))


if __name__=='__main__':main()
