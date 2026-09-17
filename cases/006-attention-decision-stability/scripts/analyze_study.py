"""Recalculate measured study records and optional timing into a new directory."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import load,write_new,external_directory
from src.study_analysis import analyze_records,distribution
from src.timing import paired_ratios,timing_interval


def read_records(paths):
    rows=[]
    for path in paths:
        if path.is_dir():rows.extend(load(p)['payload'] for p in sorted(path.glob('*.json')))
        else:rows.extend(load(path))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',nargs='+',type=Path,required=True)
    p.add_argument('--timing',nargs='*',type=Path);p.add_argument('--secondary',nargs='*',type=Path)
    p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=external_directory(a.output,a.case);summary,pairs=analyze_records(read_records(a.records))
    if a.timing:
        cells=read_records(a.timing);rows=[r for c in cells for r in c['rows']]
        summary['model_timing']={arm:timing_interval(paired_ratios(rows,arm)) for arm in ('A_PUBLIC','V4')}
        summary['model_timing']['latencies']={arm:{'wall_seconds':distribution([r['seconds_per_call'] for r in rows if r['arm']==arm]),
                  'local_first_token_seconds':distribution([r['local_first_token_seconds_per_call'] for r in rows if r['arm']==arm]),
                  'event_seconds':distribution([r['cuda_event_seconds_per_call'] for r in rows if r['arm']==arm]),
                  'round_median_seconds':{str(i):distribution([r['seconds_per_call'] for r in rows if r['arm']==arm and r['process']==i])['median'] for i in range(3)}} for arm in ('B','A_PUBLIC','V4')}
        summary['model_timing']['memory']={'peak_allocated_bytes':max(c['peak_allocated_bytes'] for c in cells),'peak_reserved_bytes':max(c['peak_reserved_bytes'] for c in cells),'scope':'Torch allocator in timed model processes, shared BF16 weights; no weight savings claim'}
    else:summary['model_timing']={'status':'NOT_RUN'}
    if a.secondary:
        rows=read_records(a.secondary);ss={'scenarios':len(rows),'evidence_kind':'SECONDARY_GENERATION','arms':{}}
        for arm in ('B','A_PUBLIC','V4'):
            armrows=[r['arms'][arm] for r in rows]
            ss['arms'][arm]={'schema_valid':sum(r['parsed']['schema_valid'] for r in armrows),'answer_correct':sum(r['parsed']['answer_correct'] for r in armrows),'evidence_correct':sum(r['parsed']['evidence_correct'] for r in armrows),
                 'length':distribution([r['length'] for r in armrows]),'request_seconds':distribution([r['request_timing']['seconds'] for r in armrows]),'timed_tokens_repeat_matches':sum(r['request_timing']['matches_diagnostic_tokens'] for r in armrows)}
            if arm!='B':ss['arms'][arm].update({'common_prefix_disagreements':sum(r['common_prefix']['first_argmax_disagreement'] is not None for r in armrows),
                'free_running_divergences':sum(r['free_running_alignment']['first_difference_zero_based'] is not None for r in armrows),
                'eight_token_realignments':sum(r['free_running_alignment']['eight_token_realignment_start'] is not None for r in armrows),
                'final_parsed_choice_matches_B':sum(r['arms']['B']['parsed']['schema_valid'] and r['arms'][arm]['parsed']['schema_valid'] and r['arms']['B']['parsed']['parsed']['choice']==r['arms'][arm]['parsed']['parsed']['choice'] for r in rows)})
        summary['secondary']=ss
    else:summary['secondary']={'status':'NOT_RUN'}
    write_new(out/'summary.json',summary);write_new(out/'pairs.json',pairs)
    print(json.dumps({'paired_items':len(pairs),'groups':list(summary['groups'])}))


if __name__=='__main__':main()
