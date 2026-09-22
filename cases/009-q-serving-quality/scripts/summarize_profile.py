"""Compact DEV trace diagnostics; never substitute these for primary timings."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def category(name):
    n=name.lower()
    if 'reshape_and_cache' in n:return 'KV write/cache'
    if 'marlin' in n:return 'Marlin linear or fused linear/activation/norm'
    if 'flash' in n or 'attention' in n:return 'attention'
    if any(t in n for t in ['gemm','gemv','cutlass']):return 'other linear/GEMM'
    if any(t in n for t in ['norm','silu','elementwise','copy']):return 'elementwise/norm/activation/copy'
    return 'other'


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report={'scope':'DEV worker trace only, 12 scheduled worker steps maximum per control; category duration sums are not disjoint end-to-end costs and not primary timing',
       'token_row_dimension':'NOT_RECORDED; client concurrency is not a GEMM row count',
       'records':[]}
    for arm in ['BF16','W4']:
        paths=sorted((a.work/f'dev-{arm}-4GiB-01/profile').glob('rank*.pt.trace.json.gz'))
        if len(paths)!=2:raise ValueError('expected fixed C1/C16 worker traces')
        for c,path in zip([1,16],paths):
            events=json.loads(gzip.decompress(path.read_bytes()))['traceEvents'];dur=defaultdict(float);counts=defaultdict(int)
            for e in events:
                if e.get('cat')=='kernel':
                    k=category(e['name']);dur[k]+=e.get('dur',0);counts[k]+=1
            memory=[e['args'] for e in events if e.get('name')=='[memory]' and e.get('args',{}).get('Device Type')==1]
            launches=[e for e in events if e.get('name')=='cudaGraphLaunch']
            report['records'].append({'arm':arm,'client_concurrency':c,'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'cudaGraphLaunch_count':len(launches),'kernel_duration_sum_us':dict(dur),'kernel_counts':dict(counts),
              'DEV_observed_max_allocator_bytes':{k:max((m.get(k,0) for m in memory),default=None) for k in ['Total Allocated','Total Reserved']},
              'CPU_graph_launch_api_duration_sum_us':sum(e.get('dur',0) for e in launches)})
    with a.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
