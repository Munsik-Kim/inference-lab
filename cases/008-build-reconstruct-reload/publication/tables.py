"""Report tables from frozen summaries and same-record diagnostics; no measured values typed into prose tables."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'posthoc'))
from recalculate import read, analyze

def md(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                     *['| '+' | '.join(map(str,r))+' |' for r in rows]])

def ci(row, scale=1, digits=6):
    return f"{row['mean']*scale:+.{digits}f} [{row['ci95'][0]*scale:+.{digits}f}, {row['ci95'][1]*scale:+.{digits}f}]"

def tables(case: Path, lang: str) -> dict[str,str]:
    ko=lang=='ko';s=read(case/'results/derived/summary.json');a=read(case/'results/derived/additional_tables.json')
    local=read(case/'results/derived/local_tables.json');post=analyze(case)
    q=s['tracks']['Q']['arms'];r=s['tracks']['R']['arms'];rr=s['tracks']['R']['repair']
    t={};models=read(case/'provenance/models.json');build=read(case/'results/raw/Q/build.json')
    runtime={arm:read(case/f'results/raw/Q/{arm}-runtime.json')['after'][0] for arm in q}
    native_bytes=sum(v['bytes'] for n,v in models['Q']['files'].items() if n.endswith('.safetensors'))
    t['q-size']=md(['측정 범위' if ko else 'Measured boundary','Q-BF16','Q-W4'],[
        ['Safetensors (bytes)',f'{native_bytes:,}',f"{build['safetensors_bytes']:,}"],
        ['Named parameters (bytes)',*[f"{sum(v['bytes'] for v in runtime[x]['parameters'].values()):,}" for x in ['Q-BF16','Q-W4']]],
        ['Allocator peak allocated (GiB)',*[f"{runtime[x]['max_allocated']/2**30:.4f}" for x in ['Q-BF16','Q-W4']]]])
    t['q-score']=md(['지표' if ko else 'Metric','Q-BF16','Q-W4'],[
        ['정답 / 입력' if ko else 'Correct / scenarios',*[f"{q[x]['correct']}/{q[x]['n']}" for x in ['Q-BF16','Q-W4']]],
        ['Δ full NLL (nats), 95% CI','0',ci(q['Q-W4']['delta_full_gold_nll'])],
        ['Δ choice NLL (nats), 95% CI','0',ci(q['Q-W4']['delta_choice_nll'])],
        ['Mean label mass',*[f"{q[x]['allowed_mass_mean']:.6f}" for x in ['Q-BF16','Q-W4']]],
        ['기준선 정답 손실 / 기준선 정답' if ko else 'Regressions / baseline correct','—',f"{q['Q-W4']['transitions']['regression']}/{q['Q-W4']['baseline_correct']}"],
        ['새 정답 / 기준선 오답' if ko else 'Gains / baseline wrong','—',f"{q['Q-W4']['transitions']['gain']}/{q['Q-W4']['baseline_wrong']}"]])
    t['r-local']=md(['구조' if ko else 'Structure','평균 상대오차: 전 → 후' if ko else 'Mean relative error: before → after','오차 에너지 회복률 [95% CI]' if ko else 'Error-energy recovery [95% CI]'],[
        [name,f"{local['splits']['heldout'][name]['mean_relative']*100:.4f}% → {local['splits']['heldout'][name+'-R']['mean_relative']*100:.4f}%",
         f"{rr[name]['pooled_recovery']*100:.4f}% [{rr[name]['ci95'][0]*100:.4f}, {rr[name]['ci95'][1]*100:.4f}]%"] for name in rr])
    t['r-score']=md(['설정 / 정답 수' if ko else 'Arm / correct','Δ full NLL (nats), 95% CI','Mean KL(B ‖ candidate)'],[
        [f"{name}: {r[name]['correct']}/{r[name]['n']}",ci(r[name]['delta_full_gold_nll']),f"{r[name]['full_kl_B_candidate_mean']:.6f}"] for name in ['R-B','I25','I25-R','P25','P25-R','S50','S50-R']])
    t['r-contrast']=md(['보정 − 삭제만' if ko else 'Repaired − uncorrected','Δ full NLL (nats), 95% CI'],[
        [name+'-R − '+name,ci(v['delta_full_gold_nll_repair_minus_uncorrected'])] for name,v in rr.items()])
    t['split-recovery']=md(['구조' if ko else 'Structure','Calibration / DEV / held-out (%)','입력 수' if ko else 'Scenarios'],[
        [name,' / '.join(f"{v[k]['pooled_recovery']*100:.4f}" for k in ['calibration','development','heldout']),
         ' / '.join(str(v[k]['n_prompts']) for k in ['calibration','development','heldout'])] for name,v in a['recovery_by_split'].items()])
    rows=[]
    for task in ['retrieval','comparison','code','all']:
        v=post['tracks']['Q'][task];b,c=v['arms']['Q-BF16'],v['arms']['Q-W4']
        rows.append([task,f"{b['correct']}/{v['n']} → {c['correct']}/{v['n']}",
                     f"{c['delta_full']:+.6f} / {c['delta_choice']:+.6f} / {c['delta_mass_nll']:+.6f}"])
    t['q-decomposition']=md(['과제' if ko else 'Task','정답: BF16 → W4' if ko else 'Correct: BF16 → W4','Δ full / choice / mass NLL (nats)'],rows)
    t['code-decisions']=md(['트랙 / 설정' if ko else 'Track / arm','네 보기 선택 / 정답' if ko else 'Four-choice prediction / correct','전체 argmax가 보기인 수' if ko else 'Full argmax in labels'],[
        [track+' / '+arm,', '.join(f'{label}: {count}' for label,count in row['prediction_histogram'].items() if count)+f"; {row['correct']}/64",f"{row['full_argmax_allowed_recorded']}/64"]
        for track in ['Q','R'] for arm,row in post['tracks'][track]['code']['arms'].items()])
    t['q-timing']=md(['요청' if ko else 'Request','BF16 / W4 mean (ms)','BF16/W4 ratio [95% CI]'],[
        [boundary,f"{v['Q-BF16']['wall_ms']:.3f} / {v['Q-W4']['wall_ms']:.3f}",
         f"{v['Q-W4']['speedup']:.4f} [{v['Q-W4']['ci95'][0]:.4f}, {v['Q-W4']['ci95'][1]:.4f}]"] for boundary,v in s['timing']['Q'].items()])
    t['r-timing']=md(['구조 / 경계' if ko else 'Structure / boundary','전/후 시간비 [95% CI]' if ko else 'Before/after time ratio [95% CI]'],[
        [name+' / '+boundary,f"{v['speedup']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]"] for name,values in a['same_size_cost'].items() for boundary,v in values.items()])
    return t

def block(name, table):
    return f'<!-- table:{name} -->\n{table}\n<!-- /table:{name} -->'

def validate_tables(case: Path):
    import re
    for lang,path in [('ko','REPORT.ko.md'),('en','REPORT.md')]:
        text=(case/path).read_text();expected=tables(case,lang)
        names=re.findall(r'<!-- table:([\w-]+) -->',text)
        if len(names)!=len(set(names)) or set(names)!=set(expected):raise ValueError('Missing/duplicate report tables')
        for name,value in expected.items():
            if block(name,value) not in text:raise ValueError('Stale report table: '+path+'/'+name)
