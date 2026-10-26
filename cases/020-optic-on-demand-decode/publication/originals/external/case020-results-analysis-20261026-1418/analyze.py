"""Read-only Case020 result analysis and standalone Korean report generation."""
import argparse,base64,csv,datetime,hashlib,html,json,math,pathlib,re,statistics
def read(p):return json.loads(p.read_text())
def info(p):b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def quantile(x,q):
 s=sorted(x);i=(len(s)-1)*q;j=math.floor(i);return s[j]+(s[min(j+1,len(s)-1)]-s[j])*(i-j)
def mdtable(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
def write_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def collect(case):
 protocol=read(case/'configs/protocol.json');run=case/'results'/protocol['run_id'];ledger=read(run/'worker_ledger.json');inputs=read(case/'provenance/input_manifest.json');sources={};actual={};lat={};memory={}
 def source(p):sources[str(p.relative_to(case))]=info(p)
 for rel in ['PUBLIC_MANIFEST.json','configs/protocol.json','provenance/freeze.json','results/summary/latency.csv','results/summary/memory.csv','results/summary/paired.csv','results/gpu_worker_budget.json']:
  source(case/rel)
 for w in ledger['workers']:
  if w['kind']!='cell':continue
  p=run/w['raw'];source(p);r=read(p);c=r['cell'];key=(c['shape'],c['M'],c['U'],c['block'],c['method']);assert key not in actual;actual[key]=r
  hs=r['timing']['host_us'];ev=r['timing']['event_bracket_per_call_us'];assert len(hs)==len(ev)==100
  lat[key]={'host_median_us':statistics.median(hs),'host_p95_us':quantile(hs,.95),'event_mean_us':sum(ev)/len(ev),'setup_ms':r['setup']['wall_seconds']*1000,'host_min_us':min(hs)}
  seen=set();payload=0
  for item in r['resident_inventory']:
   assert math.prod(item['shape'])*item['element_size']==item['payload_bytes'];sk=tuple(item['storage_key'])
   if sk not in seen:payload+=item['storage_bytes'];seen.add(sk)
  ids=sum(x['payload_bytes'] for x in r['id_inventory']);warm=r['resident_after_warmup']['allocated'];peak=r['timed_memory']['allocated_peak']
  memory[key]={'module_bytes':payload,'id_bytes':ids,'warm_allocated_bytes':warm,'residual_bytes':warm-payload-ids,'timed_increment_bytes':peak-warm,'peak_allocated_bytes':peak,'peak_reserved_bytes':r['timed_memory']['reserved_peak']}
 assert len(actual)==90
 # Independently derive main statistics from raw, then compare delivered summaries.
 for row in csv.DictReader((case/'results/summary/latency.csv').open()):
  key=(row['shape'],int(row['M']),int(row['U']),int(row['block']),row['method']);ours=lat[key]
  for col,k in [('host_median_us','host_median_us'),('host_p95_us','host_p95_us'),('event_bracket_mean_per_call_us','event_mean_us')]:assert math.isclose(float(row[col]),ours[k],rel_tol=1e-12,abs_tol=1e-8)
 pairs=[];conditions=[];bridge=[];cross=[]
 for inp in inputs:
  s,M,U=inp['shape'],inp['M'],inp['U'];pr=[]
  for b in range(3):
   a,bb,c=[lat[(s,M,U,b,m)] for m in 'ABC'];p={'shape':s,'M':M,'U':U,'condition':inp['condition'],'block':b,'B_A_host':bb['host_median_us']/a['host_median_us'],'C_B_host':c['host_median_us']/bb['host_median_us'],'C_B_event':c['event_mean_us']/bb['event_mean_us'],'C_minus_B_host_us':c['host_median_us']-bb['host_median_us']};pairs.append(p);pr.append(p)
  conditions.append({'shape':s,'M':M,'U':U,'host_C_faster_blocks':sum(x['C_B_host']<1 for x in pr),'event_C_faster_blocks':sum(x['C_B_event']<1 for x in pr),'host_ratio_min':min(x['C_B_host'] for x in pr),'host_ratio_max':max(x['C_B_host'] for x in pr),'event_ratio_min':min(x['C_B_event'] for x in pr),'event_ratio_max':max(x['C_B_event'] for x in pr)})
 for s in ['SMALL','LARGE']:
  for method in 'ABC':
   mm=[memory[(s,2048,512,b,method)] for b in range(3)];assert all(x==mm[0] for x in mm);bridge.append(dict(shape=s,method=method,**mm[0]))
 for b in range(3):
  a,bb=[lat[('LARGE',2048,512,b,m)] for m in 'AB'];numerator=(a['setup_ms']-bb['setup_ms'])*1000;denominator=bb['host_median_us']-a['host_median_us'];n=numerator/denominator if numerator>0 and denominator>0 else None
  cross.append({'block':b,'A_setup_ms':a['setup_ms'],'B_setup_ms':bb['setup_ms'],'A_host_median_us':a['host_median_us'],'B_host_median_us':bb['host_median_us'],'B_minus_A_us':denominator,'derived_crossing_requests':n,'first_integer_requests':math.ceil(n) if n else None})
 wins=[p for p in pairs if p['C_B_host']<1];eventwins=[p for p in pairs if p['C_B_event']<1]
 metrics={'primary_run_id':protocol['run_id'],'raw_cells_recomputed':90,'paired_blocks':len(pairs),'conditions':len(conditions),'B_host_faster_blocks':sum(p['B_A_host']<1 for p in pairs),'C_host_faster_blocks':len(wins),'C_event_faster_blocks':len(eventwins),'C_host_faster_without_duplicates':sum(p['M']==p['U'] for p in wins),'C_host_faster_with_duplicates':sum(p['M']>p['U'] for p in wins),'C_faster_under_both_metrics_same_pair':sum(p['C_B_host']<1 and p['C_B_event']<1 for p in pairs),'host_event_winner_disagreements':sum((p['C_B_host']<1)!=(p['C_B_event']<1) for p in pairs),'conditions_C_host_faster_all3':sum(p['host_C_faster_blocks']==3 for p in conditions),'conditions_C_event_faster_all3':sum(p['event_C_faster_blocks']==3 for p in conditions),'three_block_rule':'Descriptive consistency check chosen after diagnostic run, not a statistical significance test','sources':sources}
 assert metrics['C_host_faster_blocks']==5 and metrics['C_event_faster_blocks']==4 and metrics['C_host_faster_without_duplicates']==3 and metrics['C_faster_under_both_metrics_same_pair']==0
 return protocol,lat,memory,pairs,conditions,bridge,cross,metrics
def figures(out,lat,bridge,pairs,cross):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'optic-analysis-20261026'})
 def save(fig,name):
  fig.savefig(out/'figures'/f'{name}.png',dpi=160,bbox_inches='tight',facecolor='white');fig.savefig(out/'figures'/f'{name}.svg',bbox_inches='tight',metadata={'Date':None});plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.8));parts=[('module_bytes','Module tensors','#0284c7'),('id_bytes','Common IDs','#cbd5e1'),('residual_bytes','Warm allocated residual (unattributed)','#475569'),('timed_increment_bytes','Peak minus warm allocated','#f59e0b')]
 for ax,shape in zip(axes,['SMALL','LARGE']):
  rows=[r for r in bridge if r['shape']==shape];bottom=np.zeros(3)
  for k,label,color in parts:
   vals=np.array([r[k]/2**20 for r in rows]);ax.bar(range(3),vals,bottom=bottom,label=label,color=color,width=.58);bottom+=vals
  for i,t in enumerate(bottom):ax.text(i,t+max(bottom)*.025,f'{t:.2f}',ha='center')
  ax.set_xticks(range(3),['A dense','B request','C unique']);ax.set_ylabel('PyTorch allocated memory (MiB)');ax.set_title(shape+' | M2048 / U512');ax.set_ylim(0,max(bottom)*1.18);ax.grid(axis='y',alpha=.15)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=9,bbox_to_anchor=(.5,-.035));fig.suptitle('Module payload is only one part of the observed allocated peak',fontsize=14);fig.tight_layout(rect=(0,.1,1,.93));save(fig,'01_memory_accounting')
 fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True);order=[(s,M,U) for s in ['SMALL','LARGE'] for M,U in [(1,1),(128,128),(128,32),(2048,2048),(2048,512)]]
 for ax,key,color,title in [(axes[0],'C_B_host','#0284c7','Host median ratio'),(axes[1],'C_B_event','#d97706','Event bracket mean per call ratio')]:
  med=[]
  for i,(s,M,U) in enumerate(order):
   rows=sorted([p for p in pairs if (p['shape'],p['M'],p['U'])==(s,M,U)],key=lambda r:r['block']);med.append(statistics.median(p[key] for p in rows))
   for p in rows:ax.scatter(i+(p['block']-1)*.09,p[key],marker=['o','s','^'][p['block']],color=color,s=38,zorder=3)
  ax.plot(range(10),med,color=color,alpha=.45,label='Median of three paired ratios');ax.axhline(1,color='#b91c1c',ls='--',lw=1,label='C = B');ax.set_ylim(0,max(p[key] for p in pairs)*1.12);ax.set_ylabel('C / B (lower is faster)');ax.set_title(title);ax.grid(axis='y',alpha=.17);ax.legend(fontsize=9,loc='upper left')
 axes[1].set_xticks(range(10),[f"{'S' if s=='SMALL' else 'L'} {M}/{U}" for s,M,U in order],rotation=30,ha='right');axes[1].set_xlabel('Shape, request M / unique U');fig.suptitle('Observed C / B rankings depend on timing pass',fontsize=14);fig.text(.5,.005,'o / square / triangle = blocks0 /1 /2. Separate host and event passes; no overlapping winner among the30 paired blocks.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.95));save(fig,'02_latency_consistency')
 fig,axes=plt.subplots(1,3,figsize=(12,4.2),sharey=True);n=np.linspace(0,3200,161);ytop=1.1*max((r[k+'_setup_ms']+3200*r[k+'_host_median_us']/1000)/1000 for r in cross for k in ['A','B'])
 for ax,row in zip(axes,cross):
  aa=(row['A_setup_ms']+n*row['A_host_median_us']/1000)/1000;bb=(row['B_setup_ms']+n*row['B_host_median_us']/1000)/1000;ax.plot(n,aa,color='#334155',label='A setup + N calls');ax.plot(n,bb,color='#0284c7',label='B setup + N calls');cross_n=row['derived_crossing_requests'];ax.axvline(cross_n,color='#94a3b8',ls='--');ax.text(cross_n,.015,f'N ~ {math.ceil(cross_n):,}',rotation=90,ha='right',va='bottom');ax.set_title(f"Block {row['block']}");ax.set_xlabel('Assumed repeated calls N');ax.grid(alpha=.17);ax.set_ylim(0,ytop)
 axes[0].set_ylabel('Derived cumulative wall (s)');axes[0].legend(fontsize=8,loc='upper left');fig.suptitle('Setup amortization scenario | LARGE M2048 / U512',fontsize=14);fig.text(.5,.005,'Calculated from setup wall and host medians; not a measured lifecycle benchmark. Cold call / warm-up / worker startup excluded.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,1,.91));save(fig,'03_setup_scenario')
def report(case,out,protocol,lat,mem,pairs,conditions,bridge,cross,metrics):
 def series(s,M,U,method,key):return [lat[(s,M,U,b,method)][key] for b in range(3)]
 def fmtspan(vals,d=2):return f'{min(vals):.{d}f} - {max(vals):.{d}f}'
 focus=[r for r in bridge if r['shape']=='LARGE'];by={r['method']:r for r in focus};a,bb,c=[by[m] for m in 'ABC'];mib=lambda v:v/2**20
 payload_ratio=a['module_bytes']/bb['module_bytes'];peak_ratio=a['peak_allocated_bytes']/bb['peak_allocated_bytes'];peak_saving=(1-bb['peak_allocated_bytes']/a['peak_allocated_bytes'])*100;c_peak_saving=(1-c['peak_allocated_bytes']/bb['peak_allocated_bytes'])*100;c_reserved_growth=(c['peak_reserved_bytes']/bb['peak_reserved_bytes']-1)*100;warm_ratio=a['warm_allocated_bytes']/bb['warm_allocated_bytes']
 summary_table=mdtable(['방법','상주 module MiB','warm allocated MiB','timed allocated peak MiB','reserved peak MiB'],[[m,f"{mib(by[m]['module_bytes']):.3f}",f"{mib(by[m]['warm_allocated_bytes']):.2f}",f"{mib(by[m]['peak_allocated_bytes']):.2f}",f"{mib(by[m]['peak_reserved_bytes']):.2f}"] for m in 'ABC'])
 details=mdtable(['구성','SMALL bytes','LARGE bytes'],[['token별 코드',f"{8192*8:,}",f"{32768*8:,}"],['codebook',f"{8*256*4*2:,}",f"{8*256*4*2:,}"],['layer decoder weight',f"{8*128*32*2:,}",f"{16*256*32*2:,}"],['decoder bias',f"{8*128*2:,}",f"{16*256*2:,}"],['group offsets','64','64'],['합계','149,568','548,928']])
 latency_table=mdtable(['Block','A median / p95 us','B median / p95 us','C median / p95 us','C/B median'],[[b,*[f"{lat[('LARGE',2048,512,b,m)]['host_median_us']:.2f} / {lat[('LARGE',2048,512,b,m)]['host_p95_us']:.2f}" for m in 'ABC'],f"{next(p['C_B_host'] for p in pairs if (p['shape'],p['M'],p['U'],p['block'])==('LARGE',2048,512,b)):.3f}"] for b in range(3)])
 ratio_table=mdtable(['Shape','M / U','C/B host b0, b1, b2','C/B event b0, b1, b2'],[[s,f'{M} / {U}',', '.join(f"{p['C_B_host']:.3f}" for p in sorted(pairs,key=lambda r:r['block']) if (p['shape'],p['M'],p['U'])==(s,M,U)),', '.join(f"{p['C_B_event']:.3f}" for p in sorted(pairs,key=lambda r:r['block']) if (p['shape'],p['M'],p['U'])==(s,M,U))] for s in ['SMALL','LARGE'] for M,U in [(1,1),(128,128),(128,32),(2048,2048),(2048,512)]])
 cross_table=mdtable(['Block','A setup ms','B setup ms','B - A median us','계산된 교차 호출 수'],[[r['block'],f"{r['A_setup_ms']:.2f}",f"{r['B_setup_ms']:.2f}",f"{r['B_minus_A_us']:.2f}",f"{r['first_integer_requests']:,}"] for r in cross])
 verdict=mdtable(['질문','현재 근거로 내릴 수 있는 판단'],[['요청 시 복원이 메모리를 줄였는가','그렇다. module 상주량과 측정 allocated peak 모두 감소했다.'],['전체 GPU 메모리를489배 줄였는가','그 근거는 없다.489배는 선택한 synthetic shape의 module payload 비율이다.'],['중복이75%면 C가 더 빠른가','이 grid에서는 안정적인 속도 이득이 입증되지 않았다.'],['C가 불필요한가','그렇다고 단정할 수 없다. 일부 중복 조건에서 allocated peak를 더 낮췄다.'],['실제 모델에도 같은 배율이 적용되는가','실모델 shape·훈련 artifact·요청분포·전체 model 실행이 필요하다. 이번 결과로 확대할 수 없다.']])
 body=f'''# OPTIC 압축 PLE 실행 비용 결과 분석

같은 압축 표현에서 복원 시점과 중복 처리가 메모리와 호출 비용에 미친 영향

작성일: 2026-10-26 KST | 측정 장치: RTX 5080, sm_120 | 분석 기준: 수정 본 run {protocol['run_id']}

## 1. 핵심 결론

**필요한 임베딩만 복원하는 B는 메모리를 줄였지만, 이번 실험에서 중복 제거 C의 안정적인 속도 이득은 확인되지 않았다.** LARGE의 M2048/U512 조건에서 module 상주량은256MiB에서0.523MiB로 작아졌다. 그러나 출력과 임시 할당까지 포함한 allocated peak는304.02MiB에서80.79MiB로 감소했다. 따라서 약489배의 payload 차이와 약{peak_ratio:.2f}배의 allocated peak 차이를 구분해야 한다.[S1, S2]

C가 B보다 빠른 host block은30개 중5개였다. 이 중3개는 **U=M, 즉 요청에 중복이 없는 경우**였다. host와 event 계측에서 C가 동시에 빨랐던 동일 짝은0개였다. 이 관측은 일부 빠른 실행을 중복 제거의 계산량 절감으로 곧바로 설명하기 어렵다는 뜻이다. 현 상태의 판정은 실행 경로·수치·메모리 효과 검증, 그리고 **중복 제거 속도 이득의 미확정**이다.[S1, S3]

30개 cell을3개 process block으로 반복한 본90 cell-block은 모두 수치 gate를 통과했고 worker 실패·누락은 없었다. 아래 분석은 이미 보존된 raw를 CPU에서 재계산한 것이며 새로운 GPU 실험은 하지 않았다.[S4, S5]

## 2. 비교 대상과 결과를 읽는 범위

PLE는 token이 각 layer에서 사용하는 추가 임베딩이다. codebook은 코드 번호로 조회하는 작은 벡터 모음이고, decoder는 이 벡터를 layer 임베딩으로 바꾸는 affine 식이다. resident는 호출 사이에도 남는 상태, workspace는 호출 도중 쓰는 작업 공간이다.

- A: 공식 decoder로 모든 token을 먼저 복원한 BF16 table에서 요청 행을 읽는다.
- B: 고정한 공식 decoder가 요청된 token만 복원한다.
- C: 매번 unique와 inverse를 계산하고 U개를 공식 decoder로 복원한 뒤 원래 M개 출력 순서를 되살린다.

A의 table도 **같은 압축 artifact의 재구성**이다. 원래 미압축 PLE와 압축 PLE를 비교한 실험은 아니다. 세 경로는 같은 ID, 전체 layer, raw unscaled BF16 출력 계약을 사용한다. SMALL은 V8192/L8/H128, LARGE는 V32768/L16/H256이며 모두 Z32이다. 실모델 shape나 학습된 OPTIC artifact의 품질 실험으로 해석하지 않는다.[S4, S6]

## 3. 메모리 효과는 어느 지표를 보는지에 따라 달라진다

아래는 LARGE M2048/U512 조건이다. 메모리 값은 세 block에서 동일했다. 상주 module은 실제 parameter와 buffer의 고유 storage 합계이며, allocated/reserved는 PyTorch 계측값이다.[S2]

{summary_table}

A/B의 차이는 module payload 기준 **{payload_ratio:.2f}배**, warm allocated 기준 **{warm_ratio:.2f}배**, timed allocated peak 기준 **{peak_ratio:.2f}배**다. peak 감소율은 **{peak_saving:.2f}%**이다. 세 수치는 서로 다른 질문에 답하며, 장치 전체 VRAM 사용량의 감소율로 바꿔 쓰면 안 된다.

![상주 상태와 allocated peak의 관계](figures/01_memory_accounting.png)

그림의 잔류량은 `warm allocated - module storage - ID payload`, 호출 중 증가량은 `timed peak - warm allocated`로 계산했다. module과 ID로 설명되지 않는 약32MiB의 allocated가 남았다. 정확한 소유자는 추적하지 않았으므로 이를 특정 GEMM kernel이나 allocator 자체의 고정 비용으로 단정하지 않는다. 이 그림은 상태 간 차이를 맞춘 회계이며, 각 할당의 생존 구간을 추적한 결과는 아니다.

### 3.1 상주량이 크게 줄어든 구조적 이유

모든 token을 복원한 table은 `2 × V × L × H bytes`가 필요하다. 이번 uint8 코드/FP16 배열 계약의 compact 상태는 `V×G + 2×G×K×D + 2×L×H×Z + 2×L×H + 8×G bytes`다. 뒤의64bytes가 non-persistent group offsets이며, upstream compact_resident_bytes만 사용할 때 빠지는 항목이다.[S2, S6]

{details}

SMALL에서 LARGE로 바꾸면서 dense table은16배 커졌지만 compact 상태는약3.67배 커졌다. V·L·H가 커지는 동안 Z32와 codebook 폭을 고정했기 때문이다. 이는 선택한 형식의 저장 비용을 설명하지만 학습된 표현의 압축 품질을 검증하지는 않는다.

### 3.2 중복 제거는 속도와 별개로 allocated peak를 낮췄다

같은 LARGE M2048/U512에서 C는 B보다 allocated peak를 **{c_peak_saving:.2f}%** 낮췄다. B의 FP32 복원 중간 결과는32MiB이고 최종 BF16 출력은16MiB다. U512만 복원하면 FP32 중간 결과는8MiB, U개 BF16은4MiB로 줄어든다. inverse gather 중에는 U개 decoded4MiB와 최종 M개 출력16MiB가 겹칠 수 있다. 주요 tensor 크기는 B의 약48MiB와 C의 약20MiB 호출 중 증가량을 이해하는 데 도움이 된다. 정확한 peak에는 다른 중간 tensor도 포함된다.[S2, S6]

반면 C의 reserved peak는82MiB에서102MiB로 **{c_reserved_growth:.2f}% 증가**했다. reserved에는 allocator의 이전 사용 이력이 남으므로 작은 live tensor 합계와 같은 방향으로 움직일 필요가 없다. 이번 자료는 C가 allocated peak를 더 낮췄다는 점과 reserved가 더 컸다는 점을 함께 보여 준다. 장치 전체 메모리의 연속 peak는 측정하지 않았다.

## 4. 지연시간의 결론은 메모리보다 약하다

주 지표는 함수 호출부터 완료 synchronize까지 포함한 host wall이다. 입력과 출력은 GPU에 있고 I/O·전송·검산은 timed function 밖이다. 각 block의100개 host 표본에서 median/p95를 구했다. 다음 표는 LARGE M2048/U512의 실제 block 통계이며 서로 다른 block의 표본을 합치지 않았다.[S1]

{latency_table}

이 조건에서 C는 B보다 세 block 모두 느렸고 C/B는1.955-7.064였다. 같은 조건의 C median도247.07-869.78us로 변했다. 따라서 중앙값 하나만 골라 대표적인 처리 시간을 확정하기 어렵다.

비율 비교의30개 짝은10개 입력 조건×3개 block이다. 각 짝은 같은 입력과 block에서 별도 process로 실행한 B와 C의 통계를 비교한다. 개별 timing sample100개를 서로 짝지은 분석은 아니다.

전체 grid에서는 B가 A보다 빠른 host 짝이4/30, C가 B보다 빠른 host 짝이5/30였다. C가 세 block 모두 빠른 입력 조건은0/10이었다. 세 block의 방향을 함께 보는 규칙은 첫 진단 run 관측 뒤 수정 본 run 전에 정한 서술적 기준이며, 통계적 유의성 기준은 아니다. 이는 이번 grid에서 C를 기본적인 latency 개선책으로 채택할 근거가 약하다는 판단을 지지한다. 더 넓은 요청분포나 다른 decoder 폭에서도 C가 느리다는 보편 명제로 확대하지 않는다.[S1, S3]

### 4.1 빠른 다섯 block을 중복 제거 효과로 세기 어려운 이유

host에서 C가 빠른5개 중3개는 U=M이었다. LARGE M1/U1의 block1, LARGE M2048/U2048의 block0·1이다. 이 경우 C가 복원하는 행 수는 B와 같다. 따라서 이3개 승리는 복원 행 수 감소로 설명할 수 없다. 나머지2개 중복 조건도 다른 block에서는 같은 방향의 개선을 보이지 않았다.

repeat75는 복원할 행 수와 GEMM의 명목 연산량을4분의1로 줄인다. 예를 들어 LARGE M2048에서 GEMM 산술량을 FMA당2연산으로 세면 B는약0.537GFLOP, C(U512)는약0.134GFLOP다. 하지만 C는 unique·정렬·inverse와 최종 M개 tensor 생성의 비용을 추가로 낸다. 이 산술량 비율은 전체 operator latency의4배 개선을 보장하지 않는다. 어떤 비용이 지배적인지는 별도 진단이 필요하다.[S6]

### 4.2 host와 event의 승자도 일치하지 않았다

event는100개 bracket의 경과시간을 공통 호출 수 K로 나눈 보조 값이다. host pass와 별도로 실행됐으며 K는 M1/128/2048에서32/8/1이다. stream idle이나 CPU launch gap도 포함할 수 있어 순수 kernel 시간의 합계로 읽지 않는다.

![Host와 event에서의 C/B 짝 비율](figures/02_latency_consistency.png)

host에서는 C가5/30개, event에서는4/30개 짝에서 빨랐다. 두 집합의 교집합은0개이고, 승자 방향이 다른 짝은9/30개다. 이것은 같은 호출에 대한 두 타이머의 모순이라는 뜻은 아니다. 서로 다른 pass의 관측이다. 다만 C의 이득이 timing pass를 바꿔도 안정적으로 나타났다고 주장할 근거는 없다.[S1, S3]

수동 clock 고정이나 scheduler 원인 진단을 하지 않았으므로 변동을 특정 원인으로 단정하지 않는다. floor를 임의로 빼거나 느린 sample을 제거하지 않았다. 세 block과 내부100회 반복으로 일반적인 서비스의 신뢰구간을 만들지도 않았다.

### 4.3 성분 진단은 본 latency의 정확한 분해가 아니다

C의 unique/decode/inverse는 본 측정 뒤 추가 synchronize가 있는 진단 pass에서 측정했다. 각 성분의 합이 본 C와 같을 필요가 없다. 특히 `C - B`를 그대로 unique의 순수 비용으로 부르면 inverse·출력 재구성·pass 간 상태 차이를 누락한다. 기존 기술 보고서의 성분 그림은 원인 후보를 살피는 자료로 쓰고, 주 비교는 분해하지 않은 host 총시간을 유지한다.[S6, S7]

## 5. 전체 복원의 setup 비용은 얼마나 상각돼야 하는가

A는 요청 조회가 가벼운 대신 먼저 전체 table을 복원하고 compact 상태를 정리한다. B는 더 작은 setup 뒤 요청마다 복원한다. 이 절충을 보기 위해 `T_A(N)=setup_A + N×host_median_A`, `T_B(N)=setup_B + N×host_median_B`를 계산했다. **아래는 측정값을 상수로 대입한 시나리오이며, 실제 N회 수명주기를 측정한 결과가 아니다.**[S1]

{cross_table}

![Setup 비용 상각 시나리오](figures/03_setup_scenario.png)

이 단순 계산의 교차점은약920-2,319회다. block마다 약2.5배 차이가 난다. 첫 cold call·warm-up·worker 시작/종료·요청 간 idle·파일 cache 상태를 포함한 실제 lifecycle을 재현하지 않았으므로 운영 임계점으로 사용하지 않는다. 다만 메모리가 허용되고 table을 오래 재사용하는 상황에서는 초기 전체 복원 비용도 비교해야 한다는 점을 보여 준다.

## 6. 검증의 강도와 남은 불확실성

작은 fixture는 저장된 FP16 배열 값을 FP64로 올린 독립 참조로 검사했다. seed0/1/2, 전체257token·3layer, 비영 bias와0대조, shape/order/반복/경계 입력을 포함했다. FP64 loop/vector는 rtol1e-10·atol1e-12, BF16은 overall/layer별 relative L2≤1e-2와 finite 조건을 사용했다. 정확히0인 참조는 max_abs≤1e-6으로 판단했다.[S5]

본 P2는 모든10개 입력의 A/B/C 직접 대조를 전용 process에서 먼저 수행하고, 각 worker가 input/source preflight hash를 확인한 뒤 timing했다. 본90 cell-block은 모두 통과했다. A/B/C가 관측된90개 method-pair에서 bit-identical했지만 이를 필수 gate로 삼지는 않았다. 저장된 B block0 전체 요청 출력도 Torch를 부르지 않는 별도 NumPy 식으로 재검산했다.[S4, S5]

첫 run은 직접 A/B/C 대조를 timing 뒤에 수행한 순서 오류가 있었다. 이를 본 성능 표에서 제외하고 원90 cell-block을 진단 결과로 보존했다. 동일한 shape·seed·ID·허용 오차·loop·순서로 검산을 먼저 수행한 수정 run을 한 번 실행했다. 분석에는 수정 run만 사용했고 두 run의 GPU worker wall은 같은 장부에 누적했다. 총196worker457.424초이며 이 보고서 작성으로 GPU 예산은 추가 소비하지 않았다.[S4]

검증은 구현과 계측 계약의 대응을 뒷받침한다. 실모델 언어 능력·OPTIC 학습 품질·전체 serving의 TTFT/TPOT·KV-cache 재배분 효과는 측정하지 않았다. shape마다 artifact는 하나이고 세 block은 같은 artifact의 process 반복이다. 독립 모델3개나 사용자 과제3개로 부르지 않는다.

## 7. 활용 판단과 포트폴리오에서 설명할 내용

{verdict}

메모리 용량을 우선하면 B가 검토할 만한 실행 방식이다. C는 중복이 많은 입력에서 allocated peak를 추가로 낮추는 선택지가 될 수 있지만, 이번 자료만으로 latency 개선책으로 권하지 않는다. A는 더 큰 상주량을 감당할 수 있고 복원 table을 오래 재사용하는 경우의 비교 기준으로 남긴다. 이 판단은 본 synthetic operator 범위에 한정한다.

포트폴리오에서는 다음과 같이 설명할 수 있다.

> 공식 OPTIC decoder를 고정해 같은 압축 표현의 선복원 조회, 요청 복원, 중복 제거 후 복원을 비교했다. 독립 NumPy FP64 참조와 입력 계약으로 정확성을 검증하고, 상주 tensor·임시 할당·reserved·host/event latency를 구분했다.90개 본 cell-block에서 메모리 절충을 관측했으며, 중복 제거의 속도 이득이 block과 timing pass에 따라 바뀐다는 한계를 원자료와 함께 공개했다. 새로운 codec이나 GEMM kernel을 구현한 성과로 주장하지 않았다.

## 부록 A. 전체 C/B 짝 비율

각 칸은 b0, b1, b2 순서다.1보다 작으면 C가 빠르다. host와 event는 서로 다른 pass이고 event 값은 bracket 평균/K다.

{ratio_table}

## 부록 B. 근거와 재현

- [S1] 본 raw90개 JSON의 host100개·event100개 표본 및 setup wall. 이 보고서에서 median·linear p95·event 평균을 재계산하고 전달된 latency.csv와 대조했다.
- [S2] 본 raw의 resident_inventory, ID/output inventory, warm allocated, timed allocated/reserved. storage 단위 중복을 제거해 module bytes와 메모리 회계를 다시 계산했다.
- [S3] 같은 shape/ID/block의 원 통계에서 B/A와 C/B를 다시 계산했다. 파생 수치는 paired_analysis.csv와 condition_analysis.csv에 보존했다.
- [S4] 원 protocol.json, preflight_SMALL/LARGE.json, worker_ledger.json, gpu_worker_budget.json과 첫 run의 protocol_deviation.json.
- [S5] 원 cpu_correctness.json, GPU pair correctness, fixture_manifest와 독립 CPU 감사. 원 공개 manifest의746파일 bytes/hash도 분석 전에 확인했다.
- [S6] 고정 upstream344c712aeb6620e39eb78930fb922447f0e123f3의 compact_adapter.py 및 측정 worker source. 형식·산술·저장 dtype 해석의 근거다.
- [S7] 기존 REPORT.ko.md와 figures/03_unique_cost의 본 측정/진단 구분. 본 보고서의 원 통계는 기존 문서 문구가 아니라 raw에서 다시 계산했다.

원 Case 위치: `{case}`

본 분석의 코드·파생 JSON/CSV·그림은 이 보고서와 같은 폴더에 있다. 원 Case의 source·raw·요약·PUBLIC_MANIFEST와 이전 인계 TXT는 수정하지 않았다. HTML은 그림을 내장해 단독으로 열 수 있고, Markdown은 같은 폴더의 figures를 참조한다. 이 산출물은 로컬 분석본이며 GitHub·사이트·Pages·Drive에 게시하지 않았다.
'''
 body=re.sub(r'(?<=[가-힣])(?=[0-9])',' ',body).replace('746파일','746개 파일').replace('상태는약','상태는 약').replace('교차점은약','교차점은 약').replace('decoded4MiB','decoded 4MiB').replace('총 196worker457.424초','총 196개 worker, 457.424초').replace('없다.489배','없다. 489배').replace('다.90개','다. 90개').replace('다.1보다','다. 1보다')
 (out/'ANALYSIS.ko.md').write_text(body,encoding='utf-8');return body
def html_report(markdown,out):
 def inline(s):
  s=html.escape(s,quote=False);s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s);s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s);return s
 lines=markdown.splitlines();blocks=[];i=0
 while i<len(lines):
  line=lines[i]
  if not line.strip():i+=1;continue
  if line.startswith('#'):
   level=len(line)-len(line.lstrip('#'));blocks.append(f'<h{level}>{inline(line[level:].strip())}</h{level}>');i+=1;continue
  if line.startswith('|'):
   rows=[]
   while i<len(lines) and lines[i].startswith('|'):rows.append([x.strip() for x in lines[i].strip().strip('|').split('|')]);i+=1
   blocks.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in row)+'</tr>' for row in rows[2:])+'</tbody></table></div>');continue
  image=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
  if image:
   p=out/image[2];data=base64.b64encode(p.read_bytes()).decode();blocks.append(f'<figure><img alt="{html.escape(image[1],quote=True)}" src="data:image/png;base64,{data}"><figcaption>{inline(image[1])}</figcaption></figure>');i+=1;continue
  if line.startswith('- '):
   items=[]
   while i<len(lines) and lines[i].startswith('- '):items.append(lines[i][2:]);i+=1
   blocks.append('<ul>'+''.join('<li>'+inline(x)+'</li>' for x in items)+'</ul>');continue
  if line.startswith('> '):blocks.append('<blockquote>'+inline(line[2:])+'</blockquote>');i+=1;continue
  para=[line];i+=1
  while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','- ','![')) :para.append(lines[i]);i+=1
  blocks.append('<p>'+inline(' '.join(para))+'</p>')
 css='''html{background:#eef2f6;color:#172337;font-family:"Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR",sans-serif}body{margin:0}main{max-width:1050px;margin:32px auto;background:#fff;padding:54px 64px;box-shadow:0 8px 38px #1423340d;border-top:7px solid #0369a1}h1{font-size:30px;line-height:1.4;margin:0 0 15px;color:#0c3550}h2{font-size:23px;margin:45px 0 16px;padding-top:18px;border-top:1px solid #d9e2ec;line-height:1.5}h3{font-size:18px;margin:28px 0 12px;color:#0c4a6e}p,li{font-size:15px;line-height:1.95;word-break:keep-all;overflow-wrap:anywhere}p{margin:14px 0}li{margin:8px 0}strong{color:#075985}code{font-size:.88em;background:#f1f5f9;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;line-height:1.65}th,td{border-bottom:1px solid #dbe3ec;text-align:left;padding:10px 11px;vertical-align:top}th{background:#eff6fb;color:#153e57}tbody tr:nth-child(even){background:#f8fafc}.table-wrap{overflow-x:auto;margin:22px 0}figure{margin:30px 0}img{display:block;max-width:100%;height:auto;margin:auto}figcaption{text-align:center;font-size:12px;color:#53657a;margin-top:9px}blockquote{border-left:4px solid #0ea5e9;background:#f0f9ff;padding:18px 22px;margin:24px 0;font-size:15px;line-height:1.95}footer{color:#64748b;border-top:1px solid #dbe3ec;padding-top:18px;margin-top:40px;font-size:12px}@media(max-width:800px){main{margin:0;padding:24px 20px}h1{font-size:25px}h2{font-size:21px}th,td{padding:8px}}@media print{html{background:#fff}main{margin:0;padding:0;max-width:none;box-shadow:none;border:0}h2,h3{break-after:avoid}figure,table,blockquote{break-inside:avoid}p,li{orphans:3;widows:3}h1{font-size:24px}p,li{font-size:10pt}table{font-size:9pt}@page{size:A4;margin:18mm}}'''
 page='<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OPTIC 압축 PLE 실행 비용 결과 분석</title><style>'+css+'</style></head><body><main>'+''.join(blocks)+'<footer>Case020 결과 분석 ·2026-10-26 KST ·로컬 분석본 ·새 GPU 측정 없음</footer></main></body></html>'
 (out/'ANALYSIS.ko.html').write_text(page,encoding='utf-8')
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parent);args=p.parse_args();case=args.case.resolve();out=args.out.resolve();(out/'figures').mkdir(parents=True,exist_ok=True)
 data=collect(case);protocol,lat,mem,pairs,conditions,bridge,cross,metrics=data
 write_csv(out/'paired_analysis.csv',pairs);write_csv(out/'condition_analysis.csv',conditions);write_csv(out/'memory_accounting.csv',bridge);write_csv(out/'setup_scenario.csv',cross);(out/'analysis_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');figures(out,lat,bridge,pairs,cross);md=report(case,out,*data);html_report(md,out);print(json.dumps({k:v for k,v in metrics.items() if k!='sources'},indent=2))
if __name__=='__main__':main()
