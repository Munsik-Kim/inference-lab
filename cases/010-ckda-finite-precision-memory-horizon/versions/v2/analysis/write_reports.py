"""Write bilingual reports from completed records; never fill missing runs."""
from pathlib import Path
import argparse,csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ARMS=['NATIVE_FP32','UNIFORM_8','UNIFORM_5','LOWRANK_4_8_R2','MIXED_5_6_BUDGET']
def read(p):return json.loads(p.read_text())
def table(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
def num(x,d=2):return '—' if x is None else f'{x:.{d}f}'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
 fresh={(s,k):read(ROOT/'results/fresh'/f'seed{s}'/k/'summary.json') for s in range(3) for k in ARMS}
 precision={(s,k):read(ROOT/'results/precision'/f'seed{s}'/k/'summary.json') for s in range(3) for k in ['D00','D10','D01','D11']}
 timing={s:read(ROOT/'results/timing'/f'seed{s}'/'timing.json') for s in range(3)}
 histories=[read(p) for p in sorted((ROOT/'results/historical').glob('seed*/*/receipt.json'))]
 if len(histories)!=19 or not all(x['status']=='PASS' for x in histories):raise ValueError('Missing/failed historical parity')
 from source.metrics import paired_summary
 pairs=[]
 for s in range(3):
  for j,(a,b) in enumerate([('LOWRANK_4_8_R2','MIXED_5_6_BUDGET'),('LOWRANK_4_8_R2','UNIFORM_5'),('UNIFORM_8','NATIVE_FP32')]):
   pairs.append(paired_summary(fresh[s,a],fresh[s,b],bootstrap_seed=60101+100*s+j))
 freshtable=table(['Seed','Arm','Empirical T0.05','Supported T0.05 ≥','RMST0','Terminal /1024'],[
  [s,k,x['empirical_T05_all_tokens'],x['supported_T05_grid'],num(x['RMST0']),x['terminal_count']] for (s,k),x in fresh.items()])
 pairtable=table(['Seed','Candidate − baseline','Δ RMST0','Pointwise 95% interval'],[
  [r['model_seed'],r['candidate']+' − '+r['baseline'],num(r['RMST0_delta']),f"[{num(r['pointwise_95_CI'][0])}, {num(r['pointwise_95_CI'][1])}]"] for r in pairs])
 byt=read(ROOT/'results/runtime_byte_ledger.json')['rows'];bytes0=[x for x in byt if x['model_seed']==0]
 bytestable=table(['Arm','Stream B','Shared B','Total N1 / N16 / N128'],[[x['arm'],x['v2']['per_stream_persistent_bytes'],x['v2']['shared_bytes'],' / '.join(str(x['v2']['total_bytes'][str(n)]) for n in [1,16,128])] for x in bytes0])
 timetable=table(['Seed']+ARMS,[[s]+[num(float(np.median([r['ms_per_group_token'] for r in timing[s]['rows'] if r['arm']==k])),4) for k in ARMS] for s in range(3)])
 ptable=table(['Seed','Mode','RMST0','Empirical T0.05','S128 / S512','Token accuracy'],[
  [s,k,num(r['RMST0']),r['empirical_T05_all_tokens'],' / '.join(num(next(x['survival'] for x in r['grid_rows'] if x['horizon']==h)*100,2)+'%' for h in [128,512]),num(r['token_accuracy']*100,2)+'%'] for (s,k),r in precision.items()])
 mainpairs=[r for r in pairs if r['baseline']=='MIXED_5_6_BUDGET']
 gains=sum((fresh[s,'LOWRANK_4_8_R2']['supported_T05_grid'] or 0)>(fresh[s,'MIXED_5_6_BUDGET']['supported_T05_grid'] or 0) for s in range(3))
 means=' / '.join(num(r['RMST0_delta']) for r in mainpairs)
 terminals=sum(x['terminal_count'] for x in fresh.values())
 introko='수치적으로 실패한 recurrent stream의 상태·커서·난수를 저장하고, 새 프로세스에서도 종료 상태를 유지하는 codec를 구현했습니다. 같은 세 학습 체크포인트에서 저비트 저장 방식과 최초 판독 실패까지의 길이를 새 입력으로 비교했습니다.'
 introen='Implemented a recurrent-state codec that preserves numerical failure, cursor and RNG state across a fresh-process restart. The same three trained checkpoints were evaluated on new inputs to compare packed storage budgets with the length of continuously correct symbolic readout.'
 linksko='[전체 보고서](REPORT.ko.md) · [스칼라 재검산과 실행](REPRODUCTION.md) · [고정 설계](protocol_v2.json) · [상태 구현](source/online_v2.py)'
 linksen='[Full report](REPORT.md) · [Audit and reproduce](REPRODUCTION.md) · [Frozen design](protocol_v2.json) · [State implementation](source/online_v2.py)'
 (out/'README.ko.md').write_text(f'# Case010 v2 — 실패 상태를 보존하는 recurrent-state 재시작\n\n[English](README.md)\n\n{introko}\n\n{linksko}\n\n- 실행 계약: 정상·수치 실패·stochastic 상태의 저장/재시작을 검사했습니다. 기호 오답은 종료시키지 않습니다.\n- 저장 비용: 모든 방법에 stream당 9 B의 실패 정보를 추가했습니다. N128에서 rank2는 292,633 B, 혼합 5/6비트는 292,469 B입니다.\n- 새 입력: 각 체크포인트당 1024개 × 2048 tokens, 고정된 5개 설정. Rank2가 혼합 기준보다 더 큰 5% 위험 신뢰지원 길이를 보인 체크포인트는 {gains}/3입니다. 평균 연속 정답 길이 차이는 {means} tokens입니다.\n- 정밀도 진단: 별도 32개 입력에서 FP32/FP64 네 경로의 예측과 최초 실패는 같았습니다.\n\n![새 입력의 최초 실패 생존곡선](figures/fresh_survival.png)\n\n실패 상태의 보존은 실행 계약의 개선입니다. 기억 길이의 개선은 [동일 예산 결과](REPORT.ko.md#실험-결과)로 따로 판단합니다. 학습·GPU 실행·원격 게시를 추가하지 않았습니다.\n\n출처와 Codex 지원: [NOTICE](NOTICE.md).\n')
 (out/'README.md').write_text(f'# Case010 v2 — Restarting a recurrent stream without losing its failure\n\n[한국어](README.ko.md)\n\n{introen}\n\n{linksen}\n\n- Execution contract: finite, terminal and stochastic cache restarts are tested. A wrong symbolic label does not terminate a stream.\n- Storage: every arm adds 9 B of failure metadata per stream. At N128, rank2 uses 292,633 B and mixed 5/6 uses 292,469 B.\n- Fresh inputs: 1024 sequences × 2048 tokens per checkpoint, five frozen arms. Rank2 has a larger supported 5%-risk horizon than mixed in {gains}/3 checkpoints. Its paired mean consecutive-correct length differences are {means} tokens.\n- Precision diagnostic: the four FP32/FP64 paths make identical predictions and first failures on 32 separate diagnostic sequences per checkpoint.\n\n![Fresh first-error survival](figures/fresh_survival.png)\n\nPersisting failure improves the execution contract. A memory-horizon improvement is a separate [equal-budget result](REPORT.md#results). No new training, GPU run or remote publication occurred.\n\nAttribution and Codex assistance: [NOTICE](NOTICE.md).\n')
 common_refs='[NOTICE](NOTICE.md) · [unchanged v1 sources and hashes](provenance/v1_source_identity.json) · [upstream/checkpoint provenance](provenance/v1_checkpoints.json) · [protocol freeze](provenance/protocol_freeze.json)'
 ko=fr'''# Case010 v2 — 실패 상태 보존과 동일 저장 예산의 판독 길이

[English](REPORT.md) · [처음으로](README.ko.md)

{introko}

## 목차

[배경](#배경) · [검증 질문과 가설](#검증-질문과-가설) · [이론](#이론) · [방법](#방법) · [실험](#실험) · [실험 결과](#실험-결과) · [결과 분석](#결과-분석) · [결론](#결론) · [레퍼런스](#레퍼런스)

## 배경

v1의 numerical failure mask는 함수 호출 안에만 있었습니다. 상태 bytes에 그 이력이 없어서, 새 호출은 실패 후 저장된 0 placeholder를 정상 상태로 읽을 수 있었습니다. v2는 마지막 정상 body를 유지하고 실패 원인·최초 write 위치를 직렬화합니다. 기존 실험의 원자료와 판정은 수정하지 않았습니다.

## 검증 질문과 가설

1. 정상·실패·확률적 반올림 상태를 새 프로세스로 옮겨도 예측과 최종 bytes가 같은가?
2. 같은 FP32 계수 표현값에서 recurrence/readout의 산술 정밀도만 바꾸면 native 판독 수명이 달라지는가?
3. 새 입력에서 rank2 residual이 같은 N128 저장 상한을 쓰는 혼합5/6비트보다 5% 위험 판독 길이를 늘리는가?

후보 목록은 이미 본 v1 관측을 참고한 후속 설계입니다. 새 TEST로 rank·기저·코덱·체크포인트를 고르지 않았습니다. [protocol_v2](protocol_v2.json)에 입력bytes와 비교·구간 계산을 고정했습니다.

## 이론

물리 좌표 recurrence는 $S_t=A_tS_{{t-1}}+B_t$입니다. 보정 후보는 자신의 저장 표현 $Z+\hat R$를 전체 전이에 전달하고, 갱신값에서 생긴 residual을 다시 저장합니다. 정확한 full residual의 transport 항등식은 일반 corrected-state feedback의 성질이며 새 정리가 아닙니다. 정확한 affine 갱신에서 residual 표현 오차를 eta_t로 쓰면 e_t=A_t e_(t−1)+eta_t입니다. 유한 산술에서는 같은 dtype이어도 계산 반올림의 차이를 eta_t에 포함해야 하며, 계수를 바꾸면 별도 오차항이 필요합니다. Rank 제한·유한 산술·MLP 판독을 함께 쓰는 이번 결과에 무한 기억이나 보편적 비트 하한을 부여하지 않습니다.

$\tau$는 첫 채점 group token의 오답 또는 INVALID 시점입니다. 한번 틀린 뒤 정답으로 돌아와도 survival에는 복귀하지 않습니다. $S(t)=P(\tau>t)$이고 RMST0는 첫 실패 전 연속 정답 token 수의 평균입니다. 실패 token은 제외하며 관측 끝까지 맞으면 2048을 기여합니다. Token accuracy는 회복을 포함하는 별도값입니다.

## 방법

S3의 6개 원소를 균일하게 뽑고, 별도 정수 곱셈표로 왼쪽 누적곱 정답을 계산합니다. 기존 single-layer CKDA (d_model 48, 12 heads, head dimension 16), 세 최종 체크포인트, upstream `ef9d108…`를 그대로 사용했습니다. 새 학습 0회입니다. Native도 동일한 FP32 token table을 사용하는 순차 물리 좌표 경로이며 fused CUDA 실행이 아닙니다. CAL 64 × 32 (seed1001)의 기저·중요도를 재사용했습니다.

상태 header는 little-endian cursor 8 B + terminal code 1 B + 최초 terminal write 8 B입니다. ACTIVE는 최초 위치 UINT64_MAX를 씁니다. BOS는 write 0, 첫 채점 token은 write 1입니다. 성공·종료 no-op 모두 cursor를 증가시킵니다. 종료 이후 state/residual/RNG body는 바뀌지 않습니다.

유한 logits의 오답은 계산을 계속합니다. 상태가 유한한데 readout만 비유한하면 그 출력만 −1로 남기고 계속합니다. 재구성·transition·저장 범위·새 상태 판독의 수치 실패는 해당 행만 종료하며 state/residual/RNG 갱신을 모두 되돌립니다. 잘못된 checkpoint·shape·프로그래밍 예외·프로세스 장애는 실험 실행 오류로 따로 다룹니다. Gold와 tau는 evaluator 기록에만 있고 codec 입력에 들어가지 않습니다.

공유 config·basis·token table과 각 stream의 codes·scales·RNG·header를 실제 serialize해 계산했습니다. 혼합 기준은 모두 5 bit로 시작해 기존 CAL 점수 순서대로 29개 head/key channel을 6 bit로 올렸습니다. 다음 승격이 rank2의 N128 상한을 넘기므로 그 전에 멈췄습니다. Uniform8/native는 고정밀 참고군입니다.

{bytestable}

모든 seed의 크기는 같습니다. N=128은 저장 예산의 동시 stream 가정이며 표본수 1024나 실제 평가 batch 1024와 다릅니다. N1/16/128의 실제 buffer 길이도 [별도로 대조](results/ledger_validation.json)했습니다. Mixed는 더 작은 shared 비용과 조금 큰 stream payload를 쓰므로 N1024에서도 같은 cap에 들어간다는 주장은 하지 않습니다. 원래 공통 가중치 값 226,120 B와 checkpoint 파일 233,203 B는 별도입니다. v2는 stream당 9 B, 공유 metadata 934 B를 더 씁니다. 혼합 기준의 v1-envelope 값은 동일한 새 map에 옛header를 적용한 산술 대조이며 과거 측정 arm이 아닙니다. 한 FP32 임시 state 배열은 N1024에서 12,582,912 B이며 여러 임시 배열이 공존합니다. 실제 process RAM peak/GPU VRAM은 측정하지 않았습니다.

## 실험

- **[A] v1 재계산:** toy 125, learned 78, supplement 12. 원래 26-arm/family546와30-arm/joint630을 분리했습니다.
- **[B] 구현 대조:** 합성 fault injection, fresh-process bytes/RNG 검사, 과거 16개 TEST × 2048의 4 arm × 3 checkpoint, DEV 추가 3 arm, 실제 실패 4개 입력. 합성 실패는 실제 실패 빈도에 넣지 않았습니다.
- **[C] 새 평가:** fresh seed 50101의1024 × 2048 입력을 5 arm × 3 checkpoint가 공유합니다. 별도 seed 40101의32 × 2048 입력은 정밀도·trace 진단용입니다.

Fresh 신뢰계산은 13개 고정 grid × 5 arm × 3 checkpoint = 195 family, alpha 0.05의 exact one-sided binomial 상한입니다. Grid 밖 경험적 곡선은 같은 동시 신뢰하한이 아닙니다. 실패 0/1024의 상한은0.0080424여서 보조 1% 위험 하한도 표본수상 가능하지만, 실제 실패수로 다시 판단합니다. Paired RMST 구간은 각 checkpoint 안에서 5000회 입력 단위 bootstrap한 pointwise 구간입니다. 세 checkpoint의 표본을 합치지 않았습니다.

평가 job은 runtime과 누적 tau·INVALID·출력 구간을 별도 checkpoint로 저장합니다. 임시 디렉터리 inventory를 검증한 뒤 같은 filesystem에서 rename합니다. 이는 atomic visibility이며 fsync 기반 crash durability 보장은 아닙니다. 실제 runtime payload는 private에 두고 공개본에는 초기 0 payload, 최종hash와 compact예측을 남겼습니다.

## 실험 결과

### [A] 역사적 v1

원래 26-arm family에서는54개 budget비교 중 2개에서 supported horizon이 컸고, 더 강한 혼합 비교군을 넣은 30-arm/joint630에서는0/54였습니다. 두 기록을 섞지 않았습니다. Native의 모든token 경험적T0.05는139/108/144였으며, 원래 7개 grid 표의128/64/128과 다른 계산입니다. [전체 재계산](results/historical-reanalysis/HISTORICAL_ANALYSIS.ko.md)과 [역사적 그림](figures/historical_v1_30arm_budget.png)을 별도로 제공합니다.

### [B] 실패와 재시작

프롬프트의 계약을 바탕으로 만든 v1 합성 probe에서 실패 stream이 재시작 뒤 정상 출력으로 돌아오는 불일치를 재현했습니다. 요청된 독립 검토ZIP/원래probe 파일은 찾지 못했으므로 그 스크립트를 실행했다고 기록하지 않았습니다. 실제 체크포인트 과거 입력의 19개 대조가 통과했습니다. 정상경로는 v1/v2 예측·body가 같고, v2는 split/fresh-process 예측·최종 bytes가 같습니다. 원래 cut 1024 뒤에 발생한 Uniform3의 실패도 빠뜨리지 않도록, 별도 고정 부록에서 네 실제 실패 직전·직후를 다시 직렬화했습니다. 실패 직후 새 프로세스로 넘긴 4개 검사도 통과했습니다. [실패 경계 부록](results/historical-failure-boundaries/index.json). 실제 실패 후 v1과 다른 body는 이전 정상 body+terminal flag를 유지하는 의도한 변경입니다. [실패 probe](results/v1_failure_probe_from_request.json), [역사적 대조](results/historical/seed0/index.json), [회귀 테스트](tests/test_online_v2.py).

### [C] 고정 계수 정밀도 진단

{ptable}

D00은 FP32/FP32, D10은 FP64 recurrence 뒤 상태를 FP32로 내려 원래 판독, D01은 FP32 state를 올린 명시적 FP64 판독, D11은 FP64/FP64입니다. q/k/v/alpha/beta/g/e를 만드는 계수 경로는 FP32 표현값으로 고정했습니다. D01/D11은 그 값과 고정 weight를 승격한 뒤 출력 sigmoid(g), 출력 projection, 정규화와 MLP 판독을 실제 FP64로 계산합니다. 계수 생성의 재계산과 판독 산술의 승격을 구분하며, dtype trace와 FP32 reference parity를 검사했습니다. 네 모드는 각 seed에서 전체 예측·tau가 같았습니다. [진단 검산](provenance/diagnostic_readback.json).

![정밀도별 생존과 현재 token 정확도](figures/precision_survival_accuracy.png)

### [C] Fresh 저장 방식 비교

{freshtable}

{pairtable}

Rank2−mixed 평균 차이는 seed0/1/2에서 {means} tokens입니다. 더 높은 supported 5% 길이는 {gains}/3 checkpoint에서 관측됐습니다. 이는 서로 다른 지원하한의 기술통계 비교이며 방법 간 위험 차이의 다중비교 확증검정이 아닙니다. Fresh 15개 cell 전체에서 수치 terminal은 {terminals}/15,360 stream-arm 기록입니다. 이 15,360개를 독립 입력이나 독립 학습 모델로 세지 않습니다.

![Fresh 생존](figures/fresh_survival.png)
![동일 예산과 판독 길이](figures/fresh_budget_horizon.png)
![Paired RMST](figures/fresh_paired_rmst.png)

### 후기 상태와 비용

![Rank2 시간별 진단](figures/rank2_time_diagnostics.png)

그림의 실패 전·후 곡선은 시점마다 구성원이 달라지는 조건부 평균입니다. t128에서 seed0/1/2의 전·후 표본 수는 31/1, 28/4, 31/1이고 t512에는 모두 0/32입니다. 각 지표의 유효 표본 수는 [시간별 표](results/diagnostic-summary/trace_fixed_times.csv)에 있습니다. Rank2의 큰 후기 norm·오차·projection 손실은 첫 판독 오류 뒤에도 계산을 계속한 값입니다. Scratch 0과 terminal 이후 값은 유효 norm에 포함하지 않았습니다. [진단 원자료와 정의](results/diagnostic-summary/summary.json)는 native/shadow 오차와 자기 write-back 오차를 구분합니다. Gold margin은 evaluator에서만 계산했습니다. 첫 오답·수치 실패·큰 norm의 선후 관계는 관측이며 단독 인과 증명이 아닙니다.

다른 본 작업의 평가가 끝난 뒤 CPU 2 threads로 16 × 128, 세 반복을 순차 측정했습니다. 아래는 **complete-call ms/group-token의 중앙값**입니다. BOS 비용은 call에 포함되고 분모는 16 × 128입니다. Transition·projection·packing·상태 검사·원래 readout을 포함하며 loading·I/O·gold/shadow는 제외합니다.

{timetable}

[시간 원기록](results/timing/seed0/timing.json)은 active update와 terminal no-op 횟수를 나눕니다. 이는 Python/NumPy/PyTorch CPU 경로의 비용이며 단일 요청 latency나 GPU 가속 수치가 아닙니다. OS 배경 부하를 완전히 통제한 측정도 아닙니다.

## 결과 분석

v2의 주된 소프트웨어 성과는 종료 의미가 호출 경계를 넘어 유지되는 것입니다. 이것을 모델 기억 수명의 개선으로 세지 않습니다. 같은 값의 FP32/FP64 경로가 이번 32개 입력에서 같은 label을 냈다는 관측은 native의 이른 첫 실패를 산술 precision 하나로 설명하기 어렵게 하지만, 학습 부족·표현력 한계·필수 비트 하한을 단독으로 입증하지 않습니다.

평균 연속 정답 길이, 5% 위험 길이, token accuracy, 후기 상태 norm, CPU 비용은 서로 다른 목표입니다. Native도 장기 외삽에서 실패하므로 양자화 후보의 모든 오답을 저장 오류 탓으로 돌리지 않습니다. 기존 v1의 joint 0/54는 그대로이며, 새 family와 표본수로 바뀐 신뢰하한 자체를 방법 개선으로 해석하지 않습니다.

## 결론

실패·난수·커서를 보존하는 packed state와 evaluator 재시작 경로를 구현하고, 실제 과거 실패를 재현했습니다. 새 입력에서 같은 저장 상한의 비교와 전체 곡선을 남겼습니다. Rank2의 평균 차이와 지원하한 결과를 함께 제시했으며, 유리한 한 seed나 늦은 token 정확도를 최초 실패 목표 대신 쓰지 않았습니다. 새 학습 0, GPU 실행 0, 원격 게시 0입니다.

## 레퍼런스

{common_refs}

ComplexKDA와 기존 state-quantization/error-feedback 문헌의 대응은 v1 고지를 유지합니다. 모델·학습·기존 코덱은 upstream/v1 구현이고, 이번 기여는 failure-aware adapter, restart 계약과 그 후속 검증입니다. Codex가 구현·검사·보고서 작성을 지원했습니다.
'''
 en=fr'''# Case010 v2 — Failure-preserving restart and equal-budget memory horizons

[한국어](REPORT.ko.md) · [Home](README.md)

{introen}

## Contents

[Background](#background) · [Questions](#questions) · [Theory](#theory) · [Methods](#methods) · [Experiments](#experiments) · [Results](#results) · [Analysis](#analysis) · [Conclusion](#conclusion) · [References](#references)

## Background

V1 kept its numerical-failure mask inside a function call. The stored bytes lacked that history, so restarting could interpret a zero placeholder as active state. V2 retains the last committed finite body and serializes the failure reason and first write index. Original results, manifests and decisions remain unchanged.

## Questions

1. Do finite, terminal and stochastic caches preserve predictions and final bytes across a fresh process?
2. Does increasing recurrence or readout arithmetic precision change native first-error lifetime while holding the represented FP32 coefficients fixed?
3. On new inputs, does rank2 residual storage extend the 5%-risk readable horizon compared with a mixed 5/6-bit baseline under the same N128 cap?

This follow-up menu was informed by already explored v1 observations. Fresh TEST was not used to select rank, basis, codec or checkpoint. [Protocol](protocol_v2.json) freezes actual input bytes and the comparisons.

## Theory

The physical recurrence is $S_t=A_tS_{{t-1}}+B_t$. A corrected candidate transports its own represented state $Z+\hat R$ through the full transition and stores the residual of its own write-back. Exact full-residual transport is an accounting identity for corrected-state feedback, not a new theorem. For an exact affine update and residual representation error eta_t, e_t=A_t e_(t−1)+eta_t. Finite arithmetic also requires the difference in computation rounding to enter eta_t, even at a matched dtype; coefficient changes require additional terms. Rank truncation, finite arithmetic and an MLP decoder prevent treating this experiment as an infinite-memory or universal bit-bound result.

First error $\tau$ is the first scored group-token prediction that is wrong or INVALID. Recovery never restores first-error survival. $S(t)=P(\tau>t)$. RMST0 is the mean number of consecutive correct tokens before failure, excluding the failing token; a right-censored sequence contributes 2048. Current-token accuracy counts later recovery separately.

## Methods

Inputs are uniform draws from all six S3 elements; an independent integer left-multiplication table provides symbolic gold. The original single-layer CKDA (d_model 48, 12 heads, head dimension 16), three final checkpoints and pinned upstream `ef9d108…` are unchanged. New training updates: 0. Native uses the same canonical FP32 token table and sequential physical-coordinate path, not a fused CUDA run. The original CAL 64 × 32, seed 1001, supplies frozen bases and channel scores.

Each stream has an 8-byte little-endian cursor, 1-byte terminal code and 8-byte first-terminal write index. ACTIVE uses UINT64_MAX as the first-write sentinel. BOS is write 0 and the first scored token is write 1. Every consumed input advances the cursor, including terminal no-ops. After failure, state/residual/RNG body bytes remain fixed.

A finite wrong label keeps running. A nonfinite readout with finite state produces −1 only at that step and also keeps running. Numerical reconstruction, transition, representation-range or post-write failures terminate only that row and roll back its entire proposed state/residual/RNG commit. Corrupt checkpoints, shapes, programming exceptions and process/resource failures remain execution errors. Gold and tau belong to evaluator history, never codec inputs.

Actual serialization counts shared configuration, basis and token table plus per-stream codes, scales, RNG and header. The mixed baseline starts at 5 bits and promotes 29 head/key channels to 6 bits in the frozen CAL-score order. The next promotion exceeds the rank2 N128 cap. Uniform8 and native are fidelity/capacity references.

{bytestable}

All seeds have the same sizes. N128 is the concurrent-stream storage scenario, distinct from 1024 independent input sequences and the actual evaluation batch of 1024. [Actual buffers](results/ledger_validation.json) verify N1/16/128 sizes. Mixed trades lower shared cost for a slightly larger stream payload; equal-cap feasibility is not claimed at N1024. Common model parameter values 226,120 B and checkpoint file 233,203 B are separate. V2 adds 9 B per stream and 934 B shared metadata. The mixed baseline's v1-envelope ledger is a size-only construction, not a historical measured arm. One temporary FP32 state array occupies 12,582,912 B at N1024; multiple temporaries coexist. Process RAM peak and GPU VRAM were not measured.

## Experiments

- **[A] Historical recalculation:**125toy,78learned and12supplement records. The original 26-arm/family546 and 30-arm/joint630 views are kept separate.
- **[B] Implementation parity:** synthetic fault injection and fresh-process/RNG checks; four arms on the first 16 historical TEST inputs × 2048 in each checkpoint; three extra DEV arms; four selected actual numerical failures. Injected failures never count as measured CKDA incidence.
- **[C] Fresh evaluation:**1024×2048 inputs from seed 50101, shared by five arms and all three checkpoints. A separate 32 × 2048 cohort from seed 40101 supplies precision and trace diagnostics.

Fresh uncertainty uses 13 declared horizons × 5 arms × 3 checkpoints=195 comparisons, alpha 0.05, with exact one-sided binomial upper bounds. Off-grid empirical curves do not inherit that simultaneous confidence statement. With zero events in 1024, the upper bound is 0.0080424: a secondary 1%-risk bound is possible in principle but still depends on observed failures. Paired RMST intervals use 5000 sequence-level resamples within one fixed checkpoint and are pointwise. Checkpoints are not pooled.

The evaluation job stores runtime bytes separately from cumulative tau, INVALID counts and output boundaries. It validates a temporary-directory inventory before a same-filesystem rename. This promises atomic visibility, not fsync crash durability. Full runtime job checkpoints remain private; the review includes initial zero payloads, final hashes and compact predictions.

## Results

### [A] Historical v1

The original 26-arm family has 2/54 budget comparisons with a larger supported horizon; the stronger 30-arm/joint630 view has 0/54. Native all-token empirical T0.05 is 139/108/144, distinct from the old seven-grid values 128/64/128. [Historical recalculation](results/historical-reanalysis/HISTORICAL_ANALYSIS.ko.md) and its figures remain separate from fresh evidence.

### [B] Failure and restart

A synthetic probe reconstructed from the request reproduces v1 revival after restarting a failed stream. The requested independent-review archive and exact probe were unavailable, so their execution is not claimed. All 19 actual historical comparison cells pass. Healthy paths match v1/v2 predictions and body bytes; v2 split/fresh-process paths match complete predictions and final bytes. A separate frozen coverage appendix also tests all four actual failures immediately before and after their recorded failure writes. All four fresh-process post-failure continuations pass, including Uniform3 whose failure followed the original last interior cut. [Failure-boundary appendix](results/historical-failure-boundaries/index.json). A terminal v2 body differs deliberately from v1 by retaining the previous committed body plus failure metadata. [Probe](results/v1_failure_probe_from_request.json), [historical index](results/historical/seed0/index.json), [regression tests](tests/test_online_v2.py).

### [C] Fixed-coefficient precision

{ptable}

D00 is FP32/FP32, D10 is FP64 recurrence followed by a state cast and the original FP32 readout, D01 uses FP32 recurrence and explicit FP64 readout, and D11 is FP64/FP64. The coefficient construction path is fixed at its represented FP32 q/k/v/alpha/beta/g/e values. D01/D11 promote those values and fixed weights, then evaluate the output sigmoid(g), output projection, normalization and MLP readout in FP64. Coefficient generation is not repeated; readout arithmetic is promoted. Actual dtype traces and explicit-FP32 parity are checked. All four modes have identical complete prediction matrices and tau within each seed. [Readback](provenance/diagnostic_readback.json).

![Precision survival and token accuracy](figures/precision_survival_accuracy.png)

### [C] Fresh packed-state comparison

{freshtable}

{pairtable}

Rank2−mixed RMST0 differences are {means} tokens for seeds 0/1/2. A larger supported 5%-risk horizon appears in {gains}/3 checkpoints. Comparing supported lower bounds is descriptive, not a multiplicity-adjusted test of between-method risk superiority. There are {terminals} numerical terminal events among 15,360 stream-arm records across the 15 cells; these are not 15,360 independent inputs or models.

![Fresh survival](figures/fresh_survival.png)
![Budget and horizon](figures/fresh_budget_horizon.png)
![Paired RMST](figures/fresh_paired_rmst.png)

### Late instability and cost

![Rank2 time diagnostics](figures/rank2_time_diagnostics.png)

Pre/post-failure curves are conditional means over subsets whose membership changes with time. At t128, pre/post sample counts are 31/1, 28/4 and 31/1 for seeds 0/1/2; by t512 they are 0/32 for each seed. [Fixed-time tables](results/diagnostic-summary/trace_fixed_times.csv) report valid counts per metric. Large late rank2 norms, errors and projection losses include continued rollout after the first label error. Terminal scratch zeros do not enter valid norm statistics. [Diagnostic scalars and definitions](results/diagnostic-summary/summary.json) separate error against native/shadow state from self-write-back error. Gold margins are evaluator-only. The ordering of first error, numerical failure and large norm is an observation, not a standalone causal identification.

After the other study evaluation jobs finished, serial CPU 2-thread calls used 16 × 128 inputs and three fixed repeats. The table reports **median complete-call milliseconds per scored group token**. BOS overhead is in the call; the denominator is 16 × 128. Transition, projection, packing, status handling and original readout are included; loading, file I/O and gold/shadow diagnostics are outside the timer.

{timetable}

[Timing records](results/timing/seed0/timing.json) distinguish active updates from terminal no-ops. These are Python/NumPy/PyTorch CPU reference costs, not single-request latency or GPU speedups. System-wide OS background load was not fully controlled.

## Analysis

V2's software improvement is preserving termination across call boundaries. That is separate from increasing model memory horizon. Identical labels under four arithmetic modes on 32 diagnostic inputs make a simple precision-only account insufficient for this observation; they do not establish undertraining, representational impossibility or a necessary bit lower bound.

Mean consecutive-correct length, 5%-risk horizon, current-token accuracy, late state norm and CPU cost are different endpoints. Native also fails long extrapolation, so not every candidate error is attributable to state quantization. Historical joint 0/54 remains unchanged. A new family and sample size change confidence bounds; that numerical change alone is not a method improvement.

## Conclusion

The study implements failure-, RNG- and cursor-preserving packed state and evaluation-job restart, and replays actual historical failures. It retains a fixed-budget comparison on new inputs with complete curves. Rank2's mean effects and supported-horizon outcomes are reported together; neither a favorable checkpoint nor late token accuracy replaces the first-failure objective. New training 0, GPU runs 0, remote publication 0.

## References

{common_refs}

The v1 ComplexKDA, state-quantization and error-feedback references are preserved. The model, training and original codecs are upstream/v1 work; this follow-up implements failure-aware execution and its validation. Codex assisted implementation, checking and report writing.
'''
 (out/'REPORT.ko.md').write_text(ko);(out/'REPORT.md').write_text(en)
if __name__=='__main__':main()
