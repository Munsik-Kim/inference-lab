"""Bilingual local P2 reports from audited saved results; no model execution."""
import argparse,json,math,html
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args();case=Path(a.case);base=case/'followups/p2_core';run=case/'results/p2_r1'
    s=json.loads((run/'summary.json').read_text());r=s['resources'];sf=json.loads((run/'source_freeze.json').read_text());inp=json.loads((run/'input_manifest.json').read_text());checks=json.loads((run/'independent_audit.json').read_text());private=json.loads((run/'private_array_audit.json').read_text())
    methods=['BF16','RTN','COMP_QUARTER','COMP_HALF'];short={'BF16':'BF16','RTN':'RTN','COMP_QUARTER':'QUARTER (0.25)','COMP_HALF':'HALF (0.5)'}
    m={(x['split'],x['method']):x for x in s['method_summaries']};c={(x['split'],x['candidate']):x for x in s['paired_comparisons'] if x['baseline']=='RTN'}
    p=c['EVAL-ID','COMP_QUARTER'];q=c['SHIFT','COMP_QUARTER'];h=c['SHIFT','COMP_HALF']
    def pct(x):return f'{100*x:+.3f}%'
    def ci(x):return f"[{x['CI95'][0]:+.6f}, {x['CI95'][1]:+.6f}]"
    def table(split):
        rows=['| Method | NLL (nats/token) ↓ | PPL ↓ | KL(BF16∥method) ↓ | Argmax flip | ΔNLL vs RTN |','| --- | ---: | ---: | ---: | ---: | ---: |']
        for method in methods:
            z=m[split,method];rows.append(f"| {short[method]} | {z['NLL_token_weighted']:.6f} | {z['PPL']:.6f} | {z['mean_full_vocab_KL']:.6f} | {100*z['argmax_flip_rate']:.3f}% | {z['NLL_vs_RTN']:+.6f} |")
        return '\n'.join(rows)
    contrasts=['| Corpus | Candidate − RTN | Mean ΔNLL | Paired 95% CI | PPL change | Lower / higher / tie docs | Verdict |','| --- | --- | ---: | --- | ---: | --- | --- |']
    for split in ['EVAL-ID','SHIFT']:
        for method in methods[2:]:
            z=c[split,method];contrasts.append(f"| {'PG19' if split=='EVAL-ID' else 'ArXiv'} | {short[method]} | {z['mean_document_delta_NLL']:+.6f} | {ci(z)} | {pct(z['PPL_relative_change'])} | {z['lower_documents']} / {z['higher_documents']} / {z['exact_ties']} | {z['verdict']} |")
    contrast='\n'.join(contrasts);qr=m['EVAL-ID','COMP_QUARTER'];hr=m['EVAL-ID','COMP_HALF'];rt=m['EVAL-ID','RTN'];bf=m['EVAL-ID','BF16']
    diag=[json.loads(x) for x in (run/'CAL_diagnostics.jsonl').read_text().splitlines()];maxgap=max(x['decomposition_identity_relative_gap'] for x in diag);maxsolve=max(x['fit']['relative_solve_residual'] for x in diag)
    cost=f"""| Item | Actual measurement |
| --- | ---: |
| INT4 core files per method | 196 |
| Core bytes: RTN = QUARTER = HALF | 247,733,136 B each |
| Header total / scales total / packed codes total | 7,056 / 27,525,120 / 220,200,960 B |
| BF16 projection tensors, calculated from actual tensors | 880,803,840 B |
| Common unchanged parameters, counted once with tied weights | 311,296,000 B |
| RTN offline construction | {r['stage_seconds_inclusive']['RTN_build']:.3f} s |
| QUARTER offline construction | {r['stage_seconds_inclusive']['COMP_QUARTER_build']:.3f} s |
| HALF offline construction | {r['stage_seconds_inclusive']['COMP_HALF_build']:.3f} s |
| Cumulative worker wall, all 3 processes | {r['cumulative_worker_wall_seconds']:.3f} s / 10,800 s |
| Torch peak allocated / reserved | {r['peak_Torch_allocated_bytes']:,} / {r['peak_Torch_reserved_bytes']:,} B |
| Maximum observed post-forward whole-device used sample | {r['max_observed_post_forward_device_used_bytes']:,} B |
| Native whole-model forwards / fresh metric rows | {r['all_full_model_forwards']:,} / 320 |
"""
    ko=f"""# 약한 양자화 보정은 전체 모델에서도 도움이 될까?

Case014 P2 · 2026-10-15 · `COMPLETED_P2_CORE` · [English](REPORT.md)

## 한눈에 보기

양자화된 모델이 실제로 전달하는 입력을 다시 측정하고, 그 입력에 맞춘 약한 보정을
동일한 INT4 형식에 저장하는 도구를 구현했습니다. 각 블록의 q/k/v → attention → o →
residual/norm → gate/up → down 순서를 원래 모델 전체 forward로 따라가며 구축합니다.

Qwen3-0.6B의 전체196 projection에 적용한 QUARTER는 새 PG19 64책에서 RTN보다
평균 NLL이 **{p['mean_document_delta_NLL']:+.6f} nats/token**, 95% CI **{ci(p)}**로 낮았습니다.
PPL은 **{rt['PPL']:.3f} → {qr['PPL']:.3f} ({pct(p['PPL_relative_change'])})**였고, 64책 모두 NLL이 낮았습니다.
반면 ArXiv16개에서는 QUARTER의 점추정이 악화했고 구간에0이 포함됐습니다. HALF의
ArXiv NLL은 RTN보다 높은 구간으로 확인됐습니다. **PG19 이득과 도메인 전이 한계를 함께 관측했습니다.**

RTN·QUARTER·HALF의 실제 projection 저장량은 각각 **247,733,136 bytes**로 같습니다.
평가는 저장본을 BF16으로 복원한 dense 품질 경로입니다.

[원행·집계](../../results/p2_r1/summary.json) · [독립 CPU 감사](../../results/p2_r1/independent_audit.json) ·
[재현 명령](REPRODUCE.md) · [코드](source/run_p2.py) · [간단 결과표](demo/index.html)

목차: [질문](#질문) / [구현](#구현) / [데이터·실험](#데이터실험) / [결과](#결과) /
[분석](#분석) / [결론](#결론) / [재현·출처](#재현출처)

## 질문

P1b에서 여섯 개 독립 개입의 국소 SSE가 개선됐지만, 이들을 합친 전체 모델의 성적은
측정하지 않았습니다. P2는 그 결과를 본 뒤 설계한 후속입니다. 기존 FULL 제안 대신
고정0.25를 주 후보로, 고정0.5를 보조 후보로 등록했습니다. 새 selector를 적합하지 않았습니다.

| 확인할 질문 | 이번 관측 |
| --- | --- |
| 자체 경로의 전층 약한 보정이 같은 형식 RTN보다 새 PG19 NLL을 낮추는가? | QUARTER의 등록된 주 비교 `LOWER_NLL` |
| 국소 P1b 이득이 전층에서도 유효한가? | 고정 CAL32·모델·이번 PG19 표본에서 전층 품질 이득을 직접 관측. 여섯 국소 효과를 합산한 예측값은 아님 |
| 다른 출처에도 유지되는가? | ArXiv QUARTER는 점추정 악화·구간 불확정, HALF는 NLL 악화 |

## 구현

BF16은 원형 모델입니다. RTN은 원래196개 projection을 그대로 INT4로 저장합니다.
QUARTER/HALF는 각자 원형에서 시작하고, 현재까지 바뀐 자신 경로의 CAL 입력 Z와
완전 BF16 teacher의 X를 사용합니다. q/k/v와 gate/up은 같은 입력의 Gram factor만
그룹 안에서 공유합니다. 다른 강도의 Z·K를 공유하지 않습니다.

**기준 구현은 native CausalLM 전체 forward입니다.** 그룹마다 모델 전체를 실행하므로
attention·QK norm·RoPE·residual·MLP를 생략한 블록 단축 경로가 없습니다. 실제 hook으로
qkv/gate-up 공통 입력, 같은 블록 q 변화의 o 입력 전파, 미래 블록의 비영향을 확인했습니다.

<details><summary>적합·저장·오차 분해의 정확한 계약</summary>

W[out,in], X/Z[in,tokens]. 원 BF16 W의 표현값을 FP64로 올립니다.
G=ZZᵀ, λ=0.001 tr(G)/d_in, (G+λI)Kᵀ=[W(X−Z)Zᵀ]ᵀ를 CPU FP64
Cholesky와 triangular solve로 풉니다. full-input Gram, output-row chunk128입니다.
Wstar=W+αK이며 원래 W에 한 번만 더합니다. QUARTER/HALF는 각각196개의 새 K와
112번의 factorization을 갖습니다. 첫 블록 qkv는 X=Z, K=0, 저장 code/scale도 RTN과 같습니다.

FP32(Wstar)를 input-axis group64·output-row/group FP32 absmax/7로 양자화합니다.
signed[−8,7], nearest-even, 실제 zero group만 scale1, low-nibble-first, header36B입니다.
모든 파일은 실제 pack→새 파일→두 decoder로 reload한 후 BF16(What)를 설치했습니다.
runtime에 K/residual을 별도 실행하지 않습니다.

CAL에서 A=W(X−Z), C=αKZ, B=(What−Wstar)Z, B0=(Q(W)−W)Z입니다.
Δ=||A−C−B||²−||A−B0||²=T_inherited+T_cancellation+T_requant.
B0는 **각 방법 자신의 현재 Z에서의 국소 alpha0 대조**입니다. 별도 전체 RTN 모델의
오차가 아닙니다. 392개 CAL 분해의 최대 상대 gap은 {maxgap:.3e}, 최대 solve residual은
{maxsolve:.3e}로 사전1e−10 기준을 통과했습니다. 고정 first-o SMOKE의 실제 BF16
module output/cast/arithmetic residual은 [별도 기록](../../results/p2_r1/smoke_matrix_verification.json)에 있습니다.

</details>

embedding/lm_head의 tie, norm·RoPE·attention·bias 정책은 원형을 유지했습니다.
모든 조건은 BF16/batch1/seq256/SDPA/eval/no_grad/use_cache=False, TF32 및
BF16/FP16 reduced reduction OFF, native 전체256위치 head를 사용했습니다.

## 데이터·실험

| 용도 | 원본 단위·수 | 이번 사용 |
| --- | --- | --- |
| CAL-FIT | 원 P1 PG19 32책 | 기존 token·offset 그대로, P2 자체 경로 K만 새 적합 |
| EVAL-ID | 새 PG19 64책 | 16,320 valid next tokens/method |
| SHIFT | 새 ArXiv 16 content group | 4,080 valid next tokens/method, ID와 별도 |
| SMOKE | 기존 별도1개 | 명시적 재사용, 평가 분모 제외 |

기존209개 고유 입력(PG19 177책·ArXiv32 group)을 확인 가능한 manifest 합집합으로
제외했습니다. 새 source 전송은0 bytes입니다. PG19의 canonical book ID·content/token
hash 및 정규화 창, 5-gram 중복을 검사했습니다. 창 비교19,880회의 최대 Jaccard는
{inp['near_duplicate_check']['maximum_Jaccard']:.6f}였습니다. 전체 원문 근접 중복과 사전학습 미노출은 확인하지 않았습니다.
Pile는 article ID가 없어 content-group 기준입니다.

spec → 입력 동결 → CPU/SMOKE → RTN → HALF → QUARTER → 전체 artifact 동결
순서였습니다. **모든 저장본을 동결할 때 fresh forward는0회**였습니다. 그 뒤 BF16 →
QUARTER → HALF → RTN 순으로 새 ID·SHIFT를 평가했습니다. 데이터 seed6101521,
구축/평가 순서6101522, bootstrap6101523입니다. CAL 밖 입력은 fit 함수가 거절합니다.

add_special_tokens=False, chat/BOS/EOS 추가 없이 문서당256 input tokens입니다.
위치0..254의 원문 다음255 tokens를 NLL/argmax로 평가하고, 기존16개 고정 위치에서
full-vocabulary KL(BF16∥method)을 측정했습니다. 자유 생성이나 agent 성공률은 아닙니다.

## 결과

### 새 PG19 64책

{table('EVAL-ID')}

![PG19 NLL/PPL](../../figures/p2_r1/01_pg19.png)

QUARTER는 BF16 대비 RTN의 추가 NLL 손실 중 {100*qr['recovered_RTN_loss_fraction']:.2f}%를 회복했습니다.
HALF는 {100*hr['recovered_RTN_loss_fraction']:.2f}%였습니다. 두 후보 모두 BF16보다 NLL이 높습니다.
이 회복률의 분모는 RTN−BF16 NLL gap이며, 모델 정확도 회복률과 다릅니다.

### 새 ArXiv 16개

{table('SHIFT')}

![ArXiv NLL/PPL](../../figures/p2_r1/02_arxiv.png)

### 문서 paired 비교

{contrast}

주 비교는 PG19 QUARTER−RTN 한 개입니다. 나머지는 보조 pointwise 구간입니다.
문서 쌍을 함께10,000회 재표집했고, 원 CAL32·모델·고정 강도에 조건화한 문서
불확실성입니다. 새로운 CAL·모델 모집단까지의 구간은 아닙니다. PPL은 exp(totalNLL/count)이며
문서 PPL 평균을 사용하지 않습니다. PPL1%는 프로젝트 참고값이고, 이번 주 비교의 크기는
{100*p['PPL_relative_change']:.3f}%입니다. HALF의 PG19 점추정이 조금 낮아도 주 후보를 교체하지 않았습니다.

![전체 문서 차이](../../figures/p2_r1/03_paired_document_nll.png)

### 저장량·offline 비용

{cost}

원 모델의 공통 부분은 별도 장부이며 core 파일만으로 standalone 모델이 되지 않습니다.
구축 시간에는 native CAL capture·FP64 적합·양자화·분해 진단·I/O가 포함됩니다.
전체 worker는 main {r['worker_wall_seconds_base']:.3f}초(새 프로세스 reload 포함)와 등록된
보조 SMOKE {r['supplemental_smoke_worker_seconds']:.3f}초의 합입니다. 데이터 준비 {inp['data_prepare_seconds']:.3f}초와 CPU 감사/그림/문서화는 별도입니다.
내부 단계 시간은 포개지는 값이므로 다시 합산하지 않습니다. Torch peak와 장치 사용
표본은 다른 지표이며, 장치 표본을 연속 Windows 전체 peak로 부르지 않습니다.

![저장량과 offline 구축비용](../../figures/p2_r1/04_storage_and_offline_cost.png)

## 분석

PG19에서는 약한 전층 보정이 같은 INT4 bytes에서 RTN 손실을 상당 부분 줄였습니다.
ArXiv에서는 QUARTER의 KL이 RTN보다 조금 작아도 gold NLL은 점추정상 높았습니다.
HALF는 PG19에서 더 낮은 점추정과 함께 ArXiv에서 더 큰 손실을 보였습니다.
teacher 분포 근접도·정답 NLL·도메인 전이는 구분해서 볼 필요가 있습니다.

이 결과는 PG19 CAL에 맞춘 보정의 도메인 의존과 일관됩니다. 개별 층의 유일한
원인이나 보편 법칙을 입증한 것은 아닙니다. CAL의 세 항 분해는 국소 계산을
검산하며 전층 NLL 기여율로 더하지 않습니다. P1b의 여섯 효과와 이번 전층 효과도
측정 단위가 다릅니다. [P1b 검토 반영](REVIEW_RESPONSE.ko.md)에 기존 분모·한계를 정리했습니다.

## 결론

등록된 PG19 주 질문은 `LOWER_NLL`로 답할 수 있습니다. 동일 저장 형식에서 고정0.25의
전층 자체 경로 보정이 실제로 이득을 냈습니다. SHIFT의 결과를 고려하면 즉시 다른
도메인·모델·agent로 확대할 근거는 제한적입니다. P2 핵심 비교는 결과와 관계없이
4조건·320문서 측정 모두 완료했고, 누락/실패 cell은0개입니다.

| 판정 축 | 완료 범위 |
| --- | --- |
| 구현·저장·전층 품질 | 196 projections × 3 quantized models, 588 files, 392 K fits, PASS |
| 주 통계 비교 | PG19 QUARTER−RTN: LOWER_NLL |
| SHIFT | QUARTER INCONCLUSIVE with higher point estimate; HALF HIGHER_NLL |
| 검산 | 새 고유 P2 tests9 PASS; 기존15+P1b10은 회귀 재실행. 독립 scalar 감사320행·10비교·20구간 변환 PASS |
| 별도 실제 검증 | 새 GPU 프로세스588 files 독립 decode·4모델 SMOKE full-logit SHA 일치; CPU full SMOKE 행렬3건·고정 CAL normal equation8건 PASS |
| 보존 | 기존 Case014423파일·private4482파일·repo tracked6351파일·model6파일(전체 weight 포함)·P1b ZIP hash 일치, index/HEAD/remote 변경 없음 |
| 실행 경계 | packed storage → dense BF16 품질 reference. TTFT/TPOT/GPU busy/native packed speed 미측정 |
| 제외 | P3·4B·agent·새 selector·패키지 변경·Git stage/commit/push/PR/merge·Pages/Drive 미실행 |

## 재현·출처

[실제 CLI와 CPU/GPU 범위](REPRODUCE.md), [등록 spec](specs/p2.json),
[원자료](../../results/p2_r1/metrics/), [판정](../../results/p2_r1/decision.json),
[보존 검사](../../results/p2_r1/preservation.json), [자원 기록](../../results/p2_r1/resource_summary.json).

작업 branch는 research/case014-quantization-compensation, Git base/HEAD는
09194c3f2e2c991feff5b6abab0cf8ab81a07052입니다. 실행 source SHA256은
`{sf['execution_source_sha256']}`입니다. source tree hash·Git commit·모델 revision·개별
artifact hash를 구분합니다. 모델 revision은 c1899de289a04d12100db370d81485cdf75e47ca,
전체 local weight file SHA256은 f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b입니다.
이는 실제 로컬 identity 검증이며 새 공식 digest 또는 D 원본 대조는 아닙니다.

Guo의 [논문v1](https://arxiv.org/html/2610.16020v1)과 과거
[source audit](../../references/source_audit.json)은 이론·선행 실험의 출처입니다.
이번 P2는 프로젝트의 ridge/RTN 자체 구현이며 저자 공식 QEP 전체 재현이 아닙니다.
문헌 HTML/PDF 확보·절 대조는 기존 P0 기록을 인용하며 P2에서 새 문헌 실험으로 세지 않습니다.
Codex가 구현·분석·문서화를 보조했습니다. 사람의 독립 심사를 거쳤다고 표시하지 않습니다.

검토 ZIP은 코드·경량 원행·spec·manifest·보고서·그림을 포함합니다. model/K/INT4
weights/X/Z/logits/원문은 제외합니다. 공개 묶음의 scalar 감사는 외부 GPU 계산 자체의
재현과 다릅니다. 로컬 private 경로와 전체 재현재료 inventory는 ZIP 밖에 보존합니다.
"""
    en=f"""# Does weak quantization compensation help a whole model?

Case014 P2 · 2026-10-15 · `COMPLETED_P2_CORE` · [한국어](REPORT.ko.md)

## At a glance

This tool refits compensation on each quantized model's current input trajectory, writes the
result into the same INT4 format, reloads it, and measures actual whole-model outputs.
On64 fresh PG19 books, Qwen3-0.6B QUARTER lowered gold NLL against RTN by
**{p['mean_document_delta_NLL']:.6f} nats/token (95% CI {ci(p)})**. PPL fell from
**{rt['PPL']:.3f} to {qr['PPL']:.3f}, {pct(p['PPL_relative_change'])}**. All64 books had lower NLL.
On16 fresh ArXiv content groups, QUARTER had a higher point estimate with a CI crossing zero;
HALF had higher NLL with its auxiliary CI above zero. Domain transfer remains limited.

All three quantized models have **247,733,136 actual projection-core bytes**. Runtime is
stored INT4 decoded into dense BF16 weights. [Raw results](../../results/p2_r1/summary.json) ·
[Independent audit](../../results/p2_r1/independent_audit.json) · [Reproduce](REPRODUCE.md) ·
[Code](source/run_p2.py) · [Small results table](demo/index.html).

Contents: [Questions](#questions) / [Implementation](#implementation) / [Data and experiment](#data-and-experiment) /
[Results](#results) / [Analysis](#analysis) / [Conclusion](#conclusion) / [Reproduction and sources](#reproduction-and-sources).

## Questions

P1b measured six independent local interventions, rather than a complete compensated model.
P2 was designed after observing P1/P1b. Fixed QUARTER replaces the earlier FULL proposal as
the primary candidate; fixed HALF is auxiliary. No selector, strength grid or ridge search was run.
The question is whether a whole model following its own compensated path improves fresh-document
NLL at identical INT4 core bytes, and whether that transfers beyond the PG19 calibration domain.

## Implementation

BF16 is unchanged. RTN quantizes all196 original q/k/v/o/gate/up/down projections.
QUARTER and HALF independently start from the original checkpoint. Each group captures current
CAL Z with the native entire CausalLM forward. Teacher X always comes from the complete BF16
model. The order is q/k/v → native attention → o → residual/norm → gate/up → SiLU product → down.
Shared-input qkv and gate/up use one factorization within a method. No K/Z is shared across strengths.
There is no blockwise shortcut, skipped residual, or shortened suffix.

<details><summary>Numerical contract</summary>

W[out,in], X/Z[in,tokens]. G=ZZᵀ, λ=.001 trace(G)/d_in.
CPU FP64 Cholesky plus triangular solves compute (G+λI)Kᵀ=[W(X−Z)Zᵀ]ᵀ using the full input
Gram and output-row chunks128. Wstar=W+αK uses the original BF16-representable W once.
Each compensated model fits196 new K matrices with112 group factorizations. First-block qkv
has X=Z and K=0, with byte-identical RTN artifacts.

FP32(Wstar) is encoded as signed[−8,7], input group64, per-output-row/group FP32 absmax/7,
nearest-even, true-zero-only unit scales, low-nibble-first, header36B. Every file is saved,
independently decoded, and installed as BF16(What). K is not a runtime residual.
CAL uses A=W(X−Z), C=αKZ, B=(What−Wstar)Z and B0=(Q(W)−W)Z.
The three-term identity is Δ=T_inherited+T_cancellation+T_requant. B0 is local to the method's
current Z, not the separate whole-RTN path. Across392 CAL decompositions, maximum scaled gap
was {maxgap:.3e}; maximum solve residual was {maxsolve:.3e}, passing the preregistered1e−10 gates.
Actual module arithmetic/cast residuals are separate in the [fixed-SMOKE record](../../results/p2_r1/smoke_matrix_verification.json).

</details>

Embedding/lm_head tying, norms, RoPE, attention and bias policy are preserved. Runtime is
BF16, batch1, seq256, SDPA, eval/no_grad, use_cache=False, TF32 and BF16/FP16 reduced
reduction OFF, native full256-position head for every arm.

## Data and experiment

| Purpose | Documents | Use |
| --- | ---: | --- |
| CAL-FIT | Original P1 PG19 32 books | Same tokens/offsets, new own-trajectory P2 fitting |
| Fresh EVAL-ID | 64 new PG19 books | 16,320 valid next tokens per method |
| Fresh SHIFT | 16 new ArXiv content groups | 4,080 valid next tokens per method, separate cohort |
| SMOKE | Reused original1 | Validation only; excluded from evaluation denominators |

Known prior manifests excluded209 unique inputs (177 PG19 books and32 ArXiv groups).
All sources were cached; new transfer was0 bytes. Canonical book IDs, content/token hashes,
normalized windows and token5-gram overlap were checked. The maximum window Jaccard across
19,880 comparisons was {inp['near_duplicate_check']['maximum_Jaccard']:.6f}. Full-document near-duplicate
and pretraining exposure guarantees are unavailable. Pile grouping is by content, without article IDs.

Spec preceded new-data preparation. After CPU/SMOKE checks, construction ran RTN → HALF →
QUARTER. All588 artifacts were frozen at **zero fresh forwards**, then evaluation ran BF16 →
QUARTER → HALF → RTN. Seeds: data6101521, order6101522, bootstrap6101523. The fit interface
rejects non-CAL inputs. No chat/BOS/EOS is added; add_special_tokens=False. The255 original
next tokens provide gold NLL and argmax changes; fixed16 positions provide full-vocabulary
KL(BF16∥method). These are teacher-forced measures, not free generation or agent success.

## Results

### Fresh PG19,64 books

{table('EVAL-ID')}

![PG19 quality](../../figures/p2_r1/01_pg19.png)

QUARTER recovers {100*qr['recovered_RTN_loss_fraction']:.2f}% of the RTN−BF16 NLL gap; HALF recovers
{100*hr['recovered_RTN_loss_fraction']:.2f}%. Both remain above BF16 NLL. Recovery uses a loss-gap denominator,
not task accuracy. HALF's slightly lower point estimate does not replace the registered primary.

### Fresh ArXiv,16 content groups

{table('SHIFT')}

![ArXiv quality](../../figures/p2_r1/02_arxiv.png)

### Paired document comparisons

{contrast}

PG19 QUARTER−RTN is the single primary contrast. Other intervals are auxiliary pointwise.
The10,000 paired document bootstrap resamples condition on this model, original CAL32 and
fixed strengths. They do not cover calibration refitting or a model population. PPL is
exp(totalNLL/validTokens), not mean document PPL. The project1% PPL reference is separate
from measured effect size; the primary reduction here is {abs(100*p['PPL_relative_change']):.3f}%.

![Paired distributions](../../figures/p2_r1/03_paired_document_nll.png)

### Storage and offline costs

{cost}

Projection core excludes common embedding/head/norm parameters and is not a standalone model.
Construction includes native CAL captures, FP64 fitting, packing/reloading, local diagnostic
reductions and I/O. Cumulative worker time adds the main/reload lifetimes and registered SMOKE
process. Data preparation took {inp['data_prepare_seconds']:.3f}s; CPU audit/plots/documentation are separate.
Inclusive internal timings overlap and must not be summed again. Whole-device samples are
post-forward observations, not a continuous Windows peak. Torch allocator peaks are distinct.

![Storage and offline cost](../../figures/p2_r1/04_storage_and_offline_cost.png)

## Analysis

Weak full-model compensation substantially reduced RTN's fresh PG19 loss at identical core bytes.
ArXiv QUARTER had a slightly smaller teacher KL while gold NLL increased in point estimate;
HALF combined a strong PG19 result with a larger ArXiv loss. These observations are consistent
with calibration-domain dependence, without identifying a unique causal layer or a universal rule.
CAL local decomposition terms are not additive whole-model NLL contributions. The six P1b
interventions and this whole-model evaluation use different units; their effects are not summed.

## Conclusion

The registered primary verdict is **LOWER_NLL** for PG19. SHIFT is **INCONCLUSIVE with a
higher point estimate** for QUARTER and **HIGHER_NLL** for HALF. All four arms,320 evaluation
rows and588 new artifacts are complete; missing/failed model cells:0.

| Axis | Evidence and scope |
| --- | --- |
| Implementation | 196 projections/model;588 files;392 K fits; own-trajectory order and CAL-only guards PASS |
| Tests | 9 unique new P2 tests PASS; original15 and P1b10 rerun as regressions |
| Independent scalar audit | 320 rows,8 summaries,10 comparisons,20 interval fields,392 local decompositions,56 checkpoints PASS |
| Separate actual checks | Fresh GPU process independently decoded588 files and exactly matched4 whole-model SMOKE output hashes; CPU3 full-SMOKE matrix checks and8 selected CAL normal equations PASS |
| Preservation | Original case423 files, private4482 files, tracked6351 files, model6 files including whole local weight file, prior P1b ZIP, Git index/HEAD/remotes unchanged |
| Runtime | Packed storage → dense BF16 quality reference; serving speed/capacity, TTFT/TPOT and GPU busy time unmeasured |
| Out of scope | P3,4B,agent,new selector,package changes,git stage/commit/push/PR/merge,Pages/Drive |

## Reproduction and sources

[Actual commands](REPRODUCE.md) · [Spec](specs/p2.json) · [Raw metrics](../../results/p2_r1/metrics/) ·
[Decision](../../results/p2_r1/decision.json) · [Preservation](../../results/p2_r1/preservation.json) ·
[Resources](../../results/p2_r1/resource_summary.json).

Branch research/case014-quantization-compensation, base/current Git HEAD
09194c3f2e2c991feff5b6abab0cf8ab81a07052. Execution source SHA256:
`{sf['execution_source_sha256']}`. Git commit, source-tree digest, model revision and artifact
hashes are separate identities. Model revision: c1899de289a04d12100db370d81485cdf75e47ca.
Local whole-weight SHA256: f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b.
These verify local identity; no new official external digest or D-origin comparison was required.

Guo's [paper v1](https://arxiv.org/html/2610.16020v1) and the historical
[source audit](../../references/source_audit.json) provide the theoretical/prior-experiment context.
P2 is an independent project ridge/RTN implementation, not a complete official QEP reproduction.
The historical HTML/PDF audit is not counted as new P2 source retrieval. Codex assisted code,
analysis and reporting; no independent human review is claimed.

The review bundle contains source, small fixtures, scalar records, specs, manifests, reports and
figures. Model/K/INT4 weights/X/Z/logits/source texts are excluded. Public scalar reanalysis does
not reproduce private GPU calculations. Private material locations/inventory remain outside the ZIP.
"""
    (base/'REPORT.ko.md').write_text(ko);(base/'REPORT.md').write_text(en)
    demo=base/'demo';demo.mkdir(exist_ok=True)
    rows=''.join(f"<tr><th>{html.escape(short[x['method']])}</th><td>{'PG19 (n=64)' if x['split']=='EVAL-ID' else 'ArXiv (n=16)'}</td><td>{x['NLL_token_weighted']:.6f}</td><td>{x['PPL']:.3f}</td><td>{x['NLL_vs_RTN']:+.6f}</td></tr>" for x in s['method_summaries'])
    (demo/'index.html').write_text(f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Case014 P2 | 약한 보정의 전층 결과</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1050px;margin:auto;padding:24px;color:#1f2e3b;line-height:1.65;background:#f7fafc}}h1{{line-height:1.3}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:10px;border-bottom:1px solid #d6e0e8;text-align:right}}th:first-child{{text-align:left}}.scroll{{overflow:auto}}a{{color:#245e8c}}img{{max-width:100%;height:auto}}code{{overflow-wrap:anywhere}}.card{{padding:20px;background:white;border:1px solid #d6e0e8;border-radius:10px;margin:20px 0}}@media(max-width:520px){{body{{padding:12px}}h1{{font-size:25px}}th,td{{padding:8px;font-size:14px}}}}</style>
<h1>약한 양자화 보정은 전체 모델에서도 도움이 될까?<br><small>Does weak compensation help a whole model?</small></h1><p>Case014 P2 · Qwen3-0.6B · COMPLETED_P2_CORE</p>
<nav><a href="../REPORT.ko.md">한국어 보고서</a> · <a href="../REPORT.md">English report</a> · <a href="../REPRODUCE.md">CPU/GPU reproduction</a> · <a href="../../../results/p2_r1/summary.json">Saved results</a></nav>
<div class="card"><strong>PG19 QUARTER vs RTN: ΔNLL {p['mean_document_delta_NLL']:+.6f}, CI95 {ci(p)} / PPL {pct(p['PPL_relative_change'])}</strong><p>같은247,733,136B projection core에서 PG19 이득을 관측했습니다. ArXiv는 QUARTER 불확정·악화 점추정, HALF 악화입니다.<br>PG19 improves at identical core bytes; SHIFT limits transfer. Stored INT4 → dense BF16 quality reference.</p></div>
<div class="scroll"><table><thead><tr><th>Method</th><th>Cohort</th><th>NLL ↓</th><th>PPL ↓</th><th>ΔNLL vs RTN</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>문서별255 valid tokens, full-vocab KL16 positions. Primary: PG19 QUARTER−RTN. No serving latency or packed-kernel speed measurement.</p>
<img src="../../../figures/p2_r1/01_pg19.png" alt="Four whole-model PG19 NLL and PPL"><img src="../../../figures/p2_r1/02_arxiv.png" alt="Separate ArXiv NLL and PPL">
<p><a href="../../../figures/p2_r1/03_paired_document_nll.png">All paired document differences</a> · <a href="../../../figures/p2_r1/04_storage_and_offline_cost.png">Bytes and offline costs</a></p></html>''')
    print('Bilingual P2 reports and standalone results table generated from audited records.')
if __name__=='__main__':main()
