# Case 009 — Qwen의 요청 비용과 공식 품질 평가

[English](REPORT.md) · [개요](README.ko.md) · [방법·영어](METHODS.md) · [CPU 재계산](REPRODUCTION.md)

## 목차

[배경](#background) · [가설](#hypotheses) · [이론](#theory) · [방법](#methods) · [실험](#experiments) · [실험 결과](#results) · [결과 분석](#analysis) · [결론](#conclusion) · [레퍼런스](#references)

<a id="background"></a>
## 배경

Case 008에서 GPTQ W4A16 저장본의 변환·검사·새 프로세스 실행을 구현했습니다. 당시 eager 8토큰 요청과 별도로, 이번에는 **같은 BF16/W4 저장본**의 긴 decode와 동시 요청 비용을 조사합니다. 공식 과제 품질 평가는 시간 측정과 분리했습니다. 과거 결과와 소스는 그대로 남깁니다.

새 구현은 loopback 요청 도구, 고정 입력·과제 manifest, 프로세스별 기록, 쌍별 스칼라 분석, 설치 가능한 CPU 비교 CLI를 연결합니다. 모델·양자화 방법·추론 커널은 upstream을 사용합니다. [기여와 선행연구](../../docs/related-work/README.ko.md).

<a id="hypotheses"></a>
## 가설

W4의 가중치 이동 감소가 decode 비용을 줄일 수 있지만 attention·KV·prefill·대기열·호스트 비용도 남습니다. graph 실행과 eager 실행에서는 비용의 구성이 달라질 수 있습니다. 양자화는 공식 과제 점수와 개별 답변도 바꿀 수 있습니다. 특정 속도비 합격점, 품질 동등성 한계 또는 W4가 이겨야 한다는 조건은 두지 않았습니다.

<a id="theory"></a>
## 이론

Linear의 연산량은 대략 2MKN이고 가중치 중심 전송량은 bytes-per-weight×KN에 scale 저장량을 더한 값입니다. M은 token-row 차원이며 client concurrency와 다릅니다. Amdahl 식 1/((1−f)+f/S)는 나머지 비용이 같다는 가정의 설명 도구입니다. 전체 요청 측정을 대신하거나 MARLIN 논문의 장치별 속도비를 RTX 5080 합격점으로 만들지 않습니다.

정답 전환은 기준선 정답의 손실과 새 정답을 합친 값입니다. 전체 답변 불일치는 다른 오답으로의 변경도 포함합니다. 선택지 NLL·Brier·KL은 전체 continuation likelihood를 선택지 사이에서 정규화한 진단입니다. 
제한된 진단은 24개 fresh-process attempt 모두 정상 종료했고 543.3초가 걸렸습니다. D0·D1 모두 통과했으며 D2는 선택하지 않았습니다. 상태는 **NOT_REPRODUCED_IN_SMALL_PROBE**로, 전체 평가 실패의 수정 완료가 아닙니다.

전체 어휘 KL은 수집하지 않았습니다. WikiText perplexity는 exp(−loglikelihood 합/분모 합)이며 문서 perplexity의 단순 평균이 아닙니다.

<a id="methods"></a>
## 방법

Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554`, RTX 5080 16GB, vLLM 0.29.0, TP=1, BF16 활성값·KV, 공통 4 GiB KV 용량을 사용했습니다. 원래 저장본 바이트를 확인했으며 모델 다운로드·재양자화·보정·학습은 수행하지 않았습니다. 저장 경계의 252개 projection은 runtime의 144개 fused projection에 대응하며 동적 GPU 호출 수가 아닙니다.

시간 측정은 입력 128/1024, 출력 256, 동시 요청 상한 1/4/16/32의 포화 부하입니다. 새 서버 프로세스의 짝지은 3라운드에서 BF16/W4 순서를 교차하고 같은 조건 순열을 썼습니다. 두 모델의 graph 정책은 같습니다. 별도 eager 비교는 입력 128·출력 256·동시성 1/16입니다. 시간 측정에서는 prefix cache·speculative decoding·chunked prefill·async scheduling을 껐습니다. 품질 평가는 context 8192와 chunked prefill을 사용하므로 시간 조건과 섞지 않습니다.

TTFT는 첫 choice SSE까지, TPOT는 (마지막−첫 SSE)/(실제 출력 토큰−1), 처리량은 실제 출력 토큰 합/측정 구간 시간입니다. API usage와 반환 token ID 수를 대조했습니다. 각 표는 3라운드의 p50/p95 요약을 평균합니다. 시간비는 BF16/W4, 처리량비는 W4/BF16입니다. 구간은 동시 요청을 독립 표본으로 세지 않고 짝지은 서버 라운드를 5,000회 재표집합니다.

공식 harness commit `d6de81643928d653435c431bae19945d41d32520`을 고정했습니다. ARC-Challenge 0-shot, MMLU 5-shot, GSM8K 5-shot·greedy·최대 1,024토큰, WikiText-2 rolling likelihood입니다. chat template는 ARC/MMLU/GSM에 한 번, WikiText에는 적용하지 않습니다. 문항별 구간은 과제 안에서, MMLU 종합 구간은 과목 cluster로, WikiText는 문서별 분모 합을 다시 계산해 구합니다. [고정 조건과 정의·영어](METHODS.md).

<a id="experiments"></a>
## 실험

비용 측정 60개 셀에서 timed 요청 7,296개·출력 1,867,776토큰을 완료했습니다. 고정 warmup 120개는 별도 기록했습니다. 고유 workload prompt 512개를 여러 조건에서 재사용했으므로 7,296개의 독립 품질 표본이 아닙니다. serving DEV는 3프로세스·37요청이며, C1/16의 제한된 profiler 진단을 포함합니다. 본 timing에는 profiler가 없습니다.

공식 test 예산은 과제×모델당 전체 split 1회입니다. ARC 1,172문항, WikiText 62문서, GSM 1,319문항, MMLU 57과목·14,042문항입니다. DEV는 각 모델의 ARC/WikiText validation 각 2개입니다. 모든 요청을 사전 동결한 인자 해시와 대조했으며 실패한 test split을 반복하지 않았습니다.

**실행 상태의 제한:** 전체 계산과 JSON 저장 뒤 native 종료 실패가 발생했습니다. 실패한 전체 평가 프로세스는 **arc_challenge/W4, wikitext/BF16, wikitext/W4, mmlu/W4**입니다. `COMPUTED_WITH_PROCESS_FAILURE`는 보존한 계산 기록이며 정상 실행 완료가 아닙니다. 앞선 ARC/BF16 DEV 1회도 종료에 실패했습니다. 명시적 engine shutdown은 일부 프로세스에서 정상 종료했지만 문제를 해결하지 못했습니다. CPU만 사용한 import·dataset 진단 3개는 정상 종료했고 native 원인은 미확정입니다. benchmark·정밀도·생성 상한을 바꾸거나 실패한 split을 재실행하지 않았습니다. [실행 기록](provenance/quality_attempts.json).

<a id="results"></a>
## 실험 결과

### 동시 요청 전체 곡선

![입력 128·출력 256의 TPOT와 처리량 전체 곡선](figures/serving-L128.png)

![입력 1024·출력 256의 TPOT와 처리량 전체 곡선](figures/serving-L1024.png)

영어 축 라벨의 선은 3라운드 평균, 막대는 라운드 최솟값–최댓값입니다. 신뢰구간과 다릅니다. 모든 graph 셀에서 요청 실패나 token 길이 불일치는 없었습니다.

| Mode / input / concurrency | TPOT time ratio BF16/W4 [95% interval] | Throughput W4/BF16 [95% interval] |
|---|---:|---:|
| graph / 128 / 1 | 2.113 [2.100, 2.138] | 2.090 [2.061, 2.123] |
| graph / 128 / 4 | 2.094 [2.068, 2.109] | 2.060 [2.030, 2.078] |
| graph / 128 / 16 | 1.934 [1.898, 1.966] | 1.864 [1.845, 1.881] |
| graph / 128 / 32 | 1.656 [1.644, 1.662] | 1.609 [1.597, 1.623] |
| graph / 1024 / 1 | 2.100 [2.079, 2.126] | 2.031 [2.016, 2.053] |
| graph / 1024 / 4 | 1.868 [1.864, 1.873] | 1.805 [1.802, 1.810] |
| graph / 1024 / 16 | 1.438 [1.429, 1.448] | 1.419 [1.415, 1.426] |
| graph / 1024 / 32 | 1.235 [1.229, 1.242] | 1.237 [1.232, 1.240] |
| eager / 128 / 1 | 1.071 [1.025, 1.109] | 1.048 [0.997, 1.092] |
| eager / 128 / 16 | 1.075 [0.968, 1.211] | 1.096 [0.981, 1.214] |


표의 95% 구간은 서버 라운드를 짝지어 재표집한 기술통계 구간입니다. eager 비교에서는 상대 변화가 작고 라운드 변동이 더 컸습니다. eager끼리 비교했으며 graph W4/eager BF16 비율을 양자화 자체의 효과로 부르지 않습니다. [절대 TTFT·TPOT·요청 시간 p50/p95·처리량](results/derived/serving_table.md).

### 공식 품질 계산

영어 지표명은 공식 task 이름을 유지했습니다. FAILED는 해당 metric 바로 위에 표시합니다. 전체 문항의 입력·유한 점수·공식 집계는 검산했지만 프로세스 정상 종료 검증은 미완료입니다. benchmark와 GSM의 두 추출 필터를 합산하지 않습니다.

| Official task / metric | BF16 | W4 | Paired W4−BF16 [pointwise 95% interval] |
|---|---:|---:|---:|
| arc_challenge **process status** | COMPLETE | FAILED | retained computation below; not clean execution |
| arc_challenge/none acc | 507/1172 (43.26%) | 515/1172 (43.94%) | +0.68 [-1.02, +2.39] pp |
| arc_challenge/none acc_norm | 502/1172 (42.83%) | 493/1172 (42.06%) | -0.77 [-2.47, +0.85] pp |
| wikitext **process status** | FAILED | FAILED | retained computation below; not clean execution |
| WikiText-2 word perplexity | 13.1081 | 14.5194 | word NLL +0.10226 [+0.09620, +0.10893] nats |
| WikiText-2 byte perplexity | 1.6180 | 1.6493 | separate original UTF-8 denominator |
| gsm8k **process status** | COMPLETE | COMPLETE | original clean process exits |
| gsm8k/flexible-extract exact_match | 1211/1319 (91.81%) | 1191/1319 (90.30%) | -1.52 [-2.96, -0.15] pp |
| gsm8k/strict-match exact_match | 1145/1319 (86.81%) | 1028/1319 (77.94%) | -8.87 [-11.14, -6.60] pp |
| mmlu **process status** | COMPLETE | FAILED | retained computation below; not clean execution |
| MMLU acc (57 subjects) | 8720/14042 (62.10%) | 7616/14042 (54.24%) | -7.86 [-9.05, -6.87] pp |


| Task / metric | n | Lost correct | New correct | Both wrong | Changed wrong | All answer disagreements |
|---|---:|---:|---:|---:|---:|---:|
| arc_challenge/none acc | 1172 | 49 | 57 | 608 | 65 | 171 |
| arc_challenge/none acc_norm | 1172 | 54 | 45 | 625 | 45 | 144 |
| gsm8k **process status** | COMPLETE | COMPLETE | original clean process exits |
| gsm8k/flexible-extract exact_match | 1319 | 53 | 33 | 75 | 29 | 125 |
| gsm8k/strict-match exact_match | 1319 | 185 | 68 | 106 | 36 | 289 |
| MMLU acc (57 subjects) | 14042 | 1452 | 348 | 4974 | 600 | 2400 |


Lost correct는 기준선 정답 손실, New correct는 새 정답, Changed wrong은 다른 오답으로의 변경입니다. 정확도 차이의 단위는 퍼센트포인트(pp)입니다. [MMLU 과목별 결과·점수·생성 상한·구간](results/derived/quality_summary.json) · [문항별 공개 스칼라](results/raw/quality_scalars.jsonl). 
제한된 진단은 24개 fresh-process attempt 모두 정상 종료했고 543.3초가 걸렸습니다. D0·D1 모두 통과했으며 D2는 선택하지 않았습니다. 상태는 **NOT_REPRODUCED_IN_SMALL_PROBE**로, 전체 평가 실패의 수정 완료가 아닙니다.

전체 어휘 KL은 **NOT_RUN**입니다. 제외된 지문·생성 이유·전체 tensor는 스칼라만으로 복원하지 않습니다.

### 메모리와 대기열

| Server process | Whole-device sampled peak MiB | Sampled maximum running / waiting | Graph capture log |
|---|---:|---:|---|
| round1-BF16-graph | 14863.0 | 32 / 10 | True |
| round1-W4-graph | 9747.0 | 32 / 9 | True |
| round2-W4-graph | 9297.0 | 32 / 9 | True |
| round2-BF16-graph | 14477.0 | 32 / 10 | True |
| round3-BF16-graph | 14419.0 | 32 / 12 | True |
| round3-W4-graph | 9315.0 | 32 / 11 | True |
| round1-BF16-eager | 14269.0 | 16 / 0 | False |
| round1-W4-eager | 9175.0 | 16 / 0 | False |
| round2-W4-eager | 9175.0 | 16 / 0 | False |
| round2-BF16-eager | 14269.0 | 16 / 0 | False |
| round3-BF16-eager | 14269.0 | 16 / 0 | False |
| round3-W4-eager | 9175.0 | 16 / 0 | False |


메모리는 프로세스 시작·warmup·측정·desktop을 포함한 1 Hz 장치 전체 표본의 peak입니다. Torch allocator peak나 모델만의 VRAM이 아니며 짧은 peak를 놓칠 수 있습니다. 개별 셀과 시간 정렬한 기록이 없어 동시성별 메모리 곡선을 만들지 않았습니다. L1024/C32의 graph 6개 셀에서 각각 54회, 총 324회 preemption이 관측됐고 timing에서 제외하지 않았습니다. client concurrency 32를 계속 활성화된 32개 sequence 또는 GEMM row로 해석하지 않습니다. W4만의 최대 capacity sweep은 실행하지 않았습니다.

<a id="analysis"></a>
## 결과 분석

전체 graph grid에서 W4의 TPOT가 낮았고, 동시 요청과 입력 길이가 커질수록 상대 비율은 줄었습니다. 요청 처리와 KV·대기열 비용이 함께 포함됩니다. DEV trace는 graph launch와 W4 Marlin 커널을 보여주지만 특정 커널의 인과적 기여율을 분리하지 않습니다. [trace 요약과 관측 allocator 값](provenance/dev_profile_costs.json).

같은 GSM8K 출력에서 공식 strict-match 차이는 -8.87퍼센트포인트, flexible-extract 차이는 -1.52퍼센트포인트였습니다. 1,024토큰 상한에 도달한 출력은 BF16 12/1,319개, W4 40/1,319개이며 결과에서 제외하지 않았습니다. 추출 규칙과 길이 제한에 따른 차이를 볼 수 있지만, 어느 원인이 그 차이를 만들었는지를 분리한 실험은 아닙니다. MMLU의 문항 가중 정확도 차이는 57과목에서 -7.86퍼센트포인트였고, 과목별 결과와 과목 cluster 구간도 함께 남겼습니다.

정확도와 정규화 정확도는 방향이 다를 수 있습니다. 확률 손실·답변 전환·perplexity도 따로 봅니다. 가장 좋은 지표로 나머지를 대체하지 않습니다. 시간 라운드는 3회뿐이며 WSL desktop·배경 작업과 일부 CPU 문서·검사 작업이 함께 실행됐습니다. 이 구간은 운영 SLO나 범용 속도비를 확정하지 않습니다.

새 data-v2는 label·프로그램 조건·순서 RNG를 나누고 제한된 정수 loop의 정답을 검사합니다. 초안 123/336개에서 값 범위 밖 distractor를 발견해 원본을 보존하고 모델 평가 전에 고쳤습니다. 최종 336개 fixture는 바이트 동일하게 재생성되며 정답 양쪽에 유효 범위의 보기가 있습니다. 정답이 내부 값인 설계 단서는 남습니다. **data-v2 모델 평가는 0회**이며 공식 과제를 대체하지 않습니다.

<a id="conclusion"></a>
## 결론

공통 자원 조건의 전체 serving 곡선, 종료 실패를 표시한 공식 품질 계산 기록, 설치 가능한 paired CPU CLI를 연결했습니다. 관측한 graph workload에서는 기존 W4 저장본의 요청 비용이 낮았습니다. 품질표와 미해결 native 종료 문제가 같은 결과의 일부입니다. 배포 판정은 **NOT_ASSESSED**, 게시 상태는 **LOCAL_REVIEW**입니다.

R은 고정 구조의 ridge 보정과 구조를 복원하는 저장·로딩 구현으로 유지합니다. 다층 R 실험은 하지 않았습니다. [별도 후속 설계·영어](../../docs/phase2a-design.md). 완전히 정상 종료하는 품질 파이프라인을 주장하려면 후속 환경·종료 경로 진단이 필요합니다. 이번 실패한 전체 split은 다시 실행하지 않습니다.

<a id="references"></a>
## 검토 후 추가 분석과 실행 진단

**POST_HOC_SAME_RECORDED_SCALARS** — 원래 공개 스칼라와 집계를 다시 계산했습니다. 원래 입력·점수·과목별 구간은 변경하지 않았습니다.

- **MMLU:** 57/57과목에서 정확도가 하락했습니다. 전체 14,042문항에서 BF16 정답 8,720개 중 1,452개를 잃고, BF16 오답 중 348개를 새로 맞혀 순 1,104개가 감소했습니다. D 선택 비중은 5,842/14,042(41.60%)에서 7,163/14,042(51.01%)로 늘었습니다. 정답 label별 분모·정답 수는 [사후 JSON](publication/posthoc/analysis.json)에 있습니다. 이는 선택 위치 변화의 관측이며 원인 분해가 아닙니다.
- **WikiText-2:** 62/62문서에서 W4 로그우도가 낮았습니다. 원래 241,335단어·1,290,527 byte와 합산 로그우도로 계산한 단어 PPL은 13.1081→14.5194, 약 10.77% 증가했습니다. 문서별 PPL 평균이나 비공개 원문의 재파싱이 아닙니다.
- **GSM8K:** 같은 생성 결과에서 strict 오답·flexible 정답은 BF16 67개→W4 165개, 반대 조합도 1개→2개입니다. flexible 정답 손실 53개·개선 33개로 순 20개가 감소했습니다. 공개 집계의 평균 생성 길이는 185.20→232.54토큰, 1,024토큰 상한 도달은 12→40개입니다. rationale의 실패 이유나 이 길이비로 실제 GSM 가속을 추정하지 않습니다.
- **Serving:** L128/C1의 TPOT 시간비 2.113배와 출력 처리량비 2.090배는 서로 다른 지표입니다. TTFT는 17.069→18.520ms입니다. L1024/C32의 6개 셀에 각각 54회, 총 324회 preemption이 포함됐고 W4 처리량은 C16 1,059.33→C32 1,010.15 token/s입니다. graph·compile·fusion 묶음 비교이며 CUDA Graph 단독 효과는 분리하지 않았습니다.

### 소프트웨어 수정과 별도 lifecycle

`diova-compare` 0.1.1은 task/version 안의 지표 coverage 불일치를 명시적 오류로 거절하고, 극소 양수 확률의 KL을 로그 차이로 계산합니다. 이는 소프트웨어 결함 수정이며 새 모델 결과가 아닙니다. [수정 기록](../../packages/diova-compare/CHANGELOG.md).

원래 전체 평가의 종료 상태는 4개 정상·4개 비정상으로 유지합니다. 별도 짧은 synthetic lifecycle 진단은 generation/continuation/rolling API와 객체 정리 시점을 비교합니다. 실제 attempt·worker·parent 종료 상태는 [진단 요약](supplemental/lifecycle-v1/summary.json)과 [진단 설명](supplemental/lifecycle-v1/README.md)에 있습니다. 작은 probe 성공은 과거 full split의 복구나 오류 원인 규명을 뜻하지 않습니다.


제한된 진단은 24개 fresh-process attempt 모두 정상 종료했고 543.3초가 걸렸습니다. D0·D1 모두 통과했으며 D2는 선택하지 않았습니다. 상태는 **NOT_REPRODUCED_IN_SMALL_PROBE**로, 전체 평가 실패의 수정 완료가 아닙니다.

전체 어휘 KL은 NOT_RUN이며 data-v2는 입력 CPU 검사만 완료했습니다. 현재 공개본·원검토본 복원은 [publication 안내](publication/README.md)에서 분리합니다.

## 레퍼런스

[원문·공식 저장소·버전·구현 대응](../../docs/related-work/README.ko.md)에 MARLIN, Dutta, He, FLAP, lm-evaluation-harness, vLLM을 연결했습니다. [출처·영어](NOTICE.md) · [재현 안내·영어](REPRODUCTION.md) · [현재 실행 상태](RUN_STATE.json).
