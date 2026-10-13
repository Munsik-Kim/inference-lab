# Case 013 — 가중치 오차가 줄면 실제 답변도 좋아질까?

[English](REPORT.md) | 한국어

INT4 변환·저장·복원과 동일 입력의 품질 비교 도구를 구현했습니다. 공개 수식으로 만든 reference variant를 두 Qwen 모델 전체에 적용하고, 문장 점수·장문 답·자유 도구 호출을 연결해 확인했습니다. 작은 가중치 오차와 실제 작업 성공률은 이 실험에서 같은 방향으로 움직이지 않았습니다.

## 목차

1. [배경](#background)
2. [가설과 검증 질문](#questions)
3. [이론과 비용 모델](#theory)
4. [방법](#methods)
5. [실험](#experiments)
6. [실험 결과](#results)
7. [결과 분석](#analysis)
8. [결론](#conclusion)
9. [레퍼런스와 기여](#references)

<a id="background"></a>
## 1. 배경

양자화는 모델의 가중치를 적은 비트로 표현하는 작업입니다. 원래 숫자와 가까운지 보는 복원 오차(MSE), 다음 기호의 확률 점수(PPL), 정확한 답과 도구 행동은 서로 다른 질문입니다. 계산이 더 빨라지는지는 실제 저비트 kernel과 전체 요청 비용으로 따로 확인해야 합니다.

DualQuant는 가중치만으로 행·열 스케일을 조정합니다. 여기서는 비공개 block-format 의존성이 없는 공개 INT4 수식을 독립 모듈로 구현했습니다. [논문 v1](https://arxiv.org/html/2610.13767v1)의 저자 측정과 이 RTX 5080 reference 실험은 별도입니다.

<a id="questions"></a>
## 2. 가설과 검증 질문

공개 수식의 INT4 경로를 전체 모델에 적용할 수 있는가, RTN보다 작은 weight MSE가 실제 확률·정답 점수에도 도움이 되는가, 장문과 여러 도구 호출에서 오류가 누적되는가를 확인했습니다.

이는 REFERENCE_DIAGNOSTIC/DEV 단계입니다. 설정은 출력 관측 전에 고정했고 성능을 보고 clipping·best iteration·calibration을 선택하지 않았습니다. 16K 에이전트 확장은 기본 DEV 뒤의 후속 질문입니다. 각 family의 첫 ID 네 개를 선택하고 추가 실행 전에 입력을 고정했습니다. 최종 120-task TEST, GPTQ paired serving, 비열등성·GO_SPEED·GO_CAPACITY 판정은 수행하지 않았습니다.

<a id="theory"></a>
## 3. 이론과 비용 모델

W는 [out,in], X는 [tokens,in]입니다. Group64마다 양의 inverse row/column scale a,b를 두고, 내부 code Q=clip(round(W/(a·b)),−8,7)를 계산합니다. 고정 code에서 a의 최소제곱 갱신 후 code를 다시 계산하고 b를 갱신합니다. 분모에는 ε=1e−12가 들어갑니다. 열 갱신의 reduction에는 모든 output row가 참여합니다.

15회 이후 scaled weight에 최종 group absmax scale γ를 적용합니다. 저장 표현은 W_tilde=Q_final·a·γ, W_effective=W_tilde·b이며, D16은 linear(X·b,W_tilde,bias)입니다. 열 스케일을 입력과 weight에 중복 적용하지 않습니다. D4는 X·b에 token/group64 동적 INT4를 추가합니다. 실제 BF16 rounding에서 두 대수 표현은 완전히 같은 bytes일 필요가 없습니다.

Signed INT4는 −8…7, ties-to-even rounding입니다. 실제 두 nibble을 한 byte에 저장하고 code, FP32 γ/a/b, shape와 24-byte header를 셉니다. RTN은 identity a/b를 저장하지 않습니다. 양의 row 배율은 최종 absmax에서 상쇄될 수 있으므로 row-only의 최종 RTN 대응은 진단 대조입니다. 내부 iteration objective를 최종 저장 오차의 단조 감소로 해석하지 않습니다.

Qwen 4B의 BF16 KV 이론 payload는 2×36×8×128×2=147,456B/token/stream입니다. 8K/16K/32K에서 1.125/2.25/4.5GiB이며 weight·workspace·block rounding과 입력 뒤의 출력은 별도입니다. 이 계산은 실제 VRAM 측정이 아닙니다.

<a id="methods"></a>
## 4. 방법

공개 commit `d84b7e9dff653320e47c06a365eb8295c1825bc3`의 element INT4, 초기화, BCD, 최종 group 연산을 읽어 대응했습니다. 주 설정은 group64, output-row split1, 15iteration, max_abs row/L1 column 초기화, FP32 offline 및 scales, scale projection 없음, element-only inner입니다. 논문의 BF16 scale projection과 공개 기본 inner=false와 다르고 구조적 zero에 unit scale 예외가 있어 전체 이름은 **PAPER_INT4_REFERENCE_VARIANT**입니다. D4는 **W4A4_REFERENCE_VARIANT**입니다.

BF16 / RTN W4A16 / D16 / D4를 비교했습니다. D16과 D4의 weight artifact는 같습니다. D4만 activation INT4를 추가합니다. RTN W4A4나 기존 GPTQ를 같은 실험에서 평가한 것은 아닙니다.

각 block의 q/k/v/o/gate/up/down, 0.6B 196개·4B 252개 projection을 변환했습니다. embedding/lm_head/norm/attention/RoPE/KV와 tie 관계는 유지합니다. optimizer에는 원래 weight만 들어갑니다. [solver/adapter](source/paper_int4.py)와 [runner](source/paper_model_runner.py)를 분리했습니다.

품질 경로는 W_tilde를 BF16으로 복원해 BF16 matmul을 수행합니다. 입력 β 곱은 FP32입니다. 실제 code 저장을 구현했지만 본 품질 경로를 packed 저비트 가속이라고 부르지 않습니다. 원래 weight artifacts는 비공개 로컬에 남으며 공개 manifest는 식별자만 제공합니다.

<a id="experiments"></a>
## 5. 실험

모델 revision은 Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca`, Qwen3-4B-Instruct-2507 `cdbee75f17c01a7cc42f958dc650907174af0554`입니다. 모델을 순차 로드했습니다. 환경은 RTX5080 SM120, driver610.88, Torch2.13.0+cu130, Transformers5.17.0, NumPy2.3.5, vLLM0.29.0입니다. 새 학습·calibration search·모델 다운로드·기존 환경 upgrade는 0입니다.

WikiText-2 raw test의 비어 있지 않은 행을 두 newline으로 연결하고 BOS/EOS 없이 첫 32개 비중복 2,048-token 구간을 사용했습니다. 구간마다 첫 token을 제외한 2,047개를 채점해 모델/방법당 65,504개입니다. NLL은 전체 위치, KL(BF16||candidate)은 고정 32위치/window의 full vocabulary, argmax flip은 전체 채점 위치입니다. 같은 32구간을 네 방법에서 재사용했습니다. 여기서는 각 모델의 원형 대비 변화를 중심으로 해석합니다.

실제512-token 전체 forward는 조건당1회입니다. CSV의512-prefix PPL은 저장된2048 기록 첫511loss를 CPU로 사후 재계산한 열입니다. 전체 benchmark PPL이나 formal 새 TEST가 아닙니다.

장문 DEV는 근거 회수·문서 결합·최신 상태 각4개, 총12개입니다. 동일 ID의8K/16K 및32K 변경본을 독립 문항으로 합치지 않습니다. Greedy/batch1, 출력 최대96token, KV BF16, SDPA, logits_to_keep=1이며 offload·truncate·KV 양자화는 없습니다.

에이전트 DEV는 네 family 각2개, 총8개입니다. Gold로 정의한 공통 replay와 자유 실행을 따로 평가합니다. 도구는 연구 소유 문서/record/table/계산/JSON 수정/validation에 한정합니다. 초기 약4K/8K, 출력192token, 최대8모델호출/6도구호출,900초 REFERENCE_PILOT deadline,24K context 한도입니다. Turn 사이 prefix cache는OFF, 요청 안의KV는ON입니다. 기존 serving180초와 다른 경계입니다.

실행된 GPU worker wall 누계는2,251.160초/14,400초 budget입니다. Loading·CPU serialization·idle이 포함되어 GPU kernel busy time이 아닙니다. 이번 공개 정리는 기존 기록만 CPU로 읽었습니다.

<a id="results"></a>
## 6. 실험 결과

### A. 전체 모델의 확률 점수

| Model | Method | NLL ↓ | PPL ↓ | KL ↓ | Argmax flip | Normalized weight MSE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| 06b | BF16 | 3.057517 | 21.274661 | 0.000000 | 0.000% | 0.00000000 |
| 06b | RTN | 3.316951 | 27.576153 | 0.398976 | 28.933% | 0.01228731 |
| 06b | D16 | 3.196423 | 24.444930 | 0.261293 | 22.070% | 0.01208061 |
| 06b | D4 | 3.547731 | 34.734400 | 0.675540 | 36.998% | 0.01208061 |
| 4b | BF16 | 2.315826 | 10.133291 | 0.000000 | 0.000% | 0.00000000 |
| 4b | RTN | 2.380479 | 10.810076 | 0.213157 | 14.340% | 0.01216967 |
| 4b | D16 | 2.388248 | 10.894392 | 0.195624 | 13.532% | 0.01201816 |
| 4b | D4 | 2.512728 | 12.338547 | 0.389818 | 22.573% | 0.01201816 |

MSE와PPL의 D16−RTN 상대 변화: 06b: MSE -1.6822%, PPL -11.3548%; 4b: MSE -1.2449%, PPL +0.7800%. 0.6B에서는 두 지표가 좋아졌고4B에서는 MSE·KL·argmax 대응이 좋아져도PPL은 조금 나빠졌습니다. D4는 추가 활성값 양자화로PPL이 더 높았습니다. [원자료 CSV](records/quality_06b.csv) · [4B CSV](records/quality_4b.csv)

### B. 장문 exact 답변

| Method | 8K exact | 16K exact | 32K exact |
| --- | --- | --- | --- |
| BF16 | 12/12 | 12/12 | 10/12 |
| RTN | 11/12 | 10/12 | NOT_RUN_MEMORY_POLICY |
| D16 | 10/12 | 10/12 | NOT_RUN_MEMORY_POLICY |
| D4 | 8/12 | 7/12 | NOT_RUN_MEMORY_POLICY |

32K에서 BF16만 실행했습니다. 이후16K에서 관측한 workspace까지 반영한 보수적 메모리 정책으로 후보32K는 미실행입니다. 이는 OOM 실패나 최대 안정 문맥 측정이 아닙니다. [운영 변경 기록](records/operational_update_v3.json)

같은 ID의 획득·손실을 보존합니다. 평균이 같아도 각 문항의 결과가 다를 수 있습니다.

| Length | Baseline → candidate | Both pass | Baseline only | Candidate only | Both fail |
| --- | --- | --- | --- | --- | --- |
| 8192 | BF16 → RTN | 11 | 1 | 0 | 0 |
| 8192 | RTN → D16 | 9 | 2 | 1 | 0 |
| 8192 | RTN → D4 | 7 | 4 | 1 | 0 |
| 16384 | BF16 → RTN | 10 | 2 | 0 | 0 |
| 16384 | RTN → D16 | 10 | 0 | 0 | 2 |
| 16384 | RTN → D4 | 7 | 3 | 0 | 2 |

**POST_HOC_SAME_RECORDS:** exact scorer는 answer/evidence의 정확한 값을 요구하고 첫valid JSON object 추출을 허용합니다. Strict whole-document JSON은 별도입니다. BF16의32K 두 실패와RTN의 세 실패는 계산값이 맞아도 `21` 대신 `7 * 3 = 21`을 쓴 계약 차이였습니다. D16에는 READY-0→READY의 실제 값 손실, D4에는 key 대소문자·피연산자 반환·상태 suffix 손실·출력한도 도달이 섞였습니다. 주 점수는 그대로입니다. [전체 사후 구분](records/long_context_dev.csv)

### C. 도구 replay와 자유 실행

| Method | Base replay | Base free | 16K replay | 16K free | Base tool/parse errors |
| --- | --- | --- | --- | --- | --- |
| BF16 | 8/8 | 8/8 | 4/4 | 4/4 | 0/0 |
| RTN | 8/8 | 8/8 | 4/4 | 4/4 | 0/0 |
| D16 | 8/8 | 6/8 | 4/4 | 3/4 | 8/0 |
| D4 | 8/8 | 0/8 | 4/4 | 0/4 | 2/2 |

Base는8개 독립 task,16K는 그중4개의 후속 길이 변경본입니다. 총96개 episode 조건을96개의 독립 task로 세지 않습니다. Replay는 공통 tool 결과가 이미 제공된 문맥입니다.

D16의 AGENT-6/7은 `apply_fixture_patch`/`validate_fixture_test`를 CamelCase로 호출했고 같은 거절을 반복한 뒤 실제 파일 수정 없이 valid:true를 냈습니다. 같은 도구 계약에서BF16/RTN은 완료했습니다. 후보용 alias를 추가하지 않았습니다. D4는 도구를 쓰지 않거나 배경 문장을 상태로 읽고 잘못된 이름·유효하지 않은 fixture로 종료한 사례가 있습니다. [전체 raw episodes](records/raw/4b/D16/agent_dev.jsonl)

### D. 저장·시간·메모리

| Model | Method | Projection artifact bytes | NLL loop wall (s) | Torch peak allocated/reserved (B) | Convert or reload (s) |
| --- | --- | --- | --- | --- | --- |
| 06b | BF16 | N/A | 2.8858 | 1338743808/1400897536 | 0.000 |
| 06b | RTN | 247,730,784 | 3.2063 | 1339890688/1472200704 | 520.493 |
| 06b | D16 | 276,402,784 | 2.9750 | 1339890688/1447034880 | 48.709 |
| 06b | D4 | 276,402,784 | 4.0918 | 1339890688/1501560832 | 17.521 |
| 4b | BF16 | N/A | 7.7518 | 8297826304/8380219392 | 0.000 |
| 4b | RTN | 2,043,746,208 | 8.6668 | 8381351936/8598323200 | 107.126 |
| 4b | D16 | 2,274,662,304 | 8.7787 | 8384497664/8619294720 | 185.284 |
| 4b | D4 | 2,274,662,304 | 10.8704 | 8462288896/8732540928 | 65.451 |

Artifact는 변환 projection만의 code/scales/header이며 전체 checkpoint가 아닙니다. Runtime weights는 복원BF16이므로 artifact 크기를 실행VRAM 감소로 바꾸지 않습니다. NLL loop 시간에는 metric 전송·일부reference I/O가 들어가며 servingTTFT/TPOT가 아닙니다. D4 마지막 열은D16 artifact reload입니다.

0.6B RTN은 오래된 Python byte iteration, 이후 D16/4B는 동일 bytes의 bulk-copy serializer를 사용했습니다. 그 변환 시간 차이를 알고리즘 speedup으로 비교하지 않습니다. System GPU peak·TTFT·TPOT·cached tokens는 미측정입니다.

### E. 대수와 packed 대응

4개 실제 projection의 28행 진단에서 FP64 transport 최대 차이 0, FP32 최대 1.43051e−6, BF16 folded 표현 상대 RMSE 최대 0.0040052, FP64 row-only 상쇄 최대 6.66134e−16입니다. Tiny 1e−15 fixture의 zero beta는 명시적 실패이며 실제 모델 실패 빈도에 합치지 않았습니다.

vLLM Marlin의 실제SM120 CUDA kernel을 실행했습니다. INT4 weight를 fused unpack해 BF16 tensor-core MMA로 곱하는W4A16이며 nativeINT4 MMA/W4A4/저자의Nunchaku 경로가 아닙니다. β는 보존하지만 FP32 a·γ가 BF16 kernel scale로 바뀌는 probe variant입니다.

GEMM 상대RMSE0.003078, 한block0.005673은2%gate를 통과했습니다. 최종32token 전체forward는0.026054>0.02로미통과입니다.4token greedy suffix는 같아도 gate를 통과한 것으로 바꾸지 않았습니다. 이전probe2.5533%/2.2598%와 마지막2.6054%를 모두 보존합니다. 초기두handler의exit0에도 semantic BLOCKED였고 최종CLI는exit1입니다. [최종probe](records/diagnostics/packed_precision_aligned_03.json)

<a id="analysis"></a>
## 7. 결과 분석

작은 weight MSE는 긴 문맥의 exact 답변이나 여러 번의 tool 행동을 보장하지 않았습니다. 이 4B pilot에서는 거의 같은 PPL에서 도구 이름과 행동 안정성이 갈렸습니다. 적용 범위는 default와 다른 INT4 variant·작은 DEV·공통 cache OFF입니다.

정답 숫자와 출력 계약 위반은 분리해 읽습니다. 이를 사후에 정답으로 바꾸면 원래 exact 질문이 달라집니다. Replay 성공은 주어진 도구 결과를 읽은 관측이며, 자유 행동의 성공이나 모든 정보의 보존을 입증한 것은 아닙니다.

Packed 메모리·시간은 수치 기준을 통과하지 못한 prototype의 기록입니다. 전체 reference와 맞는 배포 경로를 확보하고 stock GPTQ와 동일 요청·KV·출력·cache·parser로 비교해야 실용 속도나 capacity를 판정할 수 있습니다. 본 4B 데이터는 BF16 weight가 상주하는 reference이며 W4 저장 artifact만으로 더 긴 문맥을 처리했다고 주장하지 않습니다.

원래 연구의 고유 CPU 39개(기존 29+신규 10)는 [receipt](records/verification.json)와 로그로 보존했습니다. 이번 publication 검사는 별도입니다. 공개 검산은 loss/prediction으로 NLL·PPL·flip과 답/실행 조건을 재계산하지만 full-logit KL·원 weight MSE·비공개 weight payload hash를 재계산하지 않습니다. 이러한 경계를 [CPU audit](analysis/audit.py)가 명시합니다.

<a id="conclusion"></a>
## 8. 결론

두 모델 전체에 적용하는 INT4 reference·실제 저장·재로딩·장문/agent 평가·독립 검산을 완료했습니다. 0.6B D16의 PPL은 RTN보다 낮았지만 4B에서는 소폭 높고 free-agent 성공도 줄었습니다. D4에는 더 큰 품질 손실이 관측됐습니다.

현재 variant와 DEV 예산에서는 weight 오차·문장 확률·작업 성공을 분리해서 판단해야 했습니다. 이를 저자의 모든 설정에 대한 판정으로 확대하지 않습니다. Packed 전체 모델 대응과 formal serving이 남았습니다. [CPU 재계산](REPRODUCTION.ko.md) 또는 [전체 기록](records)으로 직접 확인할 수 있습니다.

<a id="references"></a>
## 9. 레퍼런스와 기여

[NOTICE](NOTICE.md)에 DualQuant 논문·고정 공개 commit, Qwen revision, WikiText와 라이브러리·기여를 기록했습니다. 핵심 근거는 [수식 대응](records/paper_code_mapping.txt), [protocol](records/experiment_manifest.json), [scale 계약](records/reference_contract.json), [원형/공개 hash 대응](provenance/export_map.json)입니다.

원형 r1은 입력 준비의 pyarrow/return_dict 오류로 GPU 미실행입니다. r2에서 기존 CPU 환경과 명시적 return_dict=False로 해결했습니다. 원형 P0/r1/r2·실패 probe·Case009의 99파일을 보존했고 기존 Case001–012를 수정하지 않았습니다. 공개본의 개인 경로 비식별화는 복사본에만 적용했습니다. 이번 문서와 코드는 새 GPU 실험 없이 publication feature branch에서 준비했습니다.


## 상세 기술 기록과 그림

[논문·코드·실행 계약](METHOD_CONTRACT.md) · [평가 프로토콜](EVALUATION_PROTOCOL.md) · [실패·성공 사례](FAILURE_ANALYSIS.md) · [Packed probe](PACKED_PROBE.md) · [세 그림](figures/README.md) · [미실행 후속 계획](ROADMAP.md)

D16의 4B quantized projection payload는 RTN보다 11.2987% 큽니다. 추가 a/b scale을 포함한 비용이며 동일 bytes 비교가 아닙니다. BF16의 표기 N/A는 별도 INT4 artifact가 없다는 뜻입니다.
