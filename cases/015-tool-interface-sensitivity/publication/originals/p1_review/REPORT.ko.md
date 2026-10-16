# Case015 — 같은 도구, 다른 이름

**Tool-Interface Sensitivity of Quantized LLM Agents**

같은 작업에 제공하는 도구의 이름과 순서를 바꾸고, 모델이 실제로 작업을 끝내는지 비교하는 도구를 구현했습니다. 정확한 이름 변환, 단계별 동등성 검사, 실제 fixture 실행과 저장 응답 재감사를 연결했습니다.

Qwen3-4B의 BF16·RTN·D16을 네 인터페이스에서 비교한 96개 FREE episode를 완료했습니다. D16의 I0 patch 실패 2개는 I3 불투명 이름에서 회복됐고, I2 lowerCamelCase에서는 상태 조회 과제의 새 손실이 생겼습니다. 주 효과는 **+8.33 percentage points**, 탐색적 paired 95% 구간은 **[-8.33,+25.00] pp**입니다. 부호가 양수라는 사실만으로 D16이 더 안정적이라고 결론내릴 수 없습니다.

## 목차

[배경](#background) · [질문·가설](#questions) · [인터페이스·실행 계약](#contract) · [방법](#method) · [실험](#experiment) · [결과](#results) · [실패·복구](#failures) · [결론](#conclusion) · [재현·출처](#references)

<a id="background"></a>
## 배경

Case013에서는 원형 도구 이름에서 D16의 UpperCamelCase 호출 실패를 관측했습니다. 이후 ENUM·오류 안내 보완은 같은 기존 DEV에서 개선을 보였지만, 이번에는 두 완화책을 모두 끈 원형 기준을 사용했습니다. 같은 작업의 인터페이스만 바꾸는 평가 도구가 목적이며, 새로운 양자화법이나 CIPR 학습 재현은 수행하지 않았습니다.

RTN은 단순 반올림 INT4, D16은 DualQuant 공개 수식 기반 weight INT4 reference입니다. 실제 성공은 최종 JSON뿐 아니라 과제에 필요한 읽기·계산·파일 수정·검증과 최종 상태를 확인한 점수입니다.

<a id="questions"></a>
## 질문과 가설

1. D16의 인터페이스 변경 후 signed 성공 변화가 BF16보다 나쁜가? 등록 주 비교는 아래 Δ이며, 음수일 때 그 방향입니다.
2. 차이가 이름·의미 선택·인자·오류 후 행동·최종 상태 중 어디서 관측되는가?
3. 기존 실행 계약 보완을 별도 진단으로 재검토할 필요가 있는가?

`d(m,t)=mean_i=I1,I2,I3 Y(m,i,t)-Y(m,I0,t)`

`Δ=mean_t[d(D16,t)-d(BF16,t)]`

손실과 회복이 상쇄될 수 있어 paired 손실·회복, S=평균 절대 변경률, 모든 인터페이스 성공 수도 함께 저장했습니다. I2는 과거 이름 오류를 알고 정한 DEV 조건이며 I3는 과제당 이름 배정 한 개입니다.

<a id="contract"></a>
## 인터페이스와 실행 계약

| 조건 | 바꾼 것 | 유지한 것 |
|---|---|---|
| I0 ORIGINAL | 없음 | 원 이름·순서·schema·설명·messages |
| I1 ORDER_ONLY | 과제별 seed 기반 비항등 순열 | 이름·내용·초기 messages |
| I2 STYLE_RENAME | snake_case→lowerCamelCase | 인자·타입·설명·원 순서 |
| I3 OPAQUE_RENAME | 기능과 무관하게 배정한 tool_NN 이름 | 인자·타입·설명·원 순서 |

의미 ID는 내부 실행에만 사용했습니다. 모델은 노출 이름·설명·schema만 봅니다. typed 선언과 확인된 TASK API 문자열만 변환했고, 자유 텍스트·payload·모델 응답은 치환하지 않았습니다. 변환별 field path·before·after·역변환은 [mapping](specs/interfaces.json)에 있습니다. 설명은 원형 그대로이며 문서 읽기/수량·가격 읽기/상태 조회/곱셈/설정 교체/설정 검증을 구분합니다. opaque 이름 배정은 gold와 도구의 원래 위치를 사용하지 않았습니다.

정확히 등록된 이름만 내부 의미 ID로 해석한 다음 원 도구 함수에 dispatch했습니다. 옛 이름·오탈자는 UNKNOWN_TOOL이며 상태를 바꾸지 않습니다. 모델에게 돌려준 오류는 기존 `unknown tool/incorrect argument keys` 등 원형 문장입니다. sidecar의 오류 분류·allowed action set은 모델에게 보내지 않았습니다.

주 FUNCTIONAL_SUCCESS는 원 과제에서 요구한 읽기·계산 순서와 실제 patch·validation을 확인합니다. LEGACY_SUCCESS는 Case013의 원 최종 답 판정을 별도로 유지합니다. 추가적인 무관한 읽기 등은 진단에서 WRONG_TOOL로 기록될 수 있지만 그 사실만으로 episode를 실패 처리하지 않습니다. 동등한 곱셈 인자 교환도 허용합니다. 두 채점의 계약 차이는 [P0 검사](tests/test_contracts.py)로 확인했고, 이번 96개에서는 두 점수가 모두 같았습니다.

<a id="method"></a>
## 방법

Qwen/Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554`의 기존 WSL snapshot을 읽기 전용으로 사용했습니다. 실제 로컬 weight shard3개를 포함한12개 파일과 RTN/D16 각252개 저장 파일의 SHA256을 실행 전후 대조했습니다. 로컬 identity 확인이며 새 공식 외부 digest 대조는 아닙니다.

| Method | Projection files | Stored bytes |
|---|---:|---:|
| RTN | 252 | 2,043,746,208 |
| D16 | 252 | 2,274,662,304 |


embedding·lm_head·norm·attention·RoPE·tie 관계는 원형입니다. D16은 `FP32(X)*beta_inverse→BF16`와 `BF16(W_pre)`의 기존 adapter를 사용했습니다. weight_hat만 설치하거나 beta를 두 번 곱하지 않았습니다. artifact 누락은 실패이며 solver·encode·serialize·run_all은 실행 guard로 막았습니다.

greedy, enable_thinking=False, 최대 192 token, 최대 8 model calls·6 attempted tool calls, episode당 900초, context 24,576, batch 1입니다. SDPA·BF16·TF32=False·BF16/FP16 reduced reduction=False, 요청 내부 KV ON·턴 사이 cache OFF, logits_to_keep=1을 유지했습니다. 생성된 Python이나 shell은 실행하지 않고 여섯 등록 fixture 함수만 격리한 디렉터리에서 실행했습니다.

<a id="experiment"></a>
## 실험

기존 8개 과제는 document/combine/updates/fixture_patch 각2개이며 원래4K/8K 배정을 유지했습니다. 3방법×4인터페이스×8과제=96개 본 episode입니다. 별도 BF16/I0 AGENT-6/7 smoke2개는 분모에 넣지 않았습니다. I0 먼저 8개를 실행하고 나머지 24개는 동결 순서로 실행했습니다. model load 순서는 BF16→D16→RTN이었습니다.

interface seed6101601, 실행 seed6101602, bootstrap seed6101603입니다. P0의27개 고유 CPU 검사, 32개 oracle episode/48단계 비교가 통과한 뒤 입력·mapping·source를 동결했습니다. I0 입력 hash24개와 첫 출력 token IDs24개 모두 역사적 기록과 일치했고 legacy 점수도 같았습니다. [대조](results/p1_r1/historical_I0_comparison.json).

| 실행·검산 | 실제 수 |
|---|---:|
| 본 FREE episode / 누락 / 미관측 | 96 / 0 / 0 |
| 본 generation 요청 / 생성 token / tool attempts | 228 / 5,765 / 160 |
| 별도 smoke episode / generation 요청 / 생성 token | 2 / 4 / 100 |
| 누적 worker wall | 452.013 s |
| Torch peak allocated / reserved | 10,359,526,400 / 12,769,558,528 B |
| 새 solver·양자화·설치·모델 다운로드 / 재시도 | 0 / 0 |

worker wall은 load·decode·smoke·CPU·I/O·실패 시간을 포함하는 process wall이며 serving 지연이 아닙니다. 장치 free/total은 요청별 CUDA 표본으로 저장했고 Windows 전체 장치의 연속 peak는 측정하지 않았습니다. [자원 원자료](results/p1_r1/resource_summary.json).

<a id="results"></a>
## 결과

| Method | I0 original | I1 order | I2 lowerCamelCase | I3 opaque |
|---|---:|---:|---:|---:|
| BF16 | 8/8 | 8/8 | 8/8 | 7/8 |
| RTN | 8/8 | 8/8 | 8/8 | 8/8 |
| D16 | 6/8 | 6/8 | 5/8 | 8/8 |


분모는 각 칸의 기존 8개 과제입니다. [과제별96행](results/p1_r1/task_outcomes.csv), [전체 사건 탐색](demo/ko.html#episodes).

| Contrast | Signed interaction (pp) | Exploratory paired 95% CI (pp) | Role |
|---|---:|---|---|
| D16-minus-BF16 | +8.33 | [-8.33, +25.00] | primary |
| RTN-minus-BF16 | +4.17 | [+0.00, +12.50] | secondary |
| D16-minus-RTN | +4.17 | [-8.33, +16.67] | secondary |


주 비교 구간은 0을 포함합니다. RTN–BF16의 보조 구간도 0에 닿으므로 우위·동등성 판정에 사용하지 않습니다. 결측이 없어 주 효과의 미관측 최악/최선 범위는 동일한 +0.083333입니다.

| Method | Changed pairs /24 | S | All four interfaces successful tasks /8 |
|---|---:|---:|---:|
| BF16 | 1/24 | 4.17% | 7/8 |
| RTN | 0/24 | 0.00% | 8/8 |
| D16 | 3/24 | 12.50% | 5/8 |


D16은24개 원형→변경 쌍 중3개가 달라졌습니다. 방향을 나누면 I2에서 AGENT-4손실1개, I3에서 AGENT-6/7회복2개입니다. BF16은 I3 AGENT-3손실1개이고 RTN의 성공 여부는24쌍 모두 같았습니다. RTN도 호출 이력까지 불변인 것은 아닙니다.

| Method | Interface vs I0 | Both success | I0 only: loss | Changed only: gain | Both fail |
|---|---|---:|---:|---:|---:|
| BF16 | I1 | 8 | 0 | 0 | 0 |
| BF16 | I2 | 8 | 0 | 0 | 0 |
| BF16 | I3 | 7 | 1 | 0 | 0 |
| RTN | I1 | 8 | 0 | 0 | 0 |
| RTN | I2 | 8 | 0 | 0 | 0 |
| RTN | I3 | 8 | 0 | 0 | 0 |
| D16 | I1 | 6 | 0 | 0 | 2 |
| D16 | I2 | 5 | 1 | 0 | 2 |
| D16 | I3 | 6 | 0 | 2 | 0 |


| Family | Task IDs | Primary interaction (pp) | Leave this family out (pp) |
|---|---|---:|---:|
| combine | AGENT-2, AGENT-3 | +16.67 | +5.56 |
| document | AGENT-0, AGENT-1 | +0.00 | +11.11 |
| fixture_patch | AGENT-6, AGENT-7 | +33.33 | +0.00 |
| updates | AGENT-4, AGENT-5 | -16.67 | +16.67 |


family 하나를 제외한 주 효과 범위는0~+16.67pp입니다. task 전체12cell을 함께 재표집한10,000회 percentile95% 구간이며,4family의 외부 업무 모집단에 대한 확증 구간이나 p값은 아닙니다. 동일 I0가 세 차이에 쓰이는 의존성을 유지했습니다. 원형에서 둘 다 성공한6과제만 보면 D16의 새 손실1개와 BF16의 새 손실1개가 같은 signed 크기로 상쇄되므로, 그 선택조건부 비교도 보편적 불변성을 뜻하지 않습니다.

![Functional success](figures/p1_r1/success_matrix.svg)

높을수록 성공이 많습니다. 각 칸은8개 DEV이며 임의의 CI를 추가하지 않았습니다.

![Losses and gains](figures/p1_r1/paired_losses_gains.svg)

0아래는 원형 성공→실패 수,0위는 원형 실패→성공 수입니다.

![Interaction](figures/p1_r1/paired_interaction.svg)

축 단위는 percentage points입니다. task를 함께 재표집한 구간이며 양자화 모델의 일반 성능 순위가 아닙니다.

<a id="failures"></a>
## 실패·복구 분석

| 사례 | 실제 관측 | 판정·출처 |
|---|---|---|
| D16/I0 AGENT-6·7 | `ApplyFixturePatch`·`ValidateFixtureTest` 미등록 호출, 오류 후 같은 이름 재선택, 실제 수정 없이 valid:true | 실패. [AGENT-6 raw](results/p1_r1/workers/D16/attempt01/cells/D16__I0_ORIGINAL__AGENT-6.json) |
| D16/I2 AGENT-4 | 도구 없이 `{"status":"latest_status"}` | 이름 파싱 문제가 아닌 필요한 읽기·정답 실패. [raw](results/p1_r1/workers/D16/attempt01/cells/D16__I2_STYLE_RENAME__AGENT-4.json) |
| BF16/I3 AGENT-3 | record는11×3인데 등록된 계산 도구에10×5.5 전달, total55 | name은 유효하지만 인자 값·정답 실패. [raw](results/p1_r1/workers/BF16/attempt01/cells/BF16__I3_OPAQUE_RENAME__AGENT-3.json) |
| D16/I3 AGENT-6 | `tool_56`, `tool_89` 호출, 실제 설정 수정·성공 validation·최종 valid:true | 회복된 성공 사례. [raw](results/p1_r1/workers/D16/attempt01/cells/D16__I3_OPAQUE_RENAME__AGENT-6.json) |

| Method | Interface | Valid name / evaluable call objects |
|---|---|---:|
| BF16 | I0 | 12/12 |
| BF16 | I1 | 12/12 |
| BF16 | I2 | 12/12 |
| BF16 | I3 | 13/13 |
| RTN | I0 | 12/12 |
| RTN | I1 | 12/12 |
| RTN | I2 | 12/12 |
| RTN | I3 | 14/14 |
| D16 | I0 | 8/16 |
| D16 | I1 | 8/16 |
| D16 | I2 | 12/16 |
| D16 | I3 | 13/13 |


분모는 name을 평가할 수 있는 파싱된 호출 객체입니다. 도구를 부르지 않은 episode에100%를 주지 않았습니다. parsing 실패는 별도이며 UNKNOWN_TOOL을 의미적으로 맞았다고 자동 교정하지 않았습니다. 오류 label은 겹칠 수 있어 event 수와 실패 episode 수를 더하지 않습니다.

등록된 broad continuation 필드는 sidecar 오류 뒤의 다음 도구 사건도 포함합니다. 추가 CPU 분석에서는 **실제 error result 또는 negative validation을 받고 이후 모델 턴이 있는 경우**로 좁혀 따로 계산했습니다. 이는 사후 분모 명확화이며 원 등록 지표와 raw를 바꾸지 않았습니다. 해당6episode 중 이후 등록명 호출은2개였지만 올바른 필수 행동 회복과 최종 기능 성공은0개였습니다. [명시적 사후 분석](results/p1_r1/feedback_recovery.posthoc.json).

[72개 common-event-prefix 비교](results/p1_r1/common_event_prefixes.json)는 같은 초기 fixture와 실제 공통 사건 prefix에서만 다음 행동을 대응시킵니다. 다른 턴 번호를 억지로 맞추지 않습니다. 자유 실행 결과만으로 내부 의미 능력이나 토큰 margin 원인을 입증한 것은 아닙니다.

<a id="conclusion"></a>
## 결론

Q1의 등록 signed 주 비교는 불확정입니다. D16의 결과 변경률은 BF16보다 컸지만 회복과 손실이 공존합니다. Q2에서는 미등록 이름·잘못된 인자 값·실행 없는 답변이 서로 다른 시작점으로 관측됐습니다. Q3은 동일 오류 feedback과 성공 대조를 포함한 P2 진단을 검토할 근거가 있으며, 그 이후에 기존 ENUM·FEEDBACK의 interface별 별도 paired 비교를 고려할 수 있습니다. 이번에는 그 실험을 실행하지 않았습니다.

| 범위 | 상태 |
|---|---|
| 인터페이스 변환·동등성·fixture 실행·CPU 감사 | 실행 완료 |
| 기존8DEV의96개 FREE episode | 완료 |
| 새 task·별도 이름 배정 반복·다른 모델·D4 | 미실행 |
| P2 진단·P3 fresh64·CIPR 학습 | 미실행 |
| packed serving·고객 서비스·원격 게시 | 미실행 |

<a id="references"></a>
## 재현과 참고자료

[CPU 명령](REPRODUCE.ko.md), [source reuse](REUSE_MAP.md), [원형 보존](results/p1_r1/preservation.json), [검산](results/p1_r1/independent_audit.json), [검사 receipt](results/p1_r1/verification.json)를 제공합니다. 새 고유 CPU 검사는 27개, 기존 공통 package 재검사는 46개입니다. 반복 실행과 과거 Case01339개 검사를 합쳐 새 검사 수로 부르지 않았습니다.

최초 P0 실행의 1개 실패는 thirds의 binary exact equality를 기대한 test assertion 문제였고 freeze 전에 고쳤습니다. package 첫 실행의 3개 실패는 subprocess cwd가 달라 relative PYTHONPATH를 못 찾은 문제였으며 absolute 경로로 46개가 통과했습니다. 실패 로그도 보존했습니다. 실행 소스는 그 뒤 바꾸지 않았습니다.

CPU auditor는 저장 응답 228개를 파싱하고 실제 fixture에서 160개 도구 사건을 재적용해96개 endpoint와 주 효과·bootstrap을 다시 계산했습니다. 제외한 logits/weights의 모델 계산을 독립 재현한 것은 아닙니다. review ZIP은 모델·INT4 payload·환경·대형 tensor를 제외합니다. 코드·입력 fixture·경량 원자료·한영 탐색기는 포함합니다.

<details><summary>식별자와 출처</summary>

- Git base: `e647b474bdc9fa558d851bd9c998b664b0e33638`; 새 commit 없음.
- GPU execution source SHA256: `5a2c18000b4623c8c011390f71c8f3eb5fc91cf231d9905320e546a0dcc85a2b`. Git SHA와 다릅니다.
- [문맥 내 BPE·TASK 위치](provenance/tokenization.posthoc.json): 기존32개 입력을 재인코딩해 hash를 확인한 사후 입력 metadata이며 추가 모델 추론이 아닙니다.
- [source freeze](provenance/source_freeze.json), [model files](provenance/model_identity.json), [artifact identity](provenance/artifact_identity.json), [입력](provenance/input_manifest.json).
- [CIPR paper v1](https://arxiv.org/html/2610.20576v1), [고정 저자 README](https://github.com/ArtinTD/cross-interface-policy-regularization/tree/0465909ce219cdd1202affff75e41330697e4862), [실제 출처 확인 범위](provenance/source_audit.json).
- CIPR은 인터페이스 일관성을 학습하는 선행 연구입니다. 여기서는 학습·코드 차용 없이 이름·순서 변경의 실행 민감도를 평가했습니다. 이론·구현의 공식 재현을 주장하지 않습니다.
- Codex/ChatGPT가 구현·실험·별도 CPU 계산을 보조했습니다. 사람의 독립 심사나 외부 GPU 재현으로 표시하지 않습니다. [NOTICE](NOTICE.md).

</details>
