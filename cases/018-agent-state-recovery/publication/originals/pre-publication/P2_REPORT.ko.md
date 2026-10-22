# Case018 P2 — 상태 복구와 정확한 완료 보고의 차이

동일 설정의 신규 합성 HOLDOUT24에서 **최신 성공 검증 가드 R1의 Q/W 개선은 관측되지 않았습니다.** BF16 KV는 R0/R1 모두 Q24/24·W0/24, native NVFP4 KV는 모두 Q19/24·W5/24였습니다. R1은 NVFP4의 잘못된 완료 수용을5개 episode에서 막았지만 상태를 수리하거나 정확한 최종 보고를 늘리지는 못했습니다. 이때 추가 요청20회와 출력2,162 tokens를 사용했습니다.

Q는 종료 업무 값·보호·등록된 두 후속 절차의 적합성입니다. W는 Q에 더해 정확한 accepted final과 현재 성공 검증의 사전 관측을 요구합니다. **업무 값 복원과 정확한 완료 보고는 같은 결과가 아니었습니다.** 모델 점수가 아니라96고유 key·기술 gate·raw/token/독립 audit·보존으로 실행 완료를 판정했습니다.

## 설정·입력·동결

P0/P1의 공개 prompt·tool8·grammar·strict parser·가드·world·scorer·continuation·episode 예산은 byte 그대로 재사용했습니다. 새로 작성한 것은 P2 driver/생성·봉인·분모·누적 회계·감사 wrapper·집계입니다. [실행 계약](P2_PROTOCOL.ko.md), [설정](specs/p2_execution.json), [재사용 대응표](P2_REUSE_MAP.md)를 연결했습니다. 원 P1 보고서와 부정적 결과를 바꾸지 않았습니다.

등록 generator inventory-recovery-v1을 HOLDOUT seed61022002로 한 번 호출했습니다. A 잘못된 대상/B 중복/C 부분 적용 각8개입니다. ID를 제외한 family·원 수량 vector·사건 수량/phase의24개 scenario group은 DEV와 겹치지 않았습니다. 세 family template는 공유하므로 새로운 회사나 독립 template24개를 뜻하지 않습니다. 과제 선별·reroll·DEV 겨냥 난이도 조정은0입니다.

72개 CPU control(clean/no-recovery/scripted-correct) 모두 독립 재생과 기대 Q/W를 만족했습니다. control의 실제/reference continuation은 각각144개입니다. 별도24개 guard-twice script는 예산 여유 검증이며 모델 실측이 아닙니다. 최대 정답7/가드 포함9회, 실제 CPU 토큰화 최대 입력2,739·script 응답+EOS136으로 공통 한도 안이었습니다. 이 mock 사용량을 실제 token이나 모델 요청에 넣지 않았습니다.

생성 전 source·순서·96key·분석을 고정했고 seed61022012로 KV block K0→K1 안에서 task/policy를 shuffle했습니다. seal 시각은 **2026-10-22T03:00:57.343730+00:00**, 첫 HOLDOUT 전 execution freeze는 **2026-10-22T03:04:15.758725+00:00**입니다. 둘 다 로컬 자기기록이며 외부 시각 인증이 아닙니다. seal과 freeze에 HOLDOUT 모델 출력0, 이미 관측한 DEV24/응답180/TECH6을 명시했습니다. freeze 전 새 TECH2, K1 reload 뒤 새 TECH1이 추가됐습니다. 원 schema의 속성 순서를 포함한 bytes로 컴파일했으며 정렬된 hash 표시용 JSON을 다시 문법으로 컴파일하지 않았습니다.

매 episode는 동일 오류의 새 fixture에서 시작합니다. 최초 public messages/schema/error/fault/실제 token IDs는 같은 task의4조건에서 일치했습니다. 이후 자유 rollout 이력은 갈라질 수 있습니다. 정책/KV 이름·gold·정답 script·oracle은 모델 입력에 추가하지 않았습니다. 생성 전 관측 ID를 고정하며 그 생성에서 실행된 결과를 소급 관측으로 넣지 않았습니다.

## 지표와 HOLDOUT 결과

D=가드에 거절된 경우도 포함한 recovered:true 선언, B=업무 값 복원, P=전 과정 보호, Fresh=현재 scope/digest/version의 성공 validation을 생성 전에 관측함입니다. C1/C2는 두 고정 CPU 후속 절차 적합이며 Q=B∧P∧C1∧C2가 주 지표입니다. W=Q∧accepted final∧정확한 final∧Fresh입니다. accepted final이 없으면 Fresh는 마지막 생성 전 진단입니다.

각 행은 **24개 같은 task**, 전체96episode입니다.96개 독립 업무로 세지 않습니다.

| KV | 정책 | D | B | P | Fresh | C1 | C2 | Q | W |
|---|---|---|---|---|---|---|---|---|---|
| BF16 KV | R0 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 0 |
| BF16 KV | R1 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 0 |
| NVFP4 KV | R0 | 24 | 19 | 23 | 19 | 19 | 19 | 19 | 5 |
| NVFP4 KV | R1 | 24 | 19 | 23 | 19 | 19 | 19 | 19 | 5 |

| KV·정책 | A Q/W (8) | B Q/W (8) | C Q/W (8) |
|---|---|---|---|
| BF16 KV R0 | 8/0 | 8/0 | 8/0 |
| BF16 KV R1 | 8/0 | 8/0 | 8/0 |
| NVFP4 KV R0 | 7/3 | 8/1 | 4/1 |
| NVFP4 KV R1 | 7/3 | 8/1 | 4/1 |

총 Q86/W10은 반복 조건의 합계일 뿐 일반 성공률로 제시하지 않습니다. BF16의 Q 천장과 W 바닥이 함께 나타났고, NVFP4는 Q가 더 낮지만 W가 더 높았습니다. KV 어느 한쪽을 모든 지표에서 우월하다고 부를 수 없습니다. DEV 각6의 P1(BF16 Q6/W2, NVFP4 Q5/W0)과 이번 각24를 합쳐30과제 성공률을 만들지 않습니다.

### 짝 비교

| 후보−기준 | 지표 | 동시 성공 | 손실 | 회복 | 동시 실패 | 차이 pp | 탐색95% pp |
|---|---|---|---|---|---|---|---|
| K0_R1-R0 | Q | 24 | 0 | 0 | 0 | 0.00 | [0.00, 0.00] |
| K0_R1-R0 | W | 0 | 0 | 0 | 24 | 0.00 | [0.00, 0.00] |
| K1_R1-R0 | Q | 19 | 0 | 0 | 5 | 0.00 | [0.00, 0.00] |
| K1_R1-R0 | W | 5 | 0 | 0 | 19 | 0.00 | [0.00, 0.00] |
| R0_K1-K0 | Q | 19 | 5 | 0 | 0 | -20.83 | [-33.33, -8.33] |
| R0_K1-K0 | W | 0 | 0 | 5 | 19 | 20.83 | [8.33, 37.50] |
| R1_K1-K0 | Q | 19 | 5 | 0 | 0 | -20.83 | [-33.33, -8.33] |
| R1_K1-K0 | W | 0 | 0 | 5 | 19 | 20.83 | [8.33, 37.50] |

등록한 family-stratified paired task bootstrap10,000회/seed61022018을 그대로 사용했습니다. 한 task의4조건을 같은 재표집에 유지했고 rank(n−1)p 선형 보간 percentile을 적용했습니다. 별도 flat-delta/RNG index 구현으로 같은 수치를 교차검산했습니다. R1−R0의 [0,0] 구간은 이24개에서 짝 차이가 없기 때문에 나온 경험적 bootstrap 결과이며 일반 동등성/효과 없음 증명이 아닙니다. 모든 구간은 고정 모델·generator의 작은 합성 과제에 대한 탐색적 pointwise 구간이며 다중 비교/일반 모집단 인증이 아닙니다.

## 실제 사건과 남은 실패

**Q1/W0은76개 episode**이며 모두 final 수량이 종료 수량과 달랐습니다. 별도 사후 대조에서76개 모두의 final 수량 map이 **가장 처음 관측한 read_records 수량 map**과 일치했습니다. 이는 원문에서 확인한 수치 관계입니다. 모델 내부 memory/attention의 원인이나 KV 반올림이 특정 인지 오류를 만들었다는 증거로 확대하지 않습니다. [원문 기반 사후 표](results/p2_r1/analysis/final_value_relation.json)는 모델 feedback에 사용하지 않았습니다.

- **HOLDOUT-A00_K0_R0 (R1도 동일)**: g003에서6/6 보상, g004 release 후 실제 x/y는43/28로 복원됐습니다. g005의 e005는 현재 revision3/valid:true이며 final 전 관측됐습니다. g006 final은 최초 read의37/34를 적어 Q1/Fresh1/W0입니다. 검사기는 이를 관대하게 교정하지 않았습니다.
- **HOLDOUT-A00_K1_R0/R1의 W1**: 같은 보상/release 뒤 g005에서 현재 records를 다시 읽었습니다. e005의 수량43/28과 e006의 현재 성공 검증을 본 g007 final이 정확했습니다. 추가 read는 모델이 생성한 실제 행동이며 host가 자동 삽입한 단계가 아닙니다. 같은 최초 입력 이후의 이력 분기입니다.
- **HOLDOUT-A06_K1_R0/R1의 Q0/P0**: incident 수량은2인데 모델은 source_restore40/destination_remove30을 적용했습니다. checkpoint x/y42/28 대신 종료 x/y80/0이 됐고 SKU 총합을10 늘려 허용 중간 상태 보호 계약도 위반했습니다. 여러 valid:false를 받은 뒤 R0는 완료를 수용받았고, R1은 세 final을 거절했습니다. R1의16번째 요청도 거절된 final로 끝났으며17번째 요청은 없습니다. 무관 보호 SKU-U를 바꿨다는 뜻은 아닙니다.
- **HOLDOUT-C01_K1_R0/R1**: g000 read_operation에서 DEBITED/quantity4를 받았는데 g001에서 출발+4와 도착−4를 함께 적용했습니다. checkpoint s/d62/16 대신62/12입니다. 공개 검증은 반복해서 valid:false/CONSERVATION을 반환했습니다. R0 final의 현재 수량은 실제62/12와 같아도 checkpoint 복구 선언은 틀렸습니다. R1은 이를 수리하지 않고 g010/g012/g014 final3회를 거절한 뒤 GUARD_LIMIT으로 종료했습니다.

NVFP4의 Q 실패 고유 task는 **A06,C01,C04,C05,C06**이며 양 정책에서 같습니다. W 성공 고유 task는 **A00,A01,A04,B01,C03**으로 양 정책에서 같습니다. Q1/W0의76개와 Q0의10개 episode를 구분합니다. B1인데 후속만 실패하거나 Fresh1/B0인 모델 사례는 미관측입니다. P0 synthetic 반례를 실제 모델 사례에 더하지 않습니다.

R1이 차단한 것은15개 부적격 final 시도/5개 episode의 완료 수용입니다.5개를 복구 성공으로 바꾸지는 못했습니다. D에는 거절된 recovered:true도 남으므로 선언 자체를 제거했다고 쓰지 않습니다. terminal은 ACCEPTED_FINAL91, GUARD_LIMIT5이며 전송 오류·timeout·계약 parse 오류·cap 종료는 없었습니다. schema와 최신 검증은 final 수량의 진실을 보증하지 않습니다.

## 비용

| KV·정책 | 요청/완전 응답 | 입력 tokens | 출력 tokens | 가드 거절 | episode 초 합 |
|---|---|---|---|---|---|
| BF16 KV R0 | 168/168 | 323,375 | 10,953 | 0 | 164.144812 |
| BF16 KV R1 | 168/168 | 323,375 | 10,953 | 0 | 164.157640 |
| NVFP4 KV R0 | 186/186 | 369,881 | 11,568 | 0 | 179.978763 |
| NVFP4 KV R1 | 206/206 | 439,813 | 13,730 | 15 | 213.922486 |

HOLDOUT의728요청/완전 응답728, 입력1,456,444/output47,204 tokens는 raw IDs와 서버 usage가 일치했습니다. 출력은 EOS 등 제어 token을 포함합니다. 입력은 매 턴 재전송 history의 합계입니다. 초기 입력은 task별1,203–1,246 tokens, 관측 최대 입력4,209/응답257/episode output1,204/요청16회로 공통 한도 안이었습니다. history truncation·응답 repair·자동 retry는0입니다.

| KV·정책 | 입력/Q | 출력/Q | episode 초/Q | 입력/W | 출력/W | episode 초/W |
|---|---|---|---|---|---|---|
| BF16 KV R0 | 13,473.96 | 456.38 | 6.8394 | undefined | undefined | undefined |
| BF16 KV R1 | 13,473.96 | 456.38 | 6.8399 | undefined | undefined | undefined |
| NVFP4 KV R0 | 19,467.42 | 608.84 | 9.4726 | 73,976.20 | 2,313.60 | 35.9958 |
| NVFP4 KV R1 | 23,148.05 | 722.63 | 11.2591 | 87,962.60 | 2,746.00 | 42.7845 |

실패 포함 전체24개의 비용을 각 성공수로 나눈 값입니다. 성공 episode만의 평균 latency가 아니며 입력과 출력을 같은 연산 비용으로 합치지 않습니다. 성공0인 비용/W는 undefined입니다. NVFP4 R1은 같은 Q19/W5에서 입력+18.91%, 출력+18.69%, episode wall+18.86%를 사용했습니다. worker 비용을 정책별로 임의 배분하지 않았고, 이 C1 rollout 시간을 KV kernel 순수 speedup/serving 처리량/전력·과금으로 부르지 않습니다.

새 TECH는3회 모두 완전 응답이며 input2,901/output39 tokens입니다. P2 전체는 **731요청/731완전 응답**, input1,459,345/output47,243 tokens로 관측됐습니다. 기술 직접 forward6회와 엔진 내부 load/compile/warmup은 API 생성과 구분하며 후자의 횟수는 계측하지 않았습니다. 새 취소 요청은 없습니다.

P1의186요청/184완전 응답과 합치면 누적917요청/915완전 응답, **관측 output 하한57,987 tokens**입니다. P1 취소2건의 실제 엔진 token은 계속 미관측이므로 완전한 누적 output 합계는 null입니다. P1 DEV/TECH 비용을 HOLDOUT 분모에 넣지 않았습니다.

P2 GPU worker **855.069382600초**, P1 부채439.692244164초 포함 누적 **1,294.761626764초**, 공통3,600초 잔여 **2,305.238373236초**입니다. 과거 기술 실패185.476104969초도 지우지 않았습니다. worker는 load/compile·TECH·I/O·실패·후속·소유 종료를 포함하고 episode wall722.203702065초와 겹칩니다. 단순히 합쳐 총비용으로 만들지 않습니다.

CPU fixture 준비 wall 합0.036745733초, continuation wall 합0.823829570초를 별도로 기록했습니다. 독립 audit 함수 wall4.424403초/process CPU4.382086초/peak RSS63,600KiB입니다. tokenizer 검사 loop는2.213713초/CPU2.193370초, process peak RSS921,668KiB이며 loading/compile은 loop 시간에서 제외됐습니다. CPU control 준비 전체 wall/CPU/RSS는 별도 계측하지 않아 NOT_MEASURED입니다. P1의 미측정 CPU 비용을 소급 생성하지 않았습니다.

controller peak RSS1,193,972KiB, worker 관측 CUDA allocated peak13,561,159,168bytes(양쪽), reserved K0=13,767,802,880/K1=13,807,648,768bytes였습니다. allocator 관측이며 연속 장치 전체 peak/전력은 미측정입니다.

## 새 worker 기술 확인과 보존

동일 로컬 Qwen3-4B-Instruct-2507 revision cdbee75f17c01a7cc42f958dc650907174af0554/BF16 weights, vLLM0.32.0/Torch2.13.0 CUDA13.0/Transformers5.19.0/FlashInfer0.7.0.post1/xgrammar0.2.8을 사용했습니다. 실제 로드 경로·BF16 dtype/파라미터 수·표본 hash와 작은 config/tokenizer/index/runtime bytes를 확인했고 weight 목록·크기/역사 hash 근거를 유지했습니다. 전체 weight 현재 SHA를 다시 계산하지 않았습니다.

세 worker 모두 SM120/TP1/GRAPH/FlashInfer, maxlen28672/seqs4/batched2048, prefix/async OFF, TF32 OFF의 같은 설정이었습니다. 실제36층 cache와 prefill/decode finite 출력·원 grammar mask·NVFP4 write/read kernel·SSE/IDs/usage를 fresh 확인했습니다. 기술 전용 uncompiled 관측 바인딩은 복원했고 main GRAPH를 바꾸지 않았습니다. 매 생성 전체 logits의 유한성을 독립 검증했다고 확대하지 않습니다. 문법 mask의 -inf는 정상 차단 값입니다.

BF16 실제 pool4,293,918,720bytes·147,456B/token·29,120slots, NVFP4는4,294,508,544bytes·41,472B/token·103,552slots였습니다. 모든 NVFP4층의 packed uint8 data/FP8 block scales와 FA2 압축 cache reader를 확인했습니다. 약3.556배는 같은 약4GiB pool의 token 슬롯 수용량이며 전체 VRAM 절감·단일100K 문맥·품질/속도 보장이 아닙니다. 반올림만의 인과 효과가 아닌 KV/커널을 포함한 serving 구성 비교입니다.

독립 CPU 재생은 **96/96, endpoint 차이0/raw 차이0**이며728개 실제 모델 요청의 재토큰화·원 순서 grammar prefix/EOS·special decode가 PASS입니다. TECH3개의 raw/token도 별도 CPU 검산했습니다. ReferenceState·SSE 조립은 별도이며 strict parser/serialization/공개 builder/wire rendering을 공유합니다. P2 wrapper의 DEV loader/scenario fingerprint는 split metadata 검사에만 사용하고 World/scoring의 상태 계산을 endpoint 답으로 사용하지 않습니다. 외부 사람/다른 머신의 GPU 재현은 아닙니다.

CPU unittest는 이번 통합에서 **59개 일괄 PASS(기존48 재실행+신규11)**입니다. P1 당시의44일괄+4개별 이력을 소급 변경하지 않습니다. HOLDOUT 생성 전 실행해 원 P0의 미생성 전제도 지켰습니다. P0 전체 실험/P1 GPU/과거 모델 결과를 재실행하지 않았습니다.

P0/P1 기존606파일·P1 source39·raw343파일/111,584,798bytes·P0 원229파일 snapshot·P1 실패/final source snapshot·Case017 선택81·helper8·선택 runtime/모델 파일 범위가 모두 그대로였습니다. 전 weight/ZIP/타 프로젝트 전체 보존을 감사했다는 뜻은 아닙니다. Git HEAD/branch/status는 시작과 같고 staged0/Case018 밖 tracked diff0입니다.3개 소유 서버의 host PID/process group·port8197·GPU compute 잔류가 없습니다. upstream 자체 EngineCore force-kill/semaphore 경고는 그대로 남겼으며 경고 없는 graceful teardown으로 부르지 않습니다. 다른 서비스는 변경/종료하지 않았습니다.

## identity·재현·종료

Git base/HEAD는83e1cef6dd60384931a38e016ea5e8d069d40674입니다. P1 own source39의 e19f15776ee4e74a37849e9fa6293e37a3d520bfd0221eb8b3545556f0e9af1d와 P1 실행 e60dc421c2dc28c59e29ab26cb9faa6a2893208d5399577eed08d9745211ee7b는 보존했습니다.

새 P2 own46 source는 **46b2c1dcd90fd368e4a83afff1f5b06954b5e4e699bc856f8c0589ac24b351bd**, dependency8/runtime 포함 실행 identity는 **44287b4091ac8e9e892a263b5770200a140f9d90f97ccef19dca203a1d5241a2**입니다. commit이 아닌 working source bundle입니다. [입력 seal](results/p2_r1/input_seal.json), [실행 freeze](results/p2_r1/execution_freeze.json), [공통 facts](results/p2_r1/analysis/facts.json), [96행 장부](results/p2_r1/analysis/episode_ledger.csv), [짝 표](results/p2_r1/analysis/paired.csv)와 [실제 재현 명령](REPRODUCE.P2.ko.md)을 연결했습니다. 결과 보고서와 사후 수량 진단은 출력 후 작성했으며 사전등록으로 표시하지 않습니다.

upstream 모델/엔진/커널과 직접 작성한 가드·평가기·계측·P2 연결/분석을 구분합니다. 원문 원리 대응 범위는 P0의 SOURCE_MAP이며 논문 GUI/on-policy 실험 전체를 재현한 것이 아닙니다. 두 유한 후속 절차 성공은 모든 미래 상태 동등성이나 자율 후속 능력 증명이 아닙니다. 이 결과는 해당 신규 합성 HOLDOUT과 고정 설정의 탐색 측정입니다. Codex가 구현/검산/문서를 지원했습니다.

P2에서 종료합니다. 추가 seed/정책/원인 실험·P3/UI·사이트·Git stage/commit/push/PR/merge/Pages/Drive는 NOT_RUN입니다.
