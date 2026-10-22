# Case018 P1 — 실제 서버 기술 검증과 DEV24

이번 DEV에서는 최신 검증 완료 가드(R1)가 상태 복구와 후속 절차 적합성 Q를 개선하지 않았습니다. BF16 KV의 R0/R1은 각각 Q **6/6**, native NVFP4 KV는 각각 **5/6**입니다. 검증된 복구 workflow W는 BF16에서 각각 **2/6**, NVFP4에서 각각 **0/6**이었습니다. 상태가 복원돼도 final에 이전 수량을 적은 사례가 남았습니다. NVFP4 C01에서는 잘못된 보상 후 공개 검증이 실패했고, R1은 완료를 막았지만 상태를 고치지는 못했습니다.

P1은 기술 gate·24개 예정 key·원응답/토큰·독립 재생·보존을 완료했습니다. 낮은 W를 기술 gate 실패나 실행 미완료로 바꾸지 않습니다. HOLDOUT 입력/gold 생성·봉인, P2/P3, Git 게시와 Pages는 **NOT_RUN**입니다.

## 무엇을 비교했는가

DEV6은 잘못된 대상 수정 A2, 중복 적용 B2, 부분 적용 C2입니다. 같은 합성 checkpoint와 CPU 주입 오류를 네 조건에 제공했습니다. 자연 발생 모델 오류율을 측정한 설계가 아닙니다. 모든 모델 가중치는 BF16입니다.

| 구성 | KV 저장 | 완료 처리 |
|---|---|---|
| K0_R0 | BF16 | schema에 맞는 final 수용, 동일 scorer로 사후 판정 |
| K0_R1 | BF16 | 현재 scope/digest/version의 성공 validation을 실제 관측해야 final 수용 |
| K1_R0 | native NVFP4 | R0와 같음 |
| K1_R1 | native NVFP4 | R1과 같음 |

모든 응답에 원 P0 공개 JSON schema와 같은 strict host parser를 적용했습니다. grammar는 공개 도구/키/자료형/정적 범위를 제한하고 정답 ID·수량·다음 행동을 공급하지 않습니다. 오류 후 JSON 수리, 자유 출력 fallback, 재생성은 없습니다. episode마다 새 error fixture를 복제했습니다. 최초 messages/schema/state/fault/실제 token IDs는 같은 task의 네 조건에서 일치했습니다. 이후 rollout 이력은 달라질 수 있습니다.

R1의 가드는 hidden gold를 보지 않습니다. 부적격 final 첫 두 번에는 고정 공개 feedback을 주고 세 번째는 `GUARD_LIMIT`으로 종료합니다. 요청/parsed action 각16회, 응답512/episode8192 output tokens, 문맥24576, 요청180초/episode900초를 유지했습니다. 모델 성공수 확대 기준은 P0의 `null` 그대로입니다.

## 지표를 함께 읽기

D는 `recovered=true` 선언입니다. 거절된 final에서 한 선언도 포함하며 진실성을 뜻하지 않습니다. B는 종료 snapshot의 업무 값 복원, P는 전 과정의 보호 계약, F는 현재 대상의 성공 validation을 생성 전에 관측했는지입니다. accepted final이 없으면 F는 마지막 실제 생성 직전의 진단입니다.

C1/C2는 종료 상태와 참 checkpoint의 독립 복제에서 실행한 두 고정 CPU 후속 절차의 적합성입니다. Q=`B AND P AND C1 AND C2`가 주 지표입니다. W는 Q와 함께 accepted final·정확한 final 내용·F를 요구합니다. **Q는 종료 상태와 등록한 후속 절차에 적합**, W는 **검증된 복구 workflow 완주**를 뜻합니다. CPU 후속 절차를 모델이 자율적으로 수행했다고 부르지 않습니다.

## DEV24 결과

각 행은 같은 6개 task입니다. 네 행은 24개 episode이며 독립 업무 24개가 아닙니다.

| KV | 정책 | D | B | P | F | C1 | C2 | Q | W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | R0 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 2 |
| BF16 | R1 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 2 |
| NVFP4 | R0 | 6 | 5 | 6 | 5 | 5 | 5 | 5 | 0 |
| NVFP4 | R1 | 6 | 5 | 6 | 5 | 5 | 5 | 5 | 0 |

| KV·정책 | A Q/W (각2) | B Q/W (각2) | C Q/W (각2) |
|---|---:|---:|---:|
| BF16 R0 | 2/2 | 2/0 | 2/0 |
| BF16 R1 | 2/2 | 2/0 | 2/0 |
| NVFP4 R0 | 2/0 | 2/0 | 1/0 |
| NVFP4 R1 | 2/0 | 2/0 | 1/0 |

`ACCEPTED_FINAL` 23회는 final 수용을 뜻하며 업무 성공 label이 아닙니다. 나머지 1회는 NVFP4 C01 R1의 `GUARD_LIMIT`입니다. DEV parse/transport/timeout/cap 종료는 없었습니다. 최신 검증을 포함한 정답 상태 Q1/W0은 18개 episode에서 관측됐으며, 모두 final 수량과 실제 종료 수량이 달랐습니다.

### 같은 task의 짝 비교

| 후보−기준 | 지표 | 동시 성공 | 손실 | 회복 | 동시 실패 |
|---|---|---:|---:|---:|---:|
| BF16 R1−R0 | Q | 6 | 0 | 0 | 0 |
| BF16 R1−R0 | W | 2 | 0 | 0 | 4 |
| NVFP4 R1−R0 | Q | 5 | 0 | 0 | 1 |
| NVFP4 R1−R0 | W | 0 | 0 | 0 | 6 |
| R0 NVFP4−BF16 | Q | 5 | 1 | 0 | 0 |
| R0 NVFP4−BF16 | W | 0 | 2 | 0 | 4 |
| R1 NVFP4−BF16 | Q | 5 | 1 | 0 | 0 |
| R1 NVFP4−BF16 | W | 0 | 2 | 0 | 4 |

등록한 탐색 bootstrap은 이번 작은 DEV에서 계산하지 않았고, 짝 개수의 기술 통계만 제시합니다. 차이 0은 동등성 인증이 아닙니다. 동일 BF16 weights여도 KV 저장뿐 아니라 attention 커널 경로를 포함한 serving 구성 비교입니다.

## 원문으로 연결되는 실제 실패

아래 위치는 `results/p1_r5/episodes/` 기준입니다. 생략한 raw 전체는 각 JSON의 `requests` 및 별도 transport 파일에 있습니다. 새 합성 응답을 실측에 넣지 않았습니다.

- **A00_K1_R0: Q1/F1/W0.** `DEV-A00_K1_R0.json`, g003은 `source_restore:6,destination_remove:6`을 실행했고 g004에서 활성 처리를 해제했습니다. g005의 e005는 현재 revision3/valid:true이며 g006 생성 전에 관측됐습니다. 실제 x/y 수량은 **42/26**으로 복원됐지만 final은 **36/32**를 적었습니다. 가드가 검사하는 freshness와 답의 수량 정확성은 다릅니다. 같은 task R1도 동일한 실패입니다.
- **B00_K0_R0: Q1/W0.** `DEV-B00_K0_R0.json`, g003의 추가 중복 효과 3 보상과 g004 release 뒤 s/d는 **67/24**입니다. e005의 성공 검증을 본 final은 **64/27**을 적었습니다. 정상 1회 완료 checkpoint는 복원됐고 후속 두 절차도 맞았지만 최종 답은 틀렸습니다. C00 BF16에서도 실제 s=58, final s=53이었습니다.
- **C01_K1_R0: D1/B0/F0/Q0.** `DEV-C01_K1_R0.json`, g003은 부분 오류인데도 `source_restore:4,destination_remove:4`를 적용했습니다. 올바른 부분 오류 보상은 출발 수량만 되돌리는 것입니다. g005–g007의 e005/e006/e007은 모두 valid:false였습니다. 그 결과를 관측한 뒤에도 g008에서 recovered:true를 선언했습니다. 종료 s/d=**43/20**, final=**39/24**이며 두 후속 절차도 실패했습니다. 공개 불변식 위반을 반복 검증하는 것은 수리가 아닙니다.
- **C01_K1_R1: 같은 업무 실패, 완료 차단과 추가 비용.** `DEV-C01_K1_R1.json`은 같은 보상 뒤 g008/g010/g012의 final을 거절했습니다. 첫 두 번 feedback, 세 번째 `GUARD_LIMIT`입니다. 중간에 새 검증을 불렀지만 모두 실패했으므로 F0/Q0을 유지했습니다. 요청은 R0 9회→R1 13회입니다. accepted final은 없으며 F는 마지막 생성 전 진단입니다.

이번 DEV에는 **B1인데 후속 절차만 실패**하거나 **F1/B0**인 모델 사례가 없었습니다. 그런 반례를 검증한 P0 CPU script를 P1에서 관측한 모델 사례로 옮기지 않습니다. R1이 업무 값의 올바름이나 final 수량을 직접 강제하지 않는다는 범위가 실제 결과에도 드러났습니다.

## 실제 비용

입력은 매 턴 전체 history를 재전송한 token 합계, 출력은 EOS 등 제어 token을 포함한 반환 ID 개수입니다. 아래 비용은 실패·거절을 포함한 각 6개 전체입니다.

| KV·정책 | 요청/완전 응답 | 입력 tokens | 출력 tokens | 가드 거절 | episode wall 합(초) |
|---|---:|---:|---:|---:|---:|
| BF16 R0 | 44/44 | 86,059 | 2,866 | 0 | 42.938460 |
| BF16 R1 | 44/44 | 86,059 | 2,866 | 0 | 42.961538 |
| NVFP4 R0 | 44/44 | 84,733 | 2,298 | 0 | 36.389180 |
| NVFP4 R1 | 48/48 | 96,781 | 2,662 | 3 | 42.816407 |
| 합계 | **180/180** | **353,632** | **10,692** | **3** | **165.105585** |

| KV·정책 | 입력/Q | 출력/Q | episode 초/Q | 입력/W | 출력/W | episode 초/W |
|---|---:|---:|---:|---:|---:|---:|
| BF16 R0 | 14,343.17 | 477.67 | 7.1564 | 43,029.50 | 1,433.00 | 21.4692 |
| BF16 R1 | 14,343.17 | 477.67 | 7.1603 | 43,029.50 | 1,433.00 | 21.4808 |
| NVFP4 R0 | 16,946.60 | 459.60 | 7.2778 | undefined | undefined | undefined |
| NVFP4 R1 | 19,356.20 | 532.40 | 8.5633 | undefined | undefined | undefined |

이는 성공한 episode만의 평균 latency가 아닙니다. 입력과 출력 tokens를 같은 연산량으로 합치지 않습니다. 비용/Q와 비용/W는 서로 다른 분모입니다. NVFP4 R1은 Q/W 변화 없이 요청4회·출력364 tokens·입력12,048 tokens가 추가됐습니다. episode wall 차이는 이 짧은 C1 실행의 관측이며 serving speedup/처리량/전력·과금으로 해석하지 않습니다.

DEV 요청당 최대 관측은 입력3,250/output205 tokens, episode 최대 output795, 요청 최대13회로 상한 안이었습니다. 최초 입력은 task별 1,222–1,237 tokens였습니다. P0 script의 실제 CPU 토큰화는 최대 입력2,530, 응답+EOS134이며 정답7회/가드 여유 포함9회의 타당성을 확인했습니다. 이는 모델 응답 수가 아닙니다.

기술 생성은 **6회**(완전 응답4, 의도한 취소 timeout2)입니다. 앞선 기술 실패에서 얻은 응답도 포함합니다. 완전 TECH 응답의 입력3,868/output52 tokens는 관측됐고, 두 취소 요청은 SSE/usage를 받지 못해 실제 엔진 token 비용이 미관측입니다. 따라서 전체 요청 **186회**, 완전 응답 **184회**, 관측 output 하한 **10,744 tokens**이며 전체 output을 정확히 10,744라고 쓰지 않습니다. DEV 운송 오류는0입니다. 직접 기술 forward12회와 엔진 내부 load/compile/warmup은 생성 API와 별개이며 후자의 횟수는 계측하지 않았습니다.

공통 worker 비용은 전 시도 **439.692244164초**, 잔여 **3,160.307755836/3,600초**입니다. 실패한 기술 시도185.476104969초와 최종 시도254.216139195초를 포함합니다. load/compile·TECH·실패·I/O·DEV·소유 서버 종료가 포함되고 episode/후속 시간과 겹칩니다. DEV 후속 CPU 절차 wall 합0.178399189초는 별도이며, 독립 audit의 wall/CPU/RSS는 별도 계측하지 않아 NOT_MEASURED입니다.

최종 시도 controller peak RSS는1,165,652KiB입니다. worker의 관측 peak CUDA allocated는 양쪽13,561,159,168bytes, reserved는 K0 13,767,802,880/K1 13,889,437,696bytes였습니다. allocator 관측값이며 장치 전체 peak VRAM이 아닙니다. 종료 후 장치 점유 약1,784MiB는 기존 display/background가 포함된 단면입니다. 연속 device peak/CPU 사용량/전력은 미측정입니다.

## 이번 runtime에서 관측한 기술 gate

Qwen/Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554`, RTX5080 SM120, TP1, vLLM0.32.0/Torch2.13.0(CUDA13.0)/Transformers5.19.0/FlashInfer0.7.0.post1/xgrammar0.2.8을 재사용했습니다. 설치·다운로드·weight INT4 로드는0입니다. 모델 로딩 경로와 실제 파라미터 4,022,468,096개가 BF16인 것을 관측했고, 작은 config/tokenizer/index 및 runtime source/METADATA hash를 대조했습니다. weight 목록·크기/역사적 hash는 보존했지만 weight 전체를 재해시하지는 않았습니다.

| 항목 | BF16 KV | native NVFP4 KV |
|---|---:|---:|
| 실제 KV pool bytes | 4,293,918,720 | 4,294,508,544 |
| bytes/token | 147,456 | 41,472 |
| blocks (block16) | 1,820 | 6,472 |
| token slots | 29,120 | 103,552 |
| 실제 저장/층 | BF16, 36층 | packed uint8 + FP8 e4m3 block scales, 36층 |
| 기술 prefill/decode 유한 출력 | 36/36 각각 | 36/36 각각 |

slots 비율 약3.556배는 약4GiB pool의 저장 수용량입니다. 전체 VRAM 감소·단일100K 문맥 지원·품질 동등성 주장이 아닙니다. max_model_len28672, max_num_seqs4, batched2048, GRAPH/FlashInfer/chunked prefill ON, async/prefix cache OFF, TF32 OFF/BF16 reduction ON을 공통 적용했습니다. 실제 V2 runner를 사용했고 legacy 전환은 없습니다.

G2는 실제 V2 grammar mask의 원 함수를 호출한 뒤 차단을 관측했습니다. 최초 logit151,936개 중151,929개가 -inf로 바뀌고 유한 후보7개가 남았으며 NaN은 없었습니다. 이 -inf는 문법 차단 값입니다. 모든 생성 위치에서 full-vocabulary unmasked logits 유한성을 독립 검사한 것은 아닙니다. 실제 tokenizer+xgrammar CPU 경계23개와 원 schema의 prefix 재생180요청도 PASS입니다.

G3는 모든36층의 NVFP4 view/reader flag와 실제 `NVFP4QuantizeAppendPagedKVCacheWithSlotMappingKernel`, `BatchPrefillWithPagedKVCacheKernel<...__nv_fp4x2_e2m1...>` 이름을 관측했습니다. FA2가 압축 cache를 읽는 경로이며 BF16 fallback으로 보지 않습니다. 기술 전용 unpadded prefill8/decode1에서 실제 attention 구현 출력과 반환 hidden state가 유한했습니다. 기술 probe의 graph NONE/skip_compiled 설정은 finally 복원했고 DEV의 GRAPH를 바꾸지 않았습니다. 전체 DEV 매토큰 attention 출력을 동기화해 검사한 것은 아닙니다.

G4의 의도한 0.05초 timeout은 대기/연결/stream을 포함했습니다. K1은 waiting1→0→0, K0는0→0의 실제 running/waiting gauge로 drain을 확인했습니다. 없는 abort API를 호출하지 않았고 client close만으로 취소 완료를 단정하지 않았습니다. 마지막 두 DEV block도 drain 후 정상 종료했습니다. 소유 서버7개 모두 종료 exit0, 동일 PID/start-time 잔류0, 포트8196 listener0, 승인된 GPU query에서 compute PID0을 확인했습니다. 다른 서비스는 종료하지 않았습니다.

종료 방식은 소유 process group에 SIGTERM을 보낸 것입니다. API 부모는 exit0이었으나 upstream teardown 로그에는 자체 EngineCore의 `force killing remaining process`와 semaphore 정리 경고가 있습니다. 요청 처리 완료/drain·잔류 PID/포트 해제와 이 경고를 구분해 보존했습니다. 경고 없는 완전한 graceful teardown을 입증했다고 표현하지 않습니다. 외부 evidence의 `shutdown_log_scope.json`에 해당 원문/위치를 남겼습니다.

## 시도·동결·검산 기록

첫 DEV 응답 전 동결은 `2026-10-22T02:04:56.637629+00:00`의 로컬 자기기록입니다. 외부 시각 인증이 아닙니다. 동결 당시 DEV 출력0, 이미 관측한 TECH 요청6(그중 현 시도4)으로 기록했습니다.

- Git base/HEAD: `83e1cef6dd60384931a38e016ea5e8d069d40674`.
- 최종 미커밋 own source bundle(39파일): `e19f15776ee4e74a37849e9fa6293e37a3d520bfd0221eb8b3545556f0e9af1d`.
- own source + 읽기 전용 dependency8 + runtime config의 실행 identity: **`e60dc421c2dc28c59e29ab26cb9faa6a2893208d5399577eed08d9745211ee7b`**.
- 원 P0 bundle: `0ddcc3df452bb29998b5df2704eb3c48fbd3ca2eac5405c78782490247f77c86`. 원 protocol/grammar/DEV manifest hash는 유지했습니다.

DEV 이전 관측 구현 시도 r1–r4는 실패로 남겼습니다. module hook이 실제 attention 경로에 도달하지 못한 r1/r2, legacy mask 함수를 관측한 r3, drain gauge의 접두사 파싱 오류 r4를 amendment01–04와 source별 exact-byte 사본으로 보존했습니다. 그 후 실제 구현/V2 mask/정확한 gauge 명칭을 연결했으며 과제·정답·scorer·가드·예산은 바꾸지 않았습니다. 이들 시도에는 DEV 출력이 없고 성공한 cell만 재시도하지 않았습니다.

최종 CPU independent audit는 **24/24, endpoint 차이0, raw trace 차이0**, 모든180요청의 SSE 재조립·IDs/usage·입력/관측·예산 검사가 PASS입니다. 별도 ReferenceState가 도구 전이·보호·최종/후속 상태를 다시 계산했습니다. 공유하는 strict parser·serialization·공개 입력/wire builder의 한계는 [REUSE_MAP](REUSE_MAP.md)에 적었습니다. 이는 로컬 CPU 교차검산이며 외부 GPU 재현이 아닙니다.

실행 후 추가 token checker의 첫 버전은 정렬된 freeze JSON에서 grammar를 재컴파일해 전180요청을 실패로 표시했습니다. 원 schema의 속성 순서를 잃은 검사기 결함입니다. 그 source/실패 receipt를 남긴 뒤, 동결된 **원 schema bytes와 실행 전 compiled grammar 일치**를 확인한 별도 r2 checker로180요청 PASS를 얻었습니다. source·raw·원 점수는 그대로입니다.

CPU test 고유 수는 **48=기존37+신규11**입니다. 44개 일괄 PASS 뒤 필요한 경계4개를 각각 추가 실행했습니다. 단일48개 일괄 검사라고 소급하지 않습니다. 실제 tokenizer grammar23개는 이 unittest 수와 별도입니다. 원 동결 source로 P0의26 CPU trace도 한 번 외부 output에서 다시 audit해 PASS였습니다. mock 응답을 모델 성공에 더하지 않았습니다.

## 보존·재현·남은 범위

원 P0 229파일/23,438,503bytes는 외부 snapshot에서 exact bytes로 보존했습니다. 현재 case의 기존225파일은 동일하며, 허용한 runner/audit/CLI/REUSE_MAP 네 파일만 확장했습니다. P0 보고서·protocol·과거 receipt/results/freeze와 DEV/world/scoring/continuation 의미는 유지했습니다. Case017 선택81파일과 실제 읽기 전용 helper8파일, 선택 runtime bytes 및 모델 작은 파일/stat 범위도 변경0입니다. 모든 weight/ZIP/다른 프로젝트 전체를 검사했다는 뜻은 아닙니다.

원 Git HEAD/branch/status는 시작과 같고 staged0, Case018 밖 tracked diff0입니다. HOLDOUT input/gold0개, P2/P3/서버 추가실험/원격 쓰기는 NOT_RUN입니다. 기술 gate와 DEV 실행/audit를 마쳤지만 P2 생성·봉인·실행 권한으로 확장하지 않습니다.

모델 없는 실제 명령과 runtime 입력은 [P1 재현 안내](REPRODUCE.P1.ko.md)에 있습니다. 원자료와 [기계 집계](results/p1_r5/analysis/summary.json), [24행 장부](results/p1_r5/analysis/episode_ledger.csv), [짝 비교](results/p1_r5/analysis/paired.csv), [실행 freeze](results/p1_r5/execution_freeze.json)를 연결했습니다. 외부 evidence의 `p1_final_audit.json`, `p1_retokenization_r2.json`, `cost_verification.json`, `final_verification.json`이 별도 검산 영수증입니다.

가드가 차단한 완료 주장과 실제 상태 복구는 다른 결과입니다. 이번 DEV6에서는 R1의 Q/W 회복을 관측하지 못했고, NVFP4 C01의 잘못된 보상과 여러 task의 final 수량 오류가 남았습니다. 원문 ReTrace의 GUI/on-policy 경로·논문 전체를 재현한 것이 아니며, 정해진 두 후속 절차 성공이 모든 미래 상태 동등성을 증명하지 않습니다. upstream 모델·엔진·커널과 직접 작성한 가드/평가기·계측·재생 도구를 구분합니다. Codex가 구현과 문서 작업을 지원했습니다.
