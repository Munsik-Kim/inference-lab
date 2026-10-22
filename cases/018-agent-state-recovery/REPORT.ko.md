# Case018 — 상태 복구와 최종 보고는 함께 맞을까?

## 질문과 구현

잘못 옮긴 재고, 두 번 처리한 이동, 일부만 차감한 기록을 복구하는 작은 도구 환경을 구현했습니다. 종료 상태뿐 아니라 모델이 실제 받은 검증 결과·최종 수량 보고와 두 고정 CPU 후속 절차를 함께 검사합니다. 후속 절차는 평가기가 실행하며 모델의 자율 후속 업무가 아닙니다.

창고·상품 수량과 처리 원장을 함께 복구합니다. A는 잘못 바꾼 대상을 되돌리고 원래 작업은 후속 절차에서 수행합니다. B는 정상1회는 보존하고 중복1회만 보상합니다. C는 부분 차감을 되돌립니다. 숨은 checkpoint·정답 script를 모델에 주지 않습니다. validator는 공개 불변식만 검사합니다.

## 방법과 사전 계약

K0=BF16 KV, K1=native NVFP4 KV; 양쪽 Qwen3-4B-Instruct-2507 BF16 weights, revision cdbee75f17c01a7cc42f958dc650907174af0554. R0는 형식에 맞는 final을 수용하고 사후 채점하며 R1은 현재 scope/digest/version의 성공 validation을 final 전에 실제 받았는지만 확인합니다. final 수량 정답을 검사하거나 자동 복구하지 않습니다. 공통 grammar G, strict parser, API8, greedy, 요청16/출력512·episode8192/문맥24576/180·900초 예산입니다. 모델 성공수 확대 임계값은 원 계획에서 null입니다.

D는 recovered:true 선언(거절된 final도 포함), B는 업무 값 복원, P는 전 과정 보호, Fresh는 현재 성공 검증의 사전 관측입니다. Q=B∧P∧C1∧C2, W=Q∧accepted final∧정확한 final∧Fresh입니다. accepted final이 없을 때도 terminal Q를 계산하고 W0이며 Fresh는 마지막 생성 전 진단입니다. 예: 수량은 복원됐으나 최초 수량을 final에 쓰면 Q1/W0입니다.

## HOLDOUT 결과와 짝 비교

신규 합성 HOLDOUT24에서 BF16 KV는 두 정책 모두 Q24/W0, native NVFP4 KV는 Q19/W5였습니다. 최신 검증 가드는 NVFP4의 부적격 final 15시도/5episode 수용을 막았지만 Q/W 회복은0이었고 출력 비용은18.69% 늘었습니다. 양쪽 가중치는 BF16 고정입니다.

| KV | Policy | Q /24 | W /24 | Requests | Input tokens | Output tokens | Guard rejections | Episode wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | R0 | 24 | 0 | 168 | 323375 | 10953 | 0 | 164.144812115 |
| BF16 | R1 | 24 | 0 | 168 | 323375 | 10953 | 0 | 164.157640174 |
| NVFP4 | R0 | 19 | 5 | 186 | 369881 | 11568 | 0 | 179.978763365 |
| NVFP4 | R1 | 19 | 5 | 206 | 439813 | 13730 | 15 | 213.922486411 |

Q는 업무 값·전 과정 보호·두 고정 후속 절차의 적합성이고, W는 여기에 정확한 accepted final과 최신 성공 검증 관측을 더합니다. 작은 합성 표본의 탐색 측정이며 지표별 순위가 달랐습니다. KV 품질 동등성·서빙 속도·모델 내부 기억의 원인을 증명하지 않습니다.

24task×4조건=96episode이며 독립 단위는24task입니다. A/B/C 각8개, 4조건을 함께 유지하는 family-stratified paired bootstrap10000/seed61022018의 pointwise percentile95% 구간입니다. R1−R0 Q/W는 두 KV에서0pp [0,0]이며 일반 동등성 증명이 아닙니다. K1−K0 Q는 −20.83pp [−33.33,−8.33], W는 +20.83pp [8.33,37.50]입니다. B/C1/C2 성공 집합이 P1/P2 실제 표본에서 같아 후속 절차의 추가 판별력은 관측되지 않았습니다. P0 합성 반례는 별도 평가기 검증입니다.

## 사후 수량 관측 진단

사후 진단에서 Q1/W0 76개 모두 final 수량이 최초 관측 수량과 같았습니다. accepted final91개 중 현재 수량을 재관측한13개는 종료 수량 map과 일치했고, 재관측하지 않은78개는 달랐습니다. 이13개 중 잘못 복구한 상태를 정확히 보고한3개가 포함돼 보고 일치가 복구 성공을 뜻하지 않습니다. 가드 종료5개는 final 비교 NA입니다. 관련성만 관측했으며 재조회를 강제한 새 실험은 하지 않았습니다.

| KV | Policy | Accepted / no final | Read0 wrong | Read0 right | Read1 wrong | Read1 right |
|---|---|---|---:|---:|---:|---:|
| BF16 | R0 | 24 / 0 | 24 | 0 | 0 | 0 |
| BF16 | R1 | 24 / 0 | 24 | 0 | 0 | 0 |
| NVFP4 | R0 | 24 / 0 | 16 | 0 | 0 | 8 |
| NVFP4 | R1 | 19 / 5 | 14 | 0 | 0 | 5 |

Read0/1은 필요한 모든 공개 record의 현재 수량을 실제 전달된 read_records에서 관측했는지입니다. 마지막 수량 변경 뒤 읽은 값만 유효하며 변경 후 원복도 이전 read를 무효화합니다. 메타데이터만 바뀐 global revision은 수량 관측을 무효화하지 않습니다. 모든 상태 변경 뒤 read라는 더 엄격한 보조 flag도 별도 저장했습니다(이번 실제 표본의 flag는 일치). final의 정답이나 checkpoint를 flag 판정에 쓰지 않습니다. accepted final 전 available ID를 anchor로 하며 final이 없으면 마지막 생성 전 진단, 수량 비교 NA입니다. guard 종료5개 중3개는 현재 수량 관측1,2개는0입니다. 관측0/정확 보고 및 관측1/오보고는 accepted 표본에서 각각0이며 인과 증명이 아닙니다.

P1 별도 보조 분석: BF16 각6 중 read1/정확 보고2·read0/오보고4; NVFP4 R0는 read0/오보고6, R1은5와 no final1입니다. P1 Q1/W0 18개도 최초 관측 수량과 일치했습니다. P1/P2 분모를 합치지 않았습니다. Q1 하위 표는 결과로 선택된 조건부 집계이며 [전체 ledger](publication/summary/quantity_ledger.json)에 별도 보존했습니다. 이미 본76개/A00가 분석 동기이며 새 전체 집계 전 [정의·등록](publication/analysis/ANALYSIS_FREEZE.json)을 저장했습니다. 결과 미관측 사전등록이 아닙니다.

## 원문 사건

A00 K0: 보상6/6·release 후 실제 x/y43/28, 현재 valid:true를 관측했지만 final37/34로 Q1/Fresh1/W0입니다. K1은 보상 뒤 read43/28→검증→final로 W1입니다. A06 K1은 잘못된40/30 보상으로 x/y80/0·총량+10, P0/Q0입니다. 이후 현재 값을 정확히 보고해도 복구 성공이 아닙니다. C01 K1은 부분 차감을 되돌릴 때 도착도4를 빼 s/d62/12로 오복구했습니다. R0는 이 현재 값을 정확히 보고했지만 Q0/Fresh0/W0, R1은 같은 잘못된 상태에서 세 final을 거절받았습니다. 원문·관측 ID·종료 상태는 [선택 사건](publication/summary/selected/)에 있고 전체 원기록은 archive에 보존했습니다. 요약은 raw 자체가 아닙니다.

## 비용·KV 저장·기술 한계

NVFP4 R0→R1:186→206 requests, 입력369881→439813(+18.91%), 출력11568→13730(+18.69%), episode wall179.978763365→213.922486411초(+18.86%). Q19/W5는 그대로입니다. 실패 포함 전체24개 비용이며 성공만의 평균 latency·과금·전력이 아닙니다. 성공0이면 비용/성공은 undefined입니다. 입력과 출력을 같은 연산 비용으로 합치지 않습니다.

P1 DEV180+TECH6 attempts(4완전/2취소), P2 HOLDOUT728+TECH3완전입니다. 누적917 attempts/915완전, 관측 출력 하한57987 tokens이며 취소2건의 엔진 생성 token은 null입니다. P2 HOLDOUT 입력1456444/출력47204, TECH입력2901/출력39입니다. worker439.692244164+855.069382600=1294.761626764초는 실패 기술 시도·load/compile/종료를 포함하고 episode wall과 겹칩니다. CPU 공개 분석 비용은 별도 receipt이며 worker에 더하지 않습니다.

실제 약4GiB pool: BF16 4293918720B·147456B/token·29120slots, NVFP4 4294508544B·41472B/token·103552slots. packed uint8+FP8 block scale와36층 write/read/유한 attention 출력의 과거 기술 receipt를 보존했습니다. 약3.556배는 token-slot 수용량이지 전체 VRAM 절감·single100K context·속도 우위가 아닙니다. vLLM0.32.0/FlashInfer0.7.0.post1/xgrammar0.2.8이며 KV 반올림만이 아니라 커널 경로도 다른 구성 비교입니다. 엔진 force-kill/semaphore 경고와 실패 시도도 숨기지 않았습니다. 새 profiling/GPU/모델 요청은0입니다.

## 재현·출처·결론

```bash
python3 -B cases/018-agent-state-recovery/publication/run_checks.py --output /tmp/case018-public-check-new --tests
```

공개 archive에서120 model endpoint와P0 26 synthetic trace를 phase별 frozen source로 독립 재생합니다. ReferenceState 상태 전이·보호·후속·SSE는 별도 구현이고 strict parser/serialization/public builder/wire rendering은 공유합니다. 저장 bool을 판정 입력으로 쓰지 않습니다. 원 tokenizer 재인코딩 receipt는 재사용하며 이번 CPU 경로에서 모델·tokenizer·SDK를 로드하지 않습니다. same-machine clean-copy 검산이며 외부 사람/GPU 독립 재현이 아닙니다.

현재 수량 재관측과 최종 보고 정확성의 연관을 저장 궤적에서 사후 분석했습니다. 재조회가 정확성을 일으켰거나 NVFP4가 기억을 개선했다는 결론은 내리지 않습니다. read 강제·새 정책·P3 모델 실험은 NOT_RUN입니다. 원문의 복구 뒤 행동 평가 원리를 작은 도구 환경에 응용했으며 GUI/on-policy 논문 전체 재현이 아닙니다. [SOURCE_MAP](SOURCE_MAP.md) · [REUSE_MAP](REUSE_MAP.md) · [NOTICE](NOTICE.md) · [재현](REPRODUCE.ko.md).
