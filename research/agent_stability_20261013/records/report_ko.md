# Case 013 후속 — 같은 history에서 도구 선택과 정밀도 복원

기존 장문·도구 호출 기록을 모델 없이 비교하는 `diova-compare 0.1.2` exporter를 구현했습니다. 새 GPU 진단에서는 D16의 잘못된 도구 이름 선택과 오류 후 반복이 재현됐습니다. 출력 오차로 고른 projection 하나를 BF16으로 복원해도 두 실패는 회복되지 않았고, 같은 비용의 대조 복원은 새 손실 두 개를 만들었습니다.

기존 Case 013/r2와 구분한 DIAGNOSTIC_DEV입니다. 이미 관측한 DEV를 선택한 원인 진단이며 새 held-out 성능 시험이 아닙니다.

[구현](#구현) · [실험-계약](#실험-계약) · [동일-history의-도구-선택](#동일-history의-도구-선택) · [projection-복원](#projection-복원) · [해석](#해석) · [검증과-비용](#검증과-비용) · [직접-확인](#직접-확인) · [출처](#출처)

## 구현

- stdlib-only `export-case013`: 공개 source hash 검사 → 실제 parser/fixture 실행 조건 재채점 → method/cohort/mode/길이별 schema1 JSONL과 sidecar → 기존 validate/compare/report.
- 기존 r2 model loader·ReferenceLinear·도구 registry를 재사용한 별도 진단 runner. artifact 252개를 해시·shape 확인 후 읽기만 하고 solver를 호출하지 않습니다.
- 같은 input/prefix에서 next-token logprob·이름 continuation·오류 피드백 응답을 분리하고, 전체 nn.Linear 경로를 복원하는 조건부 개입을 구현했습니다.

P0는 historical 204행, 동질적 파일25개를 export하고 CLI validate25회·compare18회를 실행했습니다. 장문8K RTN→D16은 둘 다 정답9/기준선만2/후보만1/둘 다 오답0입니다.16K는10/0/0/2입니다. 기본 FREE는 RTN8/8→D16 6/8,16K 변경본은4/4→3/4입니다. 후보32K 미실행은0점으로 만들지 않습니다. PPL·저장 KL은 정답률과 별도 요약입니다.

## 실험 계약

- Qwen/Qwen3-4B-Instruct-2507 @ `cdbee75f17c01a7cc42f958dc650907174af0554`; 기존 RTX5080 SM120 / driver610.88 / Torch2.13.0+cu130 / Transformers5.17.0.
- P1: 기존 실패 AGENT-6/7와 성공 control AGENT-2/4. 최초 입력4,085–8,191tokens; seed610132, greedy, max192outputtokens. BF16/RTN/D16/D4를 순차 실행.
- 원 D16 artifact와 기존 D4 activation 변형 유지. FP32 beta transport → BF16 matmul, SDPA, TF32와 BF16 reduced-precision reduction OFF. request 내부 KV ON, turn 사이 prefix cache OFF.
- 최초 forward 전에 [preregistration](preregistration.json)과 [input](input_manifest.json)/[runtime](runtime_manifest.json)을 동결. 자연 응답 후 공통 prefix는 미리 정한 첫 token 분기 규칙으로 한 번 생성해 점수 forward 전에 별도 동결.
- P2: 원8개 FREE DEV, 조건별1회. 최대8modelcalls/6toolcalls,deadline900s,contextlimit24576; fixture는 조건마다 초기화. formal120task, 새로운 fitting·모델·packed kernel은 사용하지 않았습니다.

## 동일 history의 도구 선택

16개 최초 자연 응답의 output token IDs가 원 r2 기록과 모두 일치했습니다. BF16/RTN은 AGENT-6/7에서 올바른 snake_case patch 이름·인자를 호출했고, D16은 `ApplyFixturePatch`/`ValidateFixtureTest`를 호출했습니다. 성공 control의 D16 첫 행동은 `read_record`/`query_table`로 유지됐습니다.

아래 margin은 첫 다른 token의 log p(정확한 이름 token)−log p(관측된 잘못된 이름 token), 단위nats입니다. 양수는 정확한 이름을 선호합니다. 이름을 포함한 전체 문자열을 함께 tokenize했으며 독립 tokenize한 이름을 임의로 붙이지 않았습니다.

| 조건 | AGENT-6 patch | AGENT-7 patch | AGENT-6 validate | AGENT-7 validate |
|---|---:|---:|---:|---:|
| BF16 | +24.375 | +24.250 | +24.750 | +25.000 |
| RTN | +22.625 | +22.625 | +22.875 | +23.500 |
| D16 | -17.125 | -14.500 | +22.250 | +22.375 |
| D4 | -17.125 | -12.375 | +20.375 | +13.000 |

D16의 patch 이름 선호는 반전됐지만, 공통된 성공 patch 결과를 준 history의 validate 이름은 정확한 쪽을 선호했습니다. 최초 자연 분기도 두 patch task 모두6개의 공통 response token 뒤 도구 이름 첫 token에서 발생했습니다. 다른 task/조건의0-token 형식 분기와 동일하게 취급하지 않습니다.

### 동일 오류 피드백

| 조건 | 등록된 이름만 재선택 | 원 잘못된 이름 반복 | 분모 |
|---|---:|---:|---:|
| BF16 | 2 | 0 | 2 |
| RTN | 2 | 0 | 2 |
| D16 | 0 | 2 | 2 |
| D4 | 0 | 2 | 2 |

모든 조건에 원 D16의 같은 잘못된 tool_calls와 error result를 제공한 recovery probe입니다. BF16/RTN은 등록된 이름으로 바뀌었고 D16/D4는 잘못된 이름을 반복했습니다. 각 method가 자유롭게 같은 history를 방문했다는 뜻이나, 다음 응답만으로 전체 episode가 성공했다는 뜻은 아닙니다.

### 이름 전체 점수의 경계

주 등록은 이름 continuation과 닫는 따옴표를 포함한 전체 conditional logprob입니다. 마지막 따옴표가 독립 token이어서 뒤따르는 쉼표와 결합된 자연 JSON token보다 낮은 확률을 받는 경우가 있습니다. 표의 첫-token margin과 전체 name+quote 점수를 분리합니다. 모든 token logprob·합·평균·길이는 [controlled records](p1_controlled_name_scores.jsonl)에 있으며, 따옴표를 포함하는 등록을 결과 후 바꾸지 않았습니다. 두 이름만 재정규화한 확률이나 full KL은 만들지 않았습니다.

## Projection 복원

P1의 입력·artifact·head parity와 첫 행동 재현 gate를 통과했습니다. 새 I0에서 원 두 episode 실패도 재현돼 I1/I2를 진행했습니다. 선택에는 성공률·정답 margin을 쓰지 않았습니다.

- 타깃: `model.layers.2.mlp.down_proj`. AGENT-6/7 마지막 입력 위치의 평균 출력 SSE 차이/BF16 energy = 6.235706. 상대RMSE가 아니라 squared-energy 비입니다. upstream 입력 차이가 포함됩니다.
- 대조: `model.layers.35.mlp.down_proj`. 같은 projection type·shape `[2560, 9728]`·원 D16 payload bytes, 다른 block. 고정 seed610132 선택. zero-energy 제외0개.
- 해당 adapter 전체를 원 checkpoint의 BF16 nn.Linear로 교체해 beta transport를 제거했습니다. 나머지251projection은 동일 artifact를 사용합니다. 원형 weight와 bias 경로를 유지했고 수정 module 집합을 실행 전후 assert했습니다.

| 조건 | 원8개 FREE 성공 | 기준선에서 새 정답 | 기준선에서 정답 손실 |
|---|---:|---:|---:|
| I0 | 6/8 | — | — |
| I1 | 6/8 | 0 | 0 |
| I2 | 4/8 | 0 | 2 |

I0→I1: 둘 다 성공6/기준선만0/후보만0/둘 다 실패2. I0→I2:4/2/0/2. 두 복원 모두 AGENT-6/7을 회복하지 못했습니다. I2의 새 손실 AGENT-4/5는 update tool을 읽지 않고 `updated`/`latest_status`를 반환했습니다. 기존 r2 판정은 바꾸지 않았습니다.

원 D16 projection payload 15,603,736B를 원 BF16 weight 49,807,360B로 대체한다고 가정하면 두 조건의 장부 증가량은 각각 34,203,624B (32.619MiB)입니다. checkpoint index에는 해당 projection bias tensor가 없습니다. 새 packed artifact의 실측 크기나 VRAM 감소가 아닙니다. 실행은 이미 복원된 BF16 weights를 쓰는 참조 경로입니다.

## 해석

**관측:** 같은 입력에서 D16의 patch 이름 token 선호가 바뀌었고, 동일 error history에서도 오류가 반복됐습니다. 하나의 이름/문맥에 대한 선호 변화는 모든 도구에 공통인 손상을 뜻하지 않습니다. replay의 성공과 실제 도구 행동은 다른 질문입니다.

**개입 결과:** 가장 큰 마지막-position 출력 오차를 보인 projection 하나의 복원은 이 DEV의 행동을 회복하지 못했습니다. 같은 비용의 다른 block 복원은 오히려 새 손실을 만들었습니다. 이번 선택 기준의 회복 효과를 지지하는 근거를 얻지 못했습니다.

**남은 가설:** 여러 projection의 변화와 beta rescale/BF16 rounding, history-conditioned 식별자 선택의 상호작용이 관여할 수 있습니다. 어느 하나를 원인으로 확정하지 않았습니다. 마지막 input-position 출력 오차는 도구 이름 생성 위치의 local quantizer 오차나 인과적 중요도와 같지 않습니다.

도구 이름 alias·정답 교정·새 threshold를 추가하지 않았습니다. 이8개는 이미 관측·선택에 쓰인 DEV이고, 일반화·비열등성·새 알고리즘·serving 이득을 판정하는 표본이 아닙니다. 다음 과제는 별도 등록한 식별자 생성 위치의 국소/누적 오차 분리이며, 복원할 block을 바꿔가며 이번 점수를 재선택하지 않습니다.

## 검증과 비용

- 현재 package 고유46개(기존37+새9), 새 진단 CPU4개:50개 통과. wheel 설치 후46개 반복은 새 고유 테스트가 아닙니다. 원 GPU 연구39개는 과거 receipt입니다.
- 별도 동결 Case009 wrapper11단계 통과: 기존 연구24·공개 검사10·옛 package37/source 및 installed 반복. 옛 manifest/검사를 수정하지 않고 원 ZIP을 scratch에 복원합니다.
- 독립 stdlib auditor가 input bytes·성공/실행 조건·pairing·margin/sequence 합·선택 순위·module 범위·source exits를 검산했습니다. 제외된 full logits/activation/weights에서 logprob/SSE를 다시 계산했다는 뜻은 아닙니다.
- 실제 GPU worker9회, generation 요청 82회(P1 자연16+recovery8, P2 58). 총 worker wall 475.752s /7,200s. load·artifact read·CPU serialization·idle 포함, kernel busy time이 아닙니다.
- 관측 Torch peak allocated 10,667,811,328B (9.935GiB), reserved 13,006,536,704B (12.113GiB). Windows/장치 전체 peak는 미측정. 시작 시 다른 GPU 사용 약2.6GiB가 있었으며 종료하지 않았습니다.
- head parity9회는 최대차이0·argmax mismatch0. 원 tolerance0.25/argmax0을 유지했습니다. 새 solver·학습·모델 다운로드·packed kernel·원격 게시0.

## 직접 확인

[재실행 안내](README_REPRODUCE.md) · [독립 검산 JSON](independent_scalar_audit.json) · [P0 export 검산](adapter_validation.json) · [P1 gate](p1_gate.json) · [P2 선택](p2_selection.json) · [GPU 명령·exit·wall](gpu_commands.json) · [입력별 개입 기록](workers/p2/D16/attempt01/p2_interventions.jsonl) · [I0→I1 report](p2_compare_reports/I0-to-I1/report.md) · [I0→I2 report](p2_compare_reports/I0-to-I2/report.md)

## 출처

저자의 DualQuant와 DIOVA의 독립 참조/저장/평가 구현을 구분합니다. 원 Case013 METHOD_CONTRACT/EVALUATION_PROTOCOL/FAILURE_ANALYSIS/ROADMAP과 r2 source를 읽고 사용했습니다. 구현·실행·정리는 Codex 지원입니다.

- [DualQuant v1](https://arxiv.org/html/2610.13767v1), [고정 public code](https://github.com/MahsaMoonlike/dualquant/tree/d84b7e9dff653320e47c06a365eb8295c1825bc3). 이번 후속은 원 FP32-scale variant를 유지합니다.
- [Measuring the Microtask Eligibility Gap](https://arxiv.org/abs/2610.00025): agent microtask, 양자화와 logprob 진단을 다룬 선행연구. 이번 세션에서는 해당 초록을 확인했습니다.
- [Evaluating Local Language Model Agents for Reproducible Data Engineering](https://arxiv.org/abs/2610.11482): 완성된 artifact와 deterministic checker를 다룬 선행연구. 이번 세션에서는 해당 초록을 확인했습니다.
- 조회2026-10-13. 선행연구의 결과를 독립 재현하거나 이번 도구 호출/개입 원리의 first claim을 하지 않습니다.
