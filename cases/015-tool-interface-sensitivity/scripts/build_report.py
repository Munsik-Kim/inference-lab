"""Generate bilingual report tables from raw-derived summaries and identities."""
import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from interface_sensitivity.common import read,rows,atomic,sha
case=Path(__file__).resolve().parents[1];result=case/'results/p1_r1';s=read(result/'summary.json');resource=read(result/'resource_summary.json');records=rows(result/'episodes.jsonl');source=read(case/'provenance/source_freeze.json');recovery=read(result/'feedback_recovery.posthoc.json');art=read(case/'provenance/artifact_identity.json');historical=read(result/'historical_I0_comparison.json')
lookup={(r['task_id'],r['method'],r['interface_id']):r for r in records}

def matrix():
 text='| Method | I0 original | I1 order | I2 lowerCamelCase | I3 opaque |\n|---|---:|---:|---:|---:|\n'
 for m in ['BF16','RTN','D16']:
  rr=[r for r in s['success_table'] if r['method']==m];text+='| '+m+' | '+' | '.join(str(r['success'])+'/'+str(r['observed_tasks']) for r in rr)+' |\n'
 return text

def pairs():
 text='| Method | Interface vs I0 | Both success | I0 only: loss | Changed only: gain | Both fail |\n|---|---|---:|---:|---:|---:|\n'
 for r in s['paired_interface_counts']:
  text+='| '+r['method']+' | '+r['changed'].split('_')[0]+' | '+' | '.join(str(r[k]) for k in ['both_success','baseline_only','changed_only','both_fail'])+' |\n'
 return text

def sensitive():
 text='| Method | Changed pairs /24 | S | All four interfaces successful tasks /8 |\n|---|---:|---:|---:|\n'
 for r in s['sensitivity']:text+=f"| {r['method']} | {r['changed_pairs']}/24 | {r['S']*100:.2f}% | {r['all_interface_success']}/8 |\n"
 return text

def contrasts():
 text='| Contrast | Signed interaction (pp) | Exploratory paired 95% CI (pp) | Role |\n|---|---:|---|---|\n'
 for k,r in s['contrasts'].items():text+=f"| {k} | {r['estimate']*100:+.2f} | [{r['CI95'][0]*100:+.2f}, {r['CI95'][1]*100:+.2f}] | {'primary' if k=='D16-minus-BF16' else 'secondary'} |\n"
 return text

def families():
 text='| Family | Task IDs | Primary interaction (pp) | Leave this family out (pp) |\n|---|---|---:|---:|\n'
 for k,r in s['families'].items():text+=f"| {k} | {', '.join(r['tasks'])} | {r['primary_D16_minus_BF16']*100:+.2f} | {r['leave_this_family_out']*100:+.2f} |\n"
 return text

def payload():
 return '| Method | Projection files | Stored bytes |\n|---|---:|---:|\n'+''.join(f"| {m} | 252 | {art[m]['payload_bytes']:,} |\n" for m in ['RTN','D16'])

def names():
 text='| Method | Interface | Valid name / evaluable call objects |\n|---|---|---:|\n'
 for r in s['call_metrics']:text+=f"| {r['method']} | {r['interface_id'].split('_')[0]} | {r['name_valid']}/{r['name_evaluable']} |\n"
 return text

with (result/'task_outcomes.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['episode_id','task_id','family_id','method','interface_id','functional_success','legacy_success','first_error_type','model_calls','tool_attempts','generated_tokens']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in records)
with (result/'success_table.csv').open('w') as f:
 fields=['method','interface_id','success','observed_tasks','planned_tasks','legacy_success'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in s['success_table'])
with (result/'paired_interface_counts.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(s['paired_interface_counts'][0]));w.writeheader();w.writerows(s['paired_interface_counts'])

target=lookup['AGENT-6','D16','I3_OPAQUE_RENAME'];success_names=[e['raw_name'] for e in target['tool_events']]
source_hash=source['execution_source_sha256'];wall=resource['worker_wall_seconds'];allocated=resource['peak_torch_allocated_bytes'];reserved=resource['peak_torch_reserved_bytes']
intro_ko='같은 작업에 제공하는 도구의 이름과 순서를 바꾸고, 모델이 실제로 작업을 끝내는지 비교하는 도구를 구현했습니다. 정확한 이름 변환, 단계별 동등성 검사, 실제 fixture 실행과 저장 응답 재감사를 연결했습니다.'
intro_en='Implemented a local comparison tool that changes the names and ordering of the same tools, then checks whether the agent actually completes the task. It connects exact interface mappings, stepwise equivalence, real fixture execution and saved-response replay.'
for ko in [True,False]:
 lang='ko' if ko else 'en';report='REPORT.ko.md' if ko else 'REPORT.md';repro='REPRODUCE.ko.md' if ko else 'REPRODUCE.md';title='Case015 — 같은 도구, 다른 이름' if ko else 'Case015 — Same Tools, Different Interfaces'
 first=(intro_ko if ko else intro_en)
 observation=('96개 FREE episode를 완료했습니다. D16은 불투명 이름에서 두 patch 실패를 회복했지만, lowerCamelCase에서 새 손실이 생겼습니다. 주 signed interaction의 구간은 0을 포함합니다.' if ko else 'Completed 96 FREE episodes. D16 recovered two patch failures under opaque names, while lowerCamelCase introduced a new loss. The primary signed-interaction interval includes zero.')
 readme=f'''# {title}

{first}

{observation}

[{'English' if ko else '한국어'}]({'README.md' if ko else 'README.ko.md'}) · [ {'결과·사건 탐색' if ko else 'Results and episodes'} ](demo/{lang}.html) · [ {'정식 보고서' if ko else 'Report'} ]({report}) · [ {'CPU 검산' if ko else 'CPU reproduction'} ]({repro}) · [Code](src/interface_sensitivity/cli.py)

{matrix()}

{'각 칸은 같은 8개 기존 DEV 과제의 실제 기능 성공 수입니다. 96개는 독립 과제 수가 아닙니다. ENUM·개선된 오류 안내·alias는 모두 껐습니다.' if ko else 'Each cell is functional success on the same eight existing DEV tasks;96 is not an independent task count. ENUM, enhanced feedback and aliases are off.'}

- {'이름·순서 변환과 의미 ID의 정확한 대응' if ko else 'Exact name/order transformations and semantic-ID mappings'}: [registry](src/interface_sensitivity/registry.py), [transform](src/interface_sensitivity/transform.py).
- {'도구·인자·실제 상태·검증의 단계별 확인' if ko else 'Per-step actions, arguments, actual state and validation'}: [runner](src/interface_sensitivity/runner.py), [scoring](src/interface_sensitivity/scoring.py).
- {'모델 없이 원시 응답·도구 사건·paired 효과 재계산' if ko else 'Model-free response/action replay and paired-effect analysis'}: [audit](src/interface_sensitivity/audit.py), [analysis](src/interface_sensitivity/analysis.py).

{'P0 CPU 고유 검사 27개와 32개 oracle episode가 통과했습니다. GPU 요청은 본실험 228회와 별도 smoke 4회이며, worker wall은 모델 로딩·decode를 포함해' if ko else 'P0 passed 27 unique CPU tests and 32 oracle episodes. GPU generation requests: 228 primary plus 4 separate smoke. Worker wall including model load/decode:'} {wall:.3f} s. {'실행은 기존 INT4 저장본을 BF16으로 복원한 품질 reference이며 serving 속도 비교가 아닙니다.' if ko else 'Execution uses saved INT4 with dense BF16 reload; these are not serving-speed measurements.'}

[{'계약·계획' if ko else 'Contract'}](specs/protocol.json) · [{'원자료' if ko else 'Raw records'}](results/p1_r1/episodes.jsonl) · [{'독립 CPU 재계산' if ko else 'CPU replay audit'}](results/p1_r1/independent_audit.json) · [{'판정·범위' if ko else 'Decision'}](results/p1_r1/decision.json) · [{'후속 제안' if ko else 'Roadmap'}](ROADMAP.md)
'''
 (case/('README.ko.md' if ko else 'README.md')).write_text(readme)

ko_report=f'''# Case015 — 같은 도구, 다른 이름

**Tool-Interface Sensitivity of Quantized LLM Agents**

{intro_ko}

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

{payload()}

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
| 누적 worker wall | {wall:.3f} s |
| Torch peak allocated / reserved | {allocated:,} / {reserved:,} B |
| 새 solver·양자화·설치·모델 다운로드 / 재시도 | 0 / 0 |

worker wall은 load·decode·smoke·CPU·I/O·실패 시간을 포함하는 process wall이며 serving 지연이 아닙니다. 장치 free/total은 요청별 CUDA 표본으로 저장했고 Windows 전체 장치의 연속 peak는 측정하지 않았습니다. [자원 원자료](results/p1_r1/resource_summary.json).

<a id="results"></a>
## 결과

{matrix()}

분모는 각 칸의 기존 8개 과제입니다. [과제별96행](results/p1_r1/task_outcomes.csv), [전체 사건 탐색](demo/ko.html#episodes).

{contrasts()}

주 비교 구간은 0을 포함합니다. RTN–BF16의 보조 구간도 0에 닿으므로 우위·동등성 판정에 사용하지 않습니다. 결측이 없어 주 효과의 미관측 최악/최선 범위는 동일한 +0.083333입니다.

{sensitive()}

D16은24개 원형→변경 쌍 중3개가 달라졌습니다. 방향을 나누면 I2에서 AGENT-4손실1개, I3에서 AGENT-6/7회복2개입니다. BF16은 I3 AGENT-3손실1개이고 RTN의 성공 여부는24쌍 모두 같았습니다. RTN도 호출 이력까지 불변인 것은 아닙니다.

{pairs()}

{families()}

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
| D16/I2 AGENT-4 | 도구 없이 `{{"status":"latest_status"}}` | 이름 파싱 문제가 아닌 필요한 읽기·정답 실패. [raw](results/p1_r1/workers/D16/attempt01/cells/D16__I2_STYLE_RENAME__AGENT-4.json) |
| BF16/I3 AGENT-3 | record는11×3인데 등록된 계산 도구에10×5.5 전달, total55 | name은 유효하지만 인자 값·정답 실패. [raw](results/p1_r1/workers/BF16/attempt01/cells/BF16__I3_OPAQUE_RENAME__AGENT-3.json) |
| D16/I3 AGENT-6 | {', '.join('`'+n+'`' for n in success_names)} 호출, 실제 설정 수정·성공 validation·최종 valid:true | 회복된 성공 사례. [raw](results/p1_r1/workers/D16/attempt01/cells/D16__I3_OPAQUE_RENAME__AGENT-6.json) |

{names()}

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
- GPU execution source SHA256: `{source_hash}`. Git SHA와 다릅니다.
- [문맥 내 BPE·TASK 위치](provenance/tokenization.posthoc.json): 기존32개 입력을 재인코딩해 hash를 확인한 사후 입력 metadata이며 추가 모델 추론이 아닙니다.
- [source freeze](provenance/source_freeze.json), [model files](provenance/model_identity.json), [artifact identity](provenance/artifact_identity.json), [입력](provenance/input_manifest.json).
- [CIPR paper v1](https://arxiv.org/html/2610.20576v1), [고정 저자 README](https://github.com/ArtinTD/cross-interface-policy-regularization/tree/0465909ce219cdd1202affff75e41330697e4862), [실제 출처 확인 범위](provenance/source_audit.json).
- CIPR은 인터페이스 일관성을 학습하는 선행 연구입니다. 여기서는 학습·코드 차용 없이 이름·순서 변경의 실행 민감도를 평가했습니다. 이론·구현의 공식 재현을 주장하지 않습니다.
- Codex/ChatGPT가 구현·실험·별도 CPU 계산을 보조했습니다. 사람의 독립 심사나 외부 GPU 재현으로 표시하지 않습니다. [NOTICE](NOTICE.md).

</details>
'''
(case/'REPORT.ko.md').write_text(ko_report)

en_report=f'''# Case015 — Same Tools, Different Interfaces

**Tool-Interface Sensitivity of Quantized LLM Agents**

{intro_en}

Completed 96 FREE episodes with Qwen3-4B BF16/RTN/D16 and four interfaces. Opaque names recovered D16's two original patch failures; lowerCamelCase caused one new update-task loss. The registered signed interaction is **+8.33 percentage points**, exploratory paired 95% CI **[-8.33,+25.00] pp**, including zero.

## Contents

[Background](#background) · [Questions](#questions) · [Interface contract](#contract) · [Method](#method) · [Experiment](#experiment) · [Results](#results) · [Failures and recovery](#failures) · [Conclusion](#conclusion) · [Reproduction and sources](#references)

<a id="background"></a>
## Background

Case013 observed D16 UpperCamelCase name failures. Later ENUM and enhanced feedback improved the same existing DEV tasks. Both mitigations are OFF here. This study evaluates interface sensitivity with an executable evidence pipeline, without new quantization or CIPR training.

RTN is simple round-to-nearest INT4; D16 is the independent DualQuant weight-INT4 reference. Functional success checks the requested actual actions and state together with the final JSON.

<a id="questions"></a>
## Questions and hypotheses

The registered primary asks whether D16's signed interface change is worse than BF16's:

`d(m,t)=mean_i=I1,I2,I3 Y(m,i,t)-Y(m,I0,t)`

`Delta=mean_t[d(D16,t)-d(BF16,t)]`

Negative means worse signed change for D16. Gains can cancel losses, so loss/gain pairs, absolute outcome-change rate S and all-interface success are also reported. Secondary questions locate observed differences in names, semantics, arguments, recovery and end state, and motivate later diagnostic tests. I2 was designed knowing past name failures; I3 uses one fixed assignment per task.

<a id="contract"></a>
## Interface and execution contract

| Interface | Change | Preserved |
|---|---|---|
| I0 ORIGINAL | None | Original names, order, schema, descriptions and messages |
| I1 ORDER_ONLY | Seeded task-specific nonidentity permutation | All content and names |
| I2 STYLE_RENAME | snake_case to lowerCamelCase | Arguments, descriptions, order and types |
| I3 OPAQUE_RENAME | Answer-independent tool_NN assignment | Arguments, descriptions, order and types |

The semantic identity is internal and never exposed to the model. Only typed declarations and verified TASK API spans change; arbitrary text, fixture values and generated responses do not. [Mappings](specs/interfaces.json) record field paths, reversals and exact before/after values. Original descriptions distinguish document facts, quantity/price, prior/current status, multiplication, configuration replacement and validation. Opaque allocation does not read gold or declaration position.

An exact exposed-name lookup precedes dispatch to the unchanged original callable. Old names and typos produce UNKNOWN_TOOL without side effects. Model-visible error strings remain the generic original strings; sidecar labels and allowed-action sets are not feedback hints. No enum or aliases apply.

Primary FUNCTIONAL_SUCCESS requires original-task reads/calculation in order and actual patch/validation. LEGACY_SUCCESS preserves Case013 final-answer/state scoring separately. Irrelevant extra reads can receive diagnostic WRONG_TOOL labels without automatically failing the episode; operand commutation is accepted. All 96 current cells happen to have identical functional and legacy scores; the contracts remain distinct and are tested with counterexamples.

<a id="method"></a>
## Method

Read-only existing WSL Qwen/Qwen3-4B-Instruct-2507 snapshot, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Twelve local model files including all three weight shards, and 504 saved RTN/D16 projection files, were SHA256-checked before and after execution. This establishes local identity, without a new official remote digest comparison.

{payload()}

These are projection payloads, not standalone checkpoint sizes or equal-byte methods. Embeddings, head, norms, attention, RoPE and tying remain original. D16 retains FP32 input beta-inverse transport followed by BF16, and BF16 W_pre. No folded weight_hat substitution or double beta application. Missing artifacts fail; solve/encode/serialize/full-sweep paths are guarded.

Greedy, enable_thinking=False, maximum 192 new tokens, 8 model calls / 6 attempted tool calls, 900 seconds per episode, context limit 24,576, batch 1, SDPA, BF16, TF32=False, BF16/FP16 reduced precision reduction=False, within-request KV ON and across-turn cache OFF, logits_to_keep=1. Only six registered local fixture functions execute; generated shell/Python is data.

<a id="experiment"></a>
## Experiment

Eight previously observed tasks: two each document/combine/updates/fixture_patch, preserving original 4K/8K allocation. 96 primary episodes are 3 methods × 4 interfaces × 8 tasks, not 96 independent tasks. Two separate BF16/I0 AGENT-6/7 smoke episodes are excluded from primary results. Each method runs eight I0 first, then 24 seeded, shuffled cells. Model load order: BF16, D16, RTN.

Interface/execution/bootstrap seeds 6101601/6101602/6101603. 27 unique CPU tests and 32 oracle episodes / 48 step comparisons passed before source/input freeze. All 24 I0 initial input hashes, first output token-ID sequences and legacy outcomes exactly matched historical records. [Historical check](results/p1_r1/historical_I0_comparison.json).

| Measurement | Actual value |
|---|---:|
| Primary episodes / missing / unobserved | 96 / 0 / 0 |
| Primary generation requests / generated tokens / tool attempts | 228 / 5,765 / 160 |
| Separate smoke episodes / requests / generated tokens | 2 / 4 / 100 |
| Cumulative worker wall | {wall:.3f} s |
| Peak Torch allocated / reserved | {allocated:,} / {reserved:,} B |
| New fitting/quantization/install/model download; retries | 0; 0 |

Worker wall includes load, decode, smoke, CPU, I/O and failures; it is not serving latency or GPU-busy time. Request-level CUDA free/total samples are stored; a continuous Windows-wide device peak was not measured. [Resources](results/p1_r1/resource_summary.json).

<a id="results"></a>
## Results

{matrix()}

Each cell uses the same eight DEV tasks. [All 96 task rows](results/p1_r1/task_outcomes.csv), [episode explorer](demo/en.html#episodes).

{contrasts()}

The primary interval includes zero. The secondary RTN–BF16 interval touches zero and does not establish superiority/equivalence. No observations are missing: worst/best missingness bounds coincide with +0.083333.

{sensitive()}

D16 changed 3 of 24 original-to-perturbed outcomes: one I2 loss (AGENT-4), two I3 gains (AGENT-6/7). BF16 lost AGENT-3 under I3. RTN preserved outcomes on all 24 pairs; its action histories were not invariant.

{pairs()}

{families()}

Leave-one-family-out primary range:0 to +16.67 pp. The 10,000-draw bootstrap resamples each whole 12-cell task vector, retaining repeated I0 dependence. These exploratory eight-task/four-family intervals are not external-population guarantees or confirmatory p-values. Restricting to the six tasks both methods initially solved produces one new loss per method with equal signed size; that selected subset is not evidence of universal invariance.

![Functional success](figures/p1_r1/success_matrix.svg)

Higher is better; each cell is 8 DEV tasks, with no invented CI.

![Losses and gains](figures/p1_r1/paired_losses_gains.svg)

Below zero: original success becomes failure. Above zero: original failure becomes success.

![Interaction](figures/p1_r1/paired_interaction.svg)

Units are percentage points, with whole-task exploratory intervals.

<a id="failures"></a>
## Failure and recovery analysis

| Case | Observed behavior | Verdict/source |
|---|---|---|
| D16/I0 AGENT-6/7 | Unregistered ApplyFixturePatch/ValidateFixtureTest; repeats after errors; valid:true without patch | Failure; [AGENT-6raw](results/p1_r1/workers/D16/attempt01/cells/D16__I0_ORIGINAL__AGENT-6.json) |
| D16/I2 AGENT-4 | No call; returns status:latest_status | Missing required read and wrong answer; [raw](results/p1_r1/workers/D16/attempt01/cells/D16__I2_STYLE_RENAME__AGENT-4.json) |
| BF16/I3 AGENT-3 | Reads 11 and 3, but calls registered multiplication with 10 and 5.5; final 55 | Name valid, argument values and answer wrong; [raw](results/p1_r1/workers/BF16/attempt01/cells/BF16__I3_OPAQUE_RENAME__AGENT-3.json) |
| D16/I3 AGENT-6 | Calls {', '.join(success_names)}, actually patches and validates, then valid:true | Recovered success; [raw](results/p1_r1/workers/D16/attempt01/cells/D16__I3_OPAQUE_RENAME__AGENT-6.json) |

{names()}

Name-validity denominators include evaluable parsed call objects. No-call episodes are NOT_APPLICABLE; parsing failures are separate. Similar-looking unknown names never receive execution credit. Error labels can overlap, so event counts are not summed into failure-episode counts.

The registered broad continuation field includes sidecar-only errors and later tool events. A separate **post-hoc denominator refinement** considers actual error/negative-validation results followed by a later model turn, excluding same-turn calls. Among 6 eligible episodes, 2 later used a registered name, 0 recovered a correct required action, and 0 finished successfully. Raw/registered scores remain unchanged. [Explicit post-hoc analysis](results/p1_r1/feedback_recovery.posthoc.json).

[72 common-event-prefix comparisons](results/p1_r1/common_event_prefixes.json) align only the same initial state and actual shared event prefix, never arbitrary turn numbers. Free rollouts do not prove internal semantic ability or token-margin causality.

<a id="conclusion"></a>
## Conclusion

Q1's registered signed comparison is inconclusive. D16's observed outcome-change rate exceeds BF16's, with gains and losses in different tasks. Q2 reveals distinct unregistered-name, bad-value and missing-read starts. Q3 supports considering same-error-feedback diagnostics with prespecified successes as controls, then separate interface-specific comparisons of existing ENUM/FEEDBACK. Those follow-ups were not executed.

| Scope | Status |
|---|---|
| Interface transformation/equivalence/fixture execution/CPUaudit | Executed |
| Existing 8 DEV × 12 conditions | All96completed |
| New tasks/repeated opaque allocations/other models/D4 | Not run |
| P2 diagnostics / P3 fresh64 / CIPR training | Not run |
| Packed serving/business services/remote publication | Not run |

<a id="references"></a>
## Reproduction and sources

[CPUcommands](REPRODUCE.md), [reuse](REUSE_MAP.md), [preservation](results/p1_r1/preservation.json), [audit](results/p1_r1/independent_audit.json), [verification](results/p1_r1/verification.json).27 new unique CPU tests; 46 existing package tests reexecuted. Repeated runs and historical Case013 39 tests are not added to new counts.

The first P0 attempt had one exact-binary-equality assertion failure for thirds, fixed before freeze. The first package test attempt had three failures because relative PYTHONPATH broke in subprocess temporary cwd; absolute PYTHONPATH passed 46 without package changes. Failure logs are retained. The GPU execution source stayed frozen.

The auditor reparses 228 saved responses, reexecutes 160 fixture actions, checks 96 endpoints and independently recalculates the primary scalar/bootstrap. It does not reproduce excluded logits/weight computations. Review content includes code, synthetic inputs, light evidence and bilingual explorer; model / INT4 payloads / environments / large tensors are excluded.

<details><summary>Identities and attribution</summary>

- Git base:`e647b474bdc9fa558d851bd9c998b664b0e33638`; no new commit.
- Execution source SHA256:`{source_hash}`; distinct from Git SHA.
- [Context-dependent name BPE and TASK positions](provenance/tokenization.posthoc.json): post-hoc input metadata for32registeredprompts, re-encoded without model forward.
- [Source freeze](provenance/source_freeze.json), [local model identities](provenance/model_identity.json), [artifacts](provenance/artifact_identity.json), [inputs](provenance/input_manifest.json).
- [CIPR v1](https://arxiv.org/html/2610.20576v1), [pinned author README](https://github.com/ArtinTD/cross-interface-policy-regularization/tree/0465909ce219cdd1202affff75e41330697e4862), [actual source audit](provenance/source_audit.json). This evaluation does not train CIPR or claim official implementation/theory reproduction.
- Codex/ChatGPT assisted implementation, execution and separate CPU recalculation, without independent human review or external GPU replication. [NOTICE](NOTICE.md).

</details>
'''
(case/'REPORT.md').write_text(en_report)
atomic(result/'report_generation.json',{'source_records_sha256':sha(result/'episodes.jsonl'),'summary_sha256':sha(result/'summary.json'),'generator_sha256':sha(Path(__file__)),'languages':['ko','en'],'metrics_from_records':True})
print('generated bilingual reports and3CSVtables')
