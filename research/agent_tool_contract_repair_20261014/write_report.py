"""Render reports exclusively from the independently checked result summary."""
import argparse
import json
from pathlib import Path


def main(result):
    data = json.loads((result / 'summary.json').read_text())
    rows = data['conditions']; totals = data['totals']
    pick = lambda m, c: next(r for r in rows if r['method'] == m and r['condition'] == c)
    table = '| Model condition | Original | Feedback | Name enum | Both |\n|---|---:|---:|---:|---:|\n'
    for method in ('BF16', 'RTN', 'D16', 'D4'):
        values = [f"{pick(method,c)['successes']}/{pick(method,c)['tasks']}" if any(r['method']==method and r['condition']==c for r in rows) else 'NOT_RUN' for c in ('BASELINE','FEEDBACK','ENUM','BOTH')]
        table += '| ' + method + ' | ' + ' | '.join(values) + ' |\n'
    details = '| Method | Condition | Success / tasks | Unknown-name events | Tool errors | Parse errors | Model requests |\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in rows:
        details += f"| {r['method']} | {r['condition']} | {r['successes']}/{r['tasks']} | {r['unregistered_name_events']} | {r['tool_errors']} | {r['parse_errors']} | {r['model_requests']} |\n"
    pairs = '| Method | Candidate vs baseline | Both pass | Baseline only | Candidate only | Both fail |\n|---|---|---:|---:|---:|---:|\n'
    for r in data['paired']:
        pairs += f"| {r['method']} | {r['candidate']} | {r['both_success']} | {r['baseline_only']} | {r['candidate_only']} | {r['both_failed']} |\n"
    failures = '\n'.join(f"- {r['method']} / {r['condition']}: {', '.join(r['failed_task_ids']) or 'none'}" for r in rows)
    d16base, d16both, d4both = (pick('D16','BASELINE'),pick('D16','BOTH'),pick('D4','BOTH'))
    resource = f"{totals['GPU_workers']} GPU workers; {totals['episodes']} episodes / {totals['unique_previously_observed_tasks']} original task IDs / {totals['logical_conditions']} logical conditions; {totals['generation_requests']} generation requests. Conservative worker wall {totals['worker_wall_seconds']:.3f} s. Torch peak allocated {totals['peak_allocated_bytes']:,} B ({totals['peak_allocated_bytes']/2**30:.3f} GiB), reserved {totals['peak_reserved_bytes']:,} B ({totals['peak_reserved_bytes']/2**30:.3f} GiB)."
    ko = f'''# Case 013 후속 — 잘못된 도구 이름의 생성과 오류 복구

도구 호출의 이름을 실제 등록명으로 제한하는 decoder와 오류 안내를 구현하고, RTX 5080에서 기존 8개 DEV 과제를 다시 실행했습니다. D16은 기본 조건 {d16base['successes']}/8에서 결합 조건 {d16both['successes']}/8로 바뀌었습니다. 결합 조건의 미등록 이름 이벤트는 {d16both['unregistered_name_events']}회입니다. D4의 결합 조건은 {d4both['successes']}/8로, 이름 계약과 전체 과제 성공을 분리해 보고합니다.

[문제](#문제) · [수정](#수정) · [실험](#실험) · [결과](#결과) · [해석](#해석) · [검증과-비용](#검증과-비용) · [재현](#재현)

## 문제

기존 D16은 `apply_fixture_patch` 대신 `ApplyFixturePatch`, `validate_fixture_test` 대신 `ValidateFixtureTest`를 생성했습니다. parser는 모델이 만든 이름을 그대로 전달했고, registry가 미등록 이름을 거절했습니다. 인자는 맞았지만 작업은 실행되지 않았습니다. 오류 후 같은 이름을 반복하고 마지막에 `valid:true`라고 답해도 실제 수정/검증 조건을 만족하지 못했습니다. 이 사례에서 기존 오류 안내는 이름과 인자 오류를 같은 문장으로 표시했습니다.

## 수정

- BASELINE: 원래 generic 오류와 자유 생성.
- FEEDBACK: 이름 오류와 인자 오류를 분리합니다. 이름 오류에는 실제 허용 이름을 알려주며 정답 인자나 gold는 제공하지 않습니다.
- ENUM: 열린 tool-call JSON의 최상위 이름 문자열만 실제 schema enum으로 제한합니다. 이름 후보의 token score를 mask하며, 호출 여부·인자·최종 답은 자유롭게 생성합니다.
- BOTH: 두 수정을 결합합니다.

생성된 이름을 사후 alias로 고치지 않습니다. 잘못된 인자를 정답으로 바꾸지 않습니다. 원 checkpoint와 양자화 artifact, 252개 projection, beta transport, activation 양자화와 attention/KV 계산은 그대로입니다. 일반 final JSON은 제약 대상이 아니며 이름/key escape 표기는 이 ASCII 계약에서 지원하지 않습니다. 임시 tokenizer 문자열/enum mask cache는 별도 메모리를 사용합니다.

## 실험

Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f17c01a7cc42f958dc650907174af0554. BF16/RTN 기본 대조와 D16/D4의 네 조건을 순차 실행했습니다. 기존 document/combine/updates/fixture-patch 네 family의 각 2개, 총 8개 과제를 모든 조건이 공유합니다. 첫 입력 hash는 method·조건 간 대응시켰고, 이후 feedback/history의 차이는 의도한 개입입니다. 이 입력들은 이미 본 DEV이며 새 held-out 표본이 아닙니다.

greedy, batch1, 응답당 최대 192 token, episode당 모델 호출 8회/도구 호출 6회/900초, context limit 24,576. request 안 KV ON, turn 간 prefix cache OFF. SDPA, 원 BF16 reference 경로와 TF32/BF16 reduced-reduction OFF를 유지했습니다. 모델은 하나씩 로딩하고 RTN/D16 artifact 각 252개는 실제 hash/shape를 확인해 읽었습니다. backbone 학습·solver·download·기존 환경 upgrade는 0입니다.

첫 GPU forward 전에 [등록](preregistration.json), [입력 identity](input_manifest.json), code hash와 보호 원본 inventory를 저장했습니다. 별도 결과 폴더에 10개 logical 조건 / 80개 episode를 한 번씩 실행했습니다. 이 수를 80개의 독립 task로 세지 않습니다.

## 결과

전체 과제 성공은 실제 원 채점 계약의 성공 수 / 8개 과제입니다. 숫자가 큰 쪽이 좋습니다. fixture patch는 최종 JSON뿐 아니라 실제 expected config와 성공 validation 이벤트를 요구합니다.

{table}

오류 이벤트 수는 episode 수와 다릅니다. FEEDBACK은 첫 오류를 허용한 뒤 복구할 수 있고 ENUM은 이름의 생성 자체를 제한합니다.

{details}

같은 task ID의 paired 획득/손실:

{pairs}

D16의 안내 개선은 두 실패를 회복했지만 미등록 이벤트가 8회에서 4회로 줄었을 뿐 첫 잘못된 호출은 남았습니다. 이름 제한은 같은 두 실패를 회복하면서 미등록 이벤트를 0회로 만들었습니다. ENUM과 BOTH는 여기서 같은 성공/오류 결과였고, 이름 제한 조건에 다른 tool 오류가 없었으므로 결합한 안내의 추가 이득은 확인하지 못했습니다.

`diova-compare`로 같은 8개 입력을 비교해도 D16의 2개 회복/새 손실 0개가 일치했습니다. D16의 추출된 최종 JSON은 두 조건에서 모든 입력이 같았습니다. 성공 차이는 답 문장을 고친 결과가 아니라 실제 patch/validation 실행 여부의 차이입니다. CLI bootstrap은 POST_HOC_SAME_RECORDS의 기술적 구간이며 DEV 일반화 검정이 아닙니다. [실제 비교 receipt](compare_verification.json)를 함께 제공합니다.

미성공 task 목록:

{failures}

각 응답의 원문·token IDs·도구 이벤트·최종 config·decoder 개입 trace는 `workers/<method>/attempt01/episodes.jsonl`에 있습니다. 정답률은 report 요약을 oracle로 사용하지 않고 CPU에서 다시 계산했습니다. 모든 조건을 공개 준비에 포함하며 좋은 결과만 고르지 않았습니다.

## 해석

이름 생성과 오류 후 행동을 개선한 시스템 조건이 해당 DEV에서 어떤 실패를 회복했는지 확인한 실험입니다. 이름 제한에 의해 원래 높은 점수의 잘못된 token을 mask하는 경우는 별도 trace로 남겼습니다. 오류 안내만으로 회복되면 잘못된 첫 선택 자체가 사라졌다고 설명하지 않습니다. 이름 계약을 통과해도 도구 선택·인자·사실/상태 추적·최종 출력 형식의 실패는 남을 수 있습니다.

가중치 MSE/PPL이나 수치 경로를 개선한 결과가 아닙니다. 어느 projection이 오류의 근본 원인인지도 확정하지 않았습니다. 앞선 한 projection BF16 복원 실험과 그 실패 판정은 그대로 유지합니다. 기존 Case 013의 자유 생성/주 점수를 이번 constrained 조건으로 교체하지 않습니다. 이 8개 DEV의 회복을 일반화·비열등성·stock GPTQ 우위·긴 문맥 capacity·packed serving speedup으로 확대하지 않습니다.

## 검증과 비용

{resource}

worker wall에는 모델 로딩·artifact 읽기·CPU 이름 mask·serialization/idle이 포함됩니다. 생성 wall도 decoder CPU 비용을 포함하며 TTFT/TPOT나 native INT4 kernel 시간으로 표현하지 않습니다. 장치 전체/Windows peak는 미측정입니다. 기존 배경 GPU 작업을 종료하지 않았습니다.

이번 고유 CPU 기능 tests 7개 PASS와 실제 tokenizer의 mask smoke 1건을 수행했습니다. 반복 실행을 고유 tests에 더하지 않았습니다. 독립 stdlib auditor가 raw→parsed 이름/인자, 모든 도구 결과/최종 상태, 원 성공 조건, pairing/분모/누락·중복, 기존 원본 보존을 검산했습니다. 이전 39·50개 tests receipt를 새 실행 수로 합산하지 않았습니다. logits/전체 weight를 CPU에서 다시 계산했다는 뜻이 아닙니다.

기존 Case013의 닫힌 공개 집합 519개 파일 검사도 통과했습니다. 첫 실행은 system Python의 NumPy 누락으로 실패했고, 패키지 설치 없이 기존 reference Python에서 CPU 전용으로 재검증했습니다. 두 attempt를 [검증 이력](validation_attempts.json)에 남겼습니다.

기존 Case001–013 5,743개 파일, r2·이전 follow-up·원 source가 작업 전후 일치했습니다. 큰 private 배열/payload의 보존은 기존 size/mtime 비교이며 실제 사용한 RTN/D16 projection payload의 SHA는 로딩 때 확인했습니다. 원 checkpoint 전체를 다시 해시하지 않았습니다. 기존 환경 수정과 GitHub/PR/main/Pages 작업은 0입니다. UI 변경이 없어 browser 검사는 이번에 수행하지 않았습니다.

## 재현

[CPU/실제 GPU 재실행 안내](README.ko.md). 실험 완료 상태와 범위는 [decision](decision.json), CPU 이벤트 재검산은 [verification](verification.json), 실제 command/exit/wall은 [GPU receipt](gpu_commands.json)에서 확인합니다. 모델과 대형 양자화 artifact는 검토 ZIP에 포함하지 않습니다. 원형 INT4 구현/fixture harness는 기존 연구를 재사용하고, 이 작업은 decoder·오류 안내·factorial 실행·독립 검산을 추가했습니다. Codex가 구현과 기록 정리를 지원했습니다.
'''
    en = f'''# Case 013 follow-up: tool-name generation and recovery

Implemented registry-aware error feedback and a tool-name enum decoder, then reran the original eight DEV tasks on RTX 5080. D16 changed from {d16base['successes']}/8 baseline to {d16both['successes']}/8 with both changes; the combined condition had {d16both['unregistered_name_events']} unregistered-name events. D4 combined success was {d4both['successes']}/8. These are system mitigation results on observed DEV, not a new quantizer or held-out evaluation.

## Method

BASELINE retains free decoding and generic errors. FEEDBACK distinguishes unknown names from invalid arguments and lists actual registered names. ENUM masks only the top-level tool-call name string; tool use, arguments and final answers remain model-generated. BOTH combines the changes. No aliases, answer repair or gold-dependent decoding are used. ASCII names/keys and batch1 are supported; escaped name/key forms are rejected.

The same Qwen3-4B revision cdbee75f17c01a7cc42f958dc650907174af0554, saved RTN/D16 artifacts, 252 projections and dequantized BF16 reference arithmetic are unchanged. D4 retains the existing activation variant. Greedy generation: 192 output tokens per call, 8 model/6 tool calls per episode, 900-second deadline and 24,576 context limit. Within-request KV is on; across-turn prefix cache is off. Initial prompts are paired; later feedback differences are the registered intervention. Four families, two tasks each, are reused across all conditions.

## Results

Task success uses the original scorer. Patch tasks require actual expected config and successful validation, in addition to final JSON.

{table}

Event counts are not failed-episode counts:

{details}

Paired counts on the same task IDs:

{pairs}

Failed task IDs:

{failures}

## Interpretation

Feedback recovers D16's two failures while retaining four initial unknown-name events. Enum decoding recovers the same failures with zero unknown names. ENUM and BOTH have the same observed success/error counts; because no other tool errors occur under enum decoding, an additional benefit from feedback in the combined arm is not demonstrated. The existing diova-compare CLI independently agrees on two recovered tasks and no new losses. D16's extracted final answers are identical across baseline/BOTH; the difference is actual fixture execution. CLI intervals are POST_HOC_SAME_RECORDS descriptive outputs.

This tests whether decoding/feedback changes recover previously observed task failures. Feedback may repair behavior after the first erroneous call; enum decoding may prevent that name from being emitted. Neither establishes a numerical fix to DualQuant. Correct names do not guarantee correct tool choice, arguments, state tracking or final answers. Original Case 013 and the earlier unsuccessful single-projection restoration remain unchanged. No generalization, noninferiority, packed speed or context-capacity benefit is established.

## Verification and cost

{resource}

Worker wall includes loading, artifact reads, CPU masking and serialization. It is not TTFT, TPOT or native INT4 kernel time. Torch allocated/reserved differ from device-wide and Windows memory, which were not sampled. Seven unique CPU tests and one real-tokenizer mask smoke passed. A stdlib audit independently replayed tool events/config, original success rules, pairs and denominators; historical test counts were not added. No full-logit/weight recomputation was claimed.

All 5,743 Case001–013 files and original r2/prior follow-up/source inventories are preserved. Large private artifact/array preservation uses size/mtime; used projection payload hashes were checked on load. No solver, training, download, existing-environment upgrade or remote publication. Model weights and large artifacts are excluded from the review bundle. The original reference and harness are reused; this work adds the decoder, feedback, bounded factorial runner and audit. Codex assisted implementation and reporting.

The closed Case 013 public verifier also passed for 519 files. Its first system-Python attempt lacked NumPy; the existing reference Python then passed the CPU-only check without installing packages. Both attempts are retained in validation_attempts.json.

## Reproduce

See [the execution guide](README.ko.md), [decision](decision.json), [independent verification](verification.json) and [actual GPU commands](gpu_commands.json). The bundle's CPU replay explicitly skips certification of absent external originals. The full GPU run requires the existing reference environment, original model and saved projection artifacts.
'''
    (result / 'REPORT_KO.md').write_text(ko)
    (result / 'REPORT.md').write_text(en)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--result', type=Path, required=True)
    main(p.parse_args().result)
