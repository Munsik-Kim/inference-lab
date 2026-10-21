"""Shared bilingual publication copy, with all headline numbers from raw-backed groups.

This presentation module changes no experimental prompt, schema or endpoint.
"""

def public_text(data, lang):
    if lang not in ('ko', 'en'):
        raise ValueError('Unsupported language')
    ko = lang == 'ko'
    t = lambda k, e: k if ko else e
    groups = {g['policy']: g for g in data['groups'] if g['stage'] == 'P2' and g['kv'] == 'K1'}
    baseline, grammar = groups['C0'], groups['G']
    pair = next(r for r in data['comparisons'] if r['stage'] == 'P2' and r['comparison_id'] == 'K1:G-C0')
    change = {key: 100 * (grammar[key+'_per_verified'] / baseline[key+'_per_verified'] - 1)
              for key in ('input_tokens', 'output_tokens', 'wall_seconds')}
    c, g, n = baseline['verified'], grammar['verified'], grammar['n']
    s0, s1 = baseline['state_correct'], grammar['state_correct']
    recover, loss = pair['recovery'], pair['loss']
    gate = data['gate']
    return {
        'subtitle': t('호출 규칙·검증된 업무 완주·실행 비용',
                      'Tool call policies, verified task completion, and execution cost'),
        'intro': t('AI가 상품 합계를 계산해 저장하고, 창고 사이의 재고를 옮기며, 문서 정보를 바꾸는 과정을 시험했습니다. 출력 형식을 제한하거나 호출 오류 뒤 한 번 다시 시도하는 규칙이, 저장된 값·검증 결과·최종 답까지 맞게 완료하는 비율을 어떻게 바꾸는지 비교했습니다.',
                   'We tested an AI agent on synthetic records: calculating and saving order totals, transferring inventory, and updating document metadata. The benchmark checks the saved state, the validation results the agent observed, and its final answer. It compares output constraints and one retry after a contract error.'),
        'implementation': t('직접 구현한 것은 호출을 엄격하게 읽는 실행기, 안전한 가상 기록, 생성 전에 받은 정보를 추적하는 도구와 저장 결과를 재생하는 감사기입니다.',
                            'We implemented the strict action parser and runner, isolated synthetic records, pre-generation observation tracking, and an auditor that replays saved results.'),
        'kv_intro': t('모델 가중치는 같은 Qwen3-4B BF16으로 고정했습니다. 모델이 앞선 내용을 처리하며 저장하는 중간 계산값을 KV 캐시라고 합니다. BF16 KV(K0)는 16비트 기준선, NVFP4 KV(K1)는 4비트 값과 배율(scale)을 함께 저장하는 구성입니다.',
                      'Qwen3-4B weights stay BF16 in both configurations. The KV cache stores intermediate attention values for preceding text. BF16 KV (K0) is the 16-bit baseline; native NVFP4 KV (K1) stores packed four-bit values with scales.'),
        'policy_intro': t('자유 출력(C0)은 형식을 프롬프트로 안내하고 첫 호출 규칙 위반에서 종료합니다. 문법 제약(G)은 매 응답에 공개 JSON schema를 적용합니다. 중립 재시도(N)와 오류 설명 재시도(E)는 첫 위반 뒤 각각 중립 안내 또는 공개 오류 코드·위치·기대 타입으로 한 번 다시 시도합니다. schema는 정답값이나 다음 행동을 알려주지 않습니다.',
                          'Free output (C0) gives format instructions in the prompt and stops at the first contract error. Grammar (G) applies a public JSON schema to every response. Neutral retry (N) and error retry (E) each offer one retry after the first error, using a neutral message or the public error code, path and expected type. The schema supplies neither correct values nor the next action.'),
        'headline': t(f'같은 BF16 가중치를 사용하는 NVFP4 KV 구성에서, 매 응답에 JSON 문법을 적용하자 신규 합성 {n}과제의 완주 수가 {c}/{n}에서 {g}/{n}로 늘었습니다. 자유 출력에서 실패한 {recover}개가 회복됐고 손실은 {loss}개였습니다. 실제 저장된 값이 맞은 과제는 {s0}/{n}에서 {s1}/{n}로 늘었지만, 수정 후 최신 검증 결과를 받지 않은 경우는 업무 완주로 세지 않았습니다.',
                      f'With BF16 weights and native NVFP4 KV, applying a public JSON schema to every response increased verified completion from {c}/{n} to {g}/{n} on newly generated synthetic tasks: {recover} recoveries and {loss} losses relative to free output. Correct saved states increased from {s0}/{n} to {s1}/{n}. A correct state without observed validation of the updated state did not count as verified completion.'),
        'paired': t(f'완주 차이는 {pair["difference_pp"]:.2f}pp이며, 탐색적 95% 구간은 [{pair["ci_95_pp"][0]:.2f}, {pair["ci_95_pp"][1]:.2f}]pp입니다. pp는 완주율의 퍼센트포인트 차이입니다.',
                    f'The completion difference is {pair["difference_pp"]:.2f} percentage points (pp), with an exploratory 95% interval of [{pair["ci_95_pp"][0]:.2f}, {pair["ci_95_pp"][1]:.2f}]pp.'),
        'cost': t(f'완주 수가 늘면서 전체 실행 비용도 늘었습니다. 실패를 포함한 {n}개 실행의 비용을 완주 수로 나누면 입력 토큰은 약 {change["input_tokens"]:.2f}%, 출력 토큰은 {change["output_tokens"]:.2f}%, 한 과제 실행 시간(episode wall)은 {change["wall_seconds"]:.2f}% 증가했습니다.',
                  f'More tasks completed, while total execution costs also increased. Dividing all {n} runs, including failures, by completions gives approximately {change["input_tokens"]:.2f}% more input tokens, {change["output_tokens"]:.2f}% more output tokens and {change["wall_seconds"]:.2f}% more episode-wall time per completion.'),
        'deviation': t(f'확대 기준 이탈: 설정 점검용 DEV에서 BF16 KV+C0의 완주는 {gate["observed_dev_verified"]}/{gate["denominator"]}로 원안의 {gate["required_dev_verified"]}/{gate["denominator"]} 기준에 미달했습니다. 당시 로컬 계획과 실행 코드가 이 성공 조건을 누락하고 완료·동결·봉인만 검사해 P2를 진행했습니다. 별도 면제 승인 기록은 확인하지 못했습니다. 이 경위를 공개하며, P2는 해당 기준을 통과한 확인시험이 아닌 신규 합성 과제의 탐색적 측정으로 제시합니다.',
                       f'Protocol deviation: BF16 KV with free output completed {gate["observed_dev_verified"]}/{gate["denominator"]} DEV tasks, below the original {gate["required_dev_verified"]}/{gate["denominator"]} expansion criterion. The local plan and execution gate omitted this success threshold and proceeded to P2 after checking completion, freezing and sealing. No separate waiver approval was found. We disclose this deviation and present P2 as an exploratory measurement on new synthetic tasks.'),
        'home_intro': t('도구 호출 규칙이 실제 저장·검증·최종 답의 완료를 얼마나 바꾸는지 비교하는 평가기를 구현했습니다. 상품 합계 기록·창고 이동·문서 정보 변경을 소유한 가상 기록에서 시험합니다.',
                        'We built a benchmark that checks how tool call policies change completion of saved state, observed validation and final answers. It uses owned synthetic records for product totals, inventory transfers and document updates.'),
        'home_result': t(f'BF16 가중치를 고정한 NVFP4 KV의 신규 합성 {n}과제에서 문법 제약은 완주 {c}/{n}→{g}/{n}, 상태 정확 {s0}/{n}→{s1}/{n}를 기록했습니다. 회복 {recover}개·손실 {loss}개와 함께 추가 실행 비용과 남은 실패도 추적했습니다.',
                         f'With BF16 weights and native NVFP4 KV, grammar changes completion {c}/{n}→{g}/{n} and correct saved states {s0}/{n}→{s1}/{n} on {n} new synthetic tasks: {recover} recoveries and {loss} losses. Added execution costs and remaining failures are reported alongside the gain.'),
        'scope': t('한 모델·한 장비의 짧은 합성 과제 비교입니다. 최신 검증을 놓친 실패, 상품 식별자(SKU) 누락, 문서 상태값 오류를 함께 보존했습니다. P3 동시 처리량은 미실행입니다.',
                   'This compares short synthetic tasks on one model and device. Failures involving stale validation, missing SKU prefixes and document status values are retained. P3 concurrent throughput was not run.'),
    }
