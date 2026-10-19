"""Small Pages projection of the completed, separately audited P2 diagnostic."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from common import read, require

P2 = 'cases/015-tool-interface-sensitivity/followups/json-recovery-p2-v1'
SOURCES = {
    'case015-p2-report-ko.md.txt': P2 + '/REPORT.ko.md',
    'case015-p2-report-en.md.txt': P2 + '/REPORT.md',
    'case015-p2-reproduce-ko.md.txt': P2 + '/REPRODUCE.ko.md',
    'case015-p2-reproduce-en.md.txt': P2 + '/REPRODUCE.md',
    'case015-p2-decision.json.txt': P2 + '/results/gpu_r2/decision.json',
    'case015-p2-cpu-check.py.txt': P2 + '/publication/run_checks.py',
}


def load_p2(root):
    p2 = Path(root) / P2
    if not p2.exists():
        return None
    # Isolate the frozen CPU import guard from the rest of the site builder.
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='case015-p2-display-') as tmp:
        output = Path(tmp) / 'audit'
        result = subprocess.run(
            [sys.executable, '-B', str(p2 / 'execution/audit.py'),
             '--results', str(p2 / 'results/gpu_r2'), '--output', str(output)],
            env=env, capture_output=True, text=True, timeout=120)
        require(result.returncode == 0, 'Case015 P2 raw endpoint replay failed')
        summary = read(output / 'summary.json')
    require(summary == read(p2 / 'results/gpu_r2/analysis/summary.json'),
            'Case015 P2 stored summary differs from raw replay')
    require(summary['status'] == 'COMPLETED', 'Case015 P2 is incomplete')
    compact = []
    for method in ('BF16', 'RTN', 'D16'):
        item = {'method': method}
        for arm in ('C0', 'G', 'N', 'E'):
            rows = [r for r in summary['table']
                    if r['method'] == method and r['arm'] == arm]
            count = sum(r['workflow_denominator'] for r in rows)
            all_na = all(r['not_applicable_cells'] == r['planned_checkpoints']
                         for r in rows)
            item[arm] = {
                'success': sum(r['workflow_successes'] for r in rows),
                'denominator': count,
                'status': 'OBSERVED' if count else
                          'NOT_APPLICABLE' if all_na else 'NOT_RUN',
            }
        compact.append(item)
    decision = read(p2 / 'results/gpu_r2/decision.json')
    require(compact == decision['method_summary'],
            'Case015 P2 decision differs from raw endpoints')
    return {
        'summary': compact, 'C0_reproduction': summary['C0_reproduction'],
        'selected_fixtures': summary['independent_selected_fixtures'],
        'API_checkpoints': summary['API_checkpoints'], 'costs': summary['costs'],
        'execution_source_commit': decision['execution_source_commit'],
        'scope': 'Selected checkpoint recovery, separate from P1/A1 and not held-out tasks',
    }


def home_p2(data, lang, link, prefix=''):
    p2 = data.get('P2')
    if not p2:
        return ''
    row = next(r for r in p2['summary'] if r['method'] == 'D16')
    result = f"{row['G']['success']}/{row['G']['denominator']}"
    text = (f'별도 후속 P2: 같은 조회·계산 결과에서 첫 호출문의 형식을 제한하자 '
            f'D16의 실제 작업 완료가 {result}로 회복됐습니다.' if lang == 'ko' else
            f'Separate P2 followup: constraining the first call format after the same '
            f'lookup/calculation results recovered {result} complete D16 workflows.')
    label = '조건·대조군·남은 실패 보기' if lang == 'ko' else 'Conditions, controls and remaining failures'
    return '<p class="method-key">' + text + ' ' + link(prefix + '#p2-recovery', label) + '</p>'


def render_p2(data, lang, link, table):
    p2 = data.get('P2')
    if not p2:
        return ''
    ko = lang == 'ko'
    txt = lambda k, e: k if ko else e
    rows = []
    for row in p2['summary']:
        cells = [row['method']]
        for arm in ('C0', 'G', 'N', 'E'):
            r = row[arm]
            cells.append(f"{r['success']}/{r['denominator']}" if r['status'] == 'OBSERVED'
                         else txt('해당 없음', 'Not applicable')
                         if r['status'] == 'NOT_APPLICABLE' else txt('미측정', 'Not run'))
        rows.append(cells)
    intro = txt(
        '앞선 시험에서 D16은 합계를 저장할 호출문의 따옴표·항목 이름을 잘못 써 멈췄습니다. '
        '후속 P2는 같은 조회·계산 결과를 이미 받은 지점에서 다시 시작합니다. '
        'C0는 원래 생성, G는 첫 응답의 형식 제한, N은 중립 재시도, E는 오류 설명을 더한 재시도입니다.',
        'D16 previously stopped after breaking quotes or field names in a save call. '
        'P2 resumes after the same lookup/calculation results have arrived. '
        'C0 is original generation; G constrains the first response format; '
        'N retries neutrally; E adds a parse-error explanation.')
    body = '<section id="p2-recovery"><h3>' + txt(
        'P2 후속: 같은 정보에서 호출문 실패를 복구할 수 있을까요?',
        'P2 followup: can the same-observation call failure recover?') + '</h3><p>' + intro + '</p>'
    body += table([txt('모델', 'Model'), txt('C0 원형', 'C0 original'),
                   txt('G 문법 제약', 'G schema'), txt('N 중립 재시도', 'N neutral retry'),
                   txt('E 오류 설명', 'E error feedback')], rows, 'P2 verified workflow completion')
    body += '<p>' + txt(
        f"분모는 이미 선택된 합성 구매 기록 {p2['selected_fixtures']}개 × 이름 표기 2종의 "
        f"{p2['API_checkpoints']}개 지점입니다. 새 업무의 성공률이 아닙니다. "
        '성공은 실제 저장·최신 검증 결과 수신·최종 답변까지 완료한 경우입니다. '
        'N은 한 건에서 저장만 회복했고, E는 N보다 완료를 늘리지 못했습니다. '
        'BF16의 후속 검증 ID 오류는 남았습니다. BF16/RTN의 N/E는 적용할 오류가 없었습니다.',
        f"The denominator is {p2['API_checkpoints']} checkpoints from "
        f"{p2['selected_fixtures']} previously selected synthetic records × two naming versions, "
        'not new-task success rates. Success requires actual saving, receipt of current validation '
        'and a final answer. N saved the correct state once; E added no workflow completion over N. '
        'A later BF16 validation-ID error remains. BF16/RTN had no eligible N/E failure.') + '</p>'
    costs = p2['costs']
    body += '<p>' + txt(
        f"새 생성 {costs['actual_new_request_attempts']}회 = 최초 진단 "
        f"{costs['first_diagnostic_responses']} + 후속 {costs['continuation_responses']}회입니다. "
        '원 P1/A1 점수와 합산하지 않으며, 저장본을 BF16으로 복원한 품질 진단입니다.',
        f"{costs['actual_new_request_attempts']} new requests = "
        f"{costs['first_diagnostic_responses']} focal + {costs['continuation_responses']} continuation. "
        'These are separate from P1/A1; quality uses dense BF16 reload of saved weights.') + '</p>'
    body += '<p>' + link('../sources/case015-p2-report-' + lang + '.md.txt',
                        txt('P2 전체 보고서', 'Full P2 report')) + ' · ' + link(
        '../sources/case015-p2-reproduce-' + lang + '.md.txt',
        txt('P2 CPU/GPU 재현', 'P2 CPU/GPU reproduction')) + '</p></section>'
    return body
