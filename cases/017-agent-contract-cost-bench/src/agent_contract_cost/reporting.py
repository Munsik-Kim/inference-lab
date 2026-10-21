"""Post-measurement bilingual reports from audited scalar records, never inference."""
import argparse
import html
import json
from collections import Counter
from pathlib import Path

from .common import CASE, read, sha_file, utc, write_once


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def number(value, precision=3):
    return "—" if value is None else f"{value:.{precision}f}"


def policy_table(summary):
    groups = {(g['kv'], g['policy']): g for g in summary['groups']}
    return table(['KV', 'C0 FREE', 'G GRAMMAR', 'N NEUTRAL', 'E ERROR'],
                 [[kv] + [f"{groups[kv,p]['verified_success']}/{groups[kv,p]['observed']}" +
                           (f" ({groups[kv,p]['missing_or_unobserved']} missing)" if groups[kv,p]['missing_or_unobserved'] else '')
                           for p in ('C0','G','N','E')] for kv in ('K0','K1')])


def cost_table(summary):
    return table(['KV', 'Policy', 'Requests', 'Output tokens', 'Retries', 'Episode wall sum (s)', 'Tokens / verified job'],
                 [[g['kv'], g['policy'], g['model_request_attempts'], g['generated_output_tokens'],
                   f"{g['retry_consumed']}/{g['retry_triggered']}", number(g['wall_sum_seconds']),
                   number(g['tokens_per_verified_success'], 1)] for g in summary['groups']])


def pair_table(summary):
    return table(['Comparison', 'Pairs', 'Both success', 'Loss', 'Recovery', 'Both fail', 'Difference (pp)', '95% interval (pp)'],
                 [[c['comparison_id'], c['observed_pairs'], c['counts'].get('both_success',0), c['counts'].get('loss',0),
                   c['counts'].get('recovery',0), c['counts'].get('both_fail',0), number(c['effect_percentage_points'],2),
                   '—' if c['ci_95_fraction'] is None else '[' + ', '.join(number(100*x,2) for x in c['ci_95_fraction']) + ']']
                  for c in summary['comparisons']])


def figure(summary, destination):
    groups = summary['groups']
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 380" role="img">',
             '<title>Verified workflow success; higher is better</title>',
             '<rect width="800" height="380" fill="white"/>',
             f'<text x="20" y="26" font-family="sans-serif" font-size="18">{html.escape(summary["stage"])} — verified / planned tasks</text>']
    for i,g in enumerate(groups):
        y=52+i*35; width=500*g['verified_success']/g['planned'] if g['planned'] else 0
        parts += [f'<text x="20" y="{y+16}" font-family="sans-serif" font-size="14">{g["kv"]} / {g["policy"]}</text>',
                  f'<rect x="125" y="{y}" width="500" height="23" fill="#e2e8f0"/>',
                  f'<rect x="125" y="{y}" width="{width}" height="23" fill="{"#2563eb" if g["kv"]=="K0" else "#0f766e"}"/>',
                  f'<text x="640" y="{y+16}" font-family="sans-serif" font-size="14">{g["verified_success"]}/{g["planned"]}</text>']
    parts += ['<text x="125" y="355" font-family="sans-serif" font-size="12">0%                         25%                         50%                         75%                      100%</text>',
              '<text x="20" y="377" font-family="sans-serif" font-size="11">Zero-origin axis. Missing/unobserved cells are listed separately; no imputation.</text>', '</svg>']
    destination.write_text('\n'.join(parts)+'\n')


def build(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    p0=read(CASE/'results/p0_r1/gates.json')
    stages={name:read(CASE/f'results/{name.lower()}_r1/analysis/summary.json')
            for name in ('P1','P2') if (CASE/f'results/{name.lower()}_r1/analysis/summary.json').exists()}
    if set(stages)!={'P1','P2'} or any(s['status']!='COMPLETE' for s in stages.values()):
        raise ValueError('This final measured report requires both complete audited stages; missing is not success')
    from .review import load_review
    load_review(CASE)  # raw identity and display counts must agree before writing claims.
    protocol=read(CASE/'specs/protocol.json')
    cache={kv:read(CASE/f'results/p0_r1/{kv}_cache_before.json') for kv in ('K0','K1')}
    capacity=table(['KV','Actual storage','Bytes/token','Actual pool bytes','Token slots'],
        [[kv,'BF16' if kv=='K0' else 'packed uint8 + FP8 block scales',
          c['actual_unique_storage_bytes']//c['cache_config']['kv_cache_size_tokens'],
          c['actual_unique_storage_bytes'],c['cache_config']['kv_cache_size_tokens']] for kv,c in cache.items()])
    worker=[read(CASE/f'results/{s.lower()}_r1/completion.json') for s in stages]
    total_requests=p0['actual_model_request_attempts']+sum(g['model_request_attempts'] for s in stages.values() for g in s['groups'])
    total_tokens=p0['generated_output_tokens']+sum(g['generated_output_tokens'] for s in stages.values() for g in s['groups'])
    total_worker=p0['worker_wall_seconds']+sum(s['worker_wall_seconds'] for s in worker)
    diagnostics=[];examples=[]
    for stage in stages:
        for p in sorted((CASE/f'results/{stage.lower()}_r1/cells').glob('*.json')):
            ep=read(p);identity=ep['identity']
            errors=[r['parse']['error'] for r in ep['requests'] if r.get('parse') and not r['parse']['accepted']]
            diagnostics.append({'stage':stage,'task_id':identity['task_id'],'kv':identity['kv'],'policy':identity['policy'],
                'terminal_reason':ep['terminal_reason'],'first_unmet_obligation':ep['reported_endpoint']['first_unmet_obligation'],
                'contract_errors':errors,'source_file':p.relative_to(CASE).as_posix(),'source_sha256':sha_file(p)})
            if ep['terminal_reason']!='VERIFIED_SUCCESS' and len(examples)<8:
                examples.append(f"- [{p.stem}]({p.relative_to(CASE).as_posix()}): `{ep['terminal_reason']}`; "
                                f"first unmet `{ep['reported_endpoint']['first_unmet_obligation']}`.")
    write_once(output/'failure_ledger.json',diagnostics)
    terminal=table(['Stage','Terminal reason','Episodes'],[[s,k,v] for s in stages for k,v in sorted(Counter(d['terminal_reason'] for d in diagnostics if d['stage']==s).items())])
    provenance=[]
    for s in stages:
        for name in ('audit/audited_rows.json','analysis/summary.json','completion.json'):
            p=CASE/f'results/{s.lower()}_r1/{name}'
            provenance.append({'source_file':p.relative_to(CASE).as_posix(),'sha256':sha_file(p)})
        figure(stages[s],output/(s.lower()+'_success.svg'))
    write_once(output/'display_provenance.json',{'generated_utc':utc(),'source_files':provenance,
        'model_requests':total_requests,'generated_output_tokens':total_tokens,'worker_wall_seconds':total_worker,
        'data_flow':'raw responses + SSE + events -> fresh fixture replay + separate endpoint -> analysis -> display',
        'model_revision':protocol['revision'],'evidence_kind':'posthoc_display_of_audited_measurements'})
    for lang in ('ko','en'):
        ko=lang=='ko'
        title='Case017 — 도구 호출 규칙을 지키면 AI가 일을 더 정확히 끝낼까?' if ko else 'Case017 — Agent Contract–Cost Bench'
        intro=('가상 기록을 읽고 계산하거나 수정한 뒤, 실제 저장된 상태와 마지막 답이 모두 맞는지 시험했습니다. '
               '상품별 개수와 한 개 가격으로 합계를 구하는 작업, 창고 사이에 물건을 옮기는 작업, 문서의 상태·태그·담당자를 바꾸는 작업입니다. '
               '기본 모델 가중치는 그대로 두고, 출력 규칙 네 가지와 대화 기억(KV cache)의 저장 정밀도 두 가지를 비교했습니다.' if ko else
               'We tested whether an agent both changes a synthetic record correctly and returns a verified final answer. '
               'Tasks cover integer order totals, atomic warehouse transfers, and document metadata updates. '
               'BF16 model weights stay fixed while four output-contract policies and two native KV-cache precisions vary.')
        text=f'# {title}\n\n{intro}\n\n'
        text+=('[한눈에 보기](#overview) · [배경과 질문](#question) · [구현 계약](#implementation) · [방법·데이터](#method) · [결과](#results) · [분석](#analysis) · [결론](#conclusion) · [재현·출처](#reproduce)\n\n' if ko else
               '[Overview](#overview) · [Question](#question) · [Implementation](#implementation) · [Method](#method) · [Results](#results) · [Analysis](#analysis) · [Conclusion](#conclusion) · [Reproduction](#reproduce)\n\n')
        text+='## <a id="overview"></a>'+('한눈에 보기' if ko else 'Overview')+'\n\n'
        if 'P2' in stages:text+=policy_table(stages['P2'])+'\n\n'
        text+=('표는 새 HOLDOUT 24개 과제에서의 실제 업무 완주 수입니다. 파일 값만 맞거나 JSON만 유효한 것은 성공으로 세지 않습니다.\n\n' if ko else
               'The table reports verified completions on 24 new HOLDOUT tasks. Correct files alone or valid JSON alone do not count.\n\n')
        if 'P2' in stages:
            s=stages['P2'];gs={(g['kv'],g['policy']):g for g in s['groups']};cs={c['comparison_id']:c for c in s['comparisons']}
            c=cs['K1:G-C0'];e=cs['K1:E-N']
            text+=(f"NVFP4에서 G는 C0보다 완주 {c['counts'].get('recovery',0)}개를 회복했고 손실은 {c['counts'].get('loss',0)}개였습니다. "
                   f"차이는 {c['effect_percentage_points']:.2f} percentage points이며 탐색 구간은 "
                   f"[{100*c['ci_95_fraction'][0]:.2f}, {100*c['ci_95_fraction'][1]:.2f}]pp입니다. "
                   f"BF16에서는 네 정책의 완주수가 같았습니다. NVFP4의 E-N은 회복 {e['counts'].get('recovery',0)}개·손실 {e['counts'].get('loss',0)}개이며 구간은0을 포함합니다.\n\n" if ko else
                   f"On NVFP4, G recovered {c['counts'].get('recovery',0)} C0 failures with {c['counts'].get('loss',0)} losses: "
                   f"{c['effect_percentage_points']:.2f} percentage points, exploratory interval "
                   f"[{100*c['ci_95_fraction'][0]:.2f}, {100*c['ci_95_fraction'][1]:.2f}]pp. "
                   f"All BF16 policies completed the same number. NVFP4 E-N had {e['counts'].get('recovery',0)} recoveries and {e['counts'].get('loss',0)} loss; its interval includes zero.\n\n")
        text+=f"P0: {p0['status']}; total actual model requests: **{total_requests}**; output tokens: **{total_tokens}**; cumulative worker wall: **{total_worker:.3f}s**.\n\n"
        text+='## <a id="question"></a>'+('배경과 질문' if ko else 'Background and question')+'\n\n'
        text+=('Case015의 선택된 실패 복구와 Case016의 KV 저장 검증은 동기입니다. 이번 결과의 분모에 옮기지 않았습니다. '
               'Q1은 출력 규칙이 실제 완주와 비용을 바꾸는지, Q2는 같은 BF16 가중치에서 native NVFP4 KV로 바꿀 때 무엇이 달라지는지입니다.\n\n' if ko else
               'Case015 selected-failure recovery and Case016 cache verification motivate this study; their results are not included here. '
               'Q1 concerns contract-policy completion and cost. Q2 concerns native NVFP4 versus BF16 KV with identical BF16 weights.\n\n')
        text+='## <a id="implementation"></a>'+('기술 원리와 구현 계약' if ko else 'Technical principle and implementation contract')+'\n\n'
        text+=('직접 구현한 것은 엄격 parser, 정책·예산 실행기, 세 가지 안전 fixture, 관측 이력, 독립 endpoint 감사, 비교 도구입니다. '
               'vLLM·FlashInfer·xgrammar의 엔진과 커널은 upstream 구현입니다. KV cache는 한 요청 안에서 앞선 토큰의 key/value를 저장하며, 매 턴 전체 history를 재전송하고 prefix caching은 끕니다.\n\n' if ko else
               'Our implementation supplies the strict parser, policy/budget runner, three safe fixtures, observation tracking, independent endpoint audit, and comparison tools. '
               'Engine, kernels, and constrained decoding come from vLLM, FlashInfer, and xgrammar. Full history is resent each turn; prefix caching is disabled.\n\n')
        text+=table(['Policy','Contract'],[['C0','Free generation; first contract rejection terminates'],['G','Public branch-specific JSON schema on every response; no retry'],['N','Free generation; one neutral retry after first contract rejection'],['E','Free generation; one retry with public error code/path/type']])+'\n\n'
        text+=('모두 동일 endpoint·template·단일 JSON envelope·parser를 사용합니다. G에는 정답 ID·금액·행동 순서를 넣지 않습니다. '
               '`uniqueItems`는 설치 backend가 지원하지 않아 모든 정책의 host 검사로만 적용합니다. '
               '여섯 번째 action이 final이면 허용하며, 다른 도구이면 실행 후 종료합니다. final도 예산을 쓰므로 일곱 번째 생성은 없습니다.\n\n' if ko else
               'All policies share endpoint, template, one JSON envelope, and parser. G receives no gold IDs, amounts, or action order. '
               '`uniqueItems` remains a common host-only check because the installed backend does not support it. '
               'A sixth final is allowed; a sixth nonfinal executes then terminates. Final consumes an action, so there is no seventh generation.\n\n')
        text+=capacity+'\n\n'
        text+=('이 비율은 같은 약 4GiB KV pool의 token 슬롯 비율입니다. 전체 VRAM 감소나 단일 100K context 지원이 아닙니다.\n\n' if ko else
               'Capacity concerns token slots in the same approximately 4GiB KV pool; it is not total VRAM reduction or a 100K single-context claim.\n\n')
        text+='## <a id="method"></a>'+('방법·실험' if ko else 'Methods and experiments')+'\n\n'
        text+=f"Qwen/Qwen3-4B-Instruct-2507; revision `{protocol['revision']}`; BF16 weights; vLLM 0.32.0; FlashInfer; xgrammar; TP1; GRAPH; episode concurrency1.\n\n"
        text+='Actual numerical settings: TF32=False; BF16 reduced-precision reduction=True, identical across KV configurations.\n\n'
        text+=('DEV8(A3/B3/C2)×8조건=64개, 이후 동결된 생성기로 다른 template·상태의 HOLDOUT24(A/B/C 각8)×8조건=192개를 생성·봉인했습니다. '
               'COST6은 CPU 정답 경로만 준비했고 P3는 미실행입니다. 초기 입력은 약 2K 목표이며 실제 길이는 data/*_model_inputs.json에 있습니다. '
               '같은 task의 최초 token IDs는 KV·정책 사이에 일치해야 하며 이후 자유 이력은 달라질 수 있습니다.\n\n' if ko else
               'DEV8 (A3/B3/C2) yields 64 episodes. After P1, the frozen generator seals HOLDOUT24 (eight per family), with different templates and structural states, yielding 192 episodes. '
               'COST6 has CPU gold paths only; P3 is not run. Initial prompts target approximately 2K tokens; actual lengths are recorded. Initial IDs must match across KV/policies; later free histories may diverge.\n\n')
        text+='Limits: 512 tokens/request; 8 model attempts; 6 parsed tool/final attempts; 4096 output tokens; one retry; request180s/episode900s; client context24576; server context28672.\n\n'
        text+=('비용표의 출력 token은 반환된 원 token IDs의 개수이며 usage와 대조했습니다. EOS 등 제어 token도 원 기록에 보존합니다. '
               'episode wall에는 로컬 tokenization·도구·통신·실패가 포함됩니다. worker wall에는 서버 로드·컴파일·종료도 포함하며, warmup forward를 별도 API 생성 요청으로 세지 않습니다.\n\n' if ko else
               'Output-token cost counts returned raw token IDs and is checked against usage; EOS/control IDs are retained. '
               'Episode wall includes local tokenization, tools, transport, and failures. Worker wall additionally includes server load, compile, and shutdown; warmup forwards are not additional API generation requests.\n\n')
        text+=('각 generation 직전의 관측 사건 ID를 고정합니다. 정답 경로는 read 결과를 받은 뒤 계산·수정하고, 최신 validation을 받은 뒤 final을 생성해야 합니다. '
               'validation은 공개 불변식만 검사하며 hidden gold를 알려주지 않습니다. 최종 정답은 별도 CPU oracle가 판정합니다.\n\n' if ko else
               'Pre-generation observation IDs are immutable. Dependent calculations/updates require observed reads; final requires the latest observed validation. '
               'Validation checks public invariants, never hidden gold. A separate CPU oracle evaluates the endpoint.\n\n')
        text+='## <a id="results"></a>'+('실험 결과' if ko else 'Experimental results')+'\n\n'
        for s,summary in stages.items():
            text+=f'### {s} — {summary["status"]}\n\n'+policy_table(summary)+'\n\n'+cost_table(summary)+'\n\n'
            text+=f'![{s} verified success](figures/{s.lower()}_success.svg)\n\n'
            fam={(g['kv'],g['policy'],g['family']):g for g in summary['family_groups']}
            text+=table(['KV','Policy','A: totals','B: transfer','C: metadata'],
                [[kv,p]+[f"{fam[kv,p,f]['verified_success']}/{fam[kv,p,f]['observed']}" for f in ('A','B','C')]
                 for kv in ('K0','K1') for p in ('C0','G','N','E')])+'\n\n'
            text+=pair_table(summary)+'\n\n'
        text+=terminal+'\n\n'
        text+='## <a id="analysis"></a>'+('분석과 불확실성' if ko else 'Analysis and uncertainty')+'\n\n'
        text+=('업무 실패·계약 거절·실행 미관측을 구분했습니다. 위 구간은 family별 task를 함께 재표집한 10,000회 탐색적 pointwise 구간(seed6102013)입니다. '
               '같은 task의 8조건을 같이 유지합니다. 합성 24과제는 독립 회사나 일반 업무 모집단이 아닙니다. 폭0인 구간도 동등성 인증이 아닙니다. '
               'E와 N은 독립 자유 실행 정책이므로 차이를 오류 설명 정보만의 인과효과로 해석하지 않습니다. '
               'KV 변경에는 커널 경로 차이가 포함될 수 있습니다.\n\n' if ko else
               'Functional failures, contract rejections, and unobserved infrastructure outcomes are separate. Intervals use 10,000 family-stratified paired task draws (seed6102013), preserving all eight cells. '
               'These synthetic tasks are not independent companies or an external business population. Zero-width intervals do not establish equivalence. '
               'N/E are independent free policies, not an isolated feedback-information experiment. KV changes may include kernel-path differences.\n\n')
        text+='\n'.join(examples)+'\n\n'
        if 'P2' in stages:
            text+=('대표 원행: [HOLDOUT-A00 / NVFP4 C0](results/p2_r1/cells/HOLDOUT-A00_K1_C0.json)는 첫 read JSON 뒤에 `}`를 하나 더 생성해 실행되지 않았습니다. '
                   '[같은 과제의 G](results/p2_r1/cells/HOLDOUT-A00_K1_G.json)는 read→계산→저장→검증→final을 실제 완료했습니다. '
                   '이는 이번 조건에서 형식 병목이 완화된 사례입니다.\n\n'
                   '[HOLDOUT-B01 / BF16 G](results/p2_r1/cells/HOLDOUT-B01_K0_G.json)는 이동 전 검증을 받은 뒤 재고를 옮기고, 새 상태 검증 없이 final을 냈습니다. '
                   '파일과 최종 숫자는 맞지만 관측 계약을 충족하지 못했습니다. '
                   '[HOLDOUT-C00 / NVFP4 G](results/p2_r1/cells/HOLDOUT-C00_K1_G.json)는 요청된 `review` 대신 schema상 허용되는 `published`를 저장했습니다. '
                   'G는 올바른 상태값을 알려주지 않으므로 이 의미 오류는 그대로 실패입니다.\n\n' if ko else
                   'Example: [HOLDOUT-A00 / NVFP4 C0](results/p2_r1/cells/HOLDOUT-A00_K1_C0.json) generated an extra closing brace after its first read JSON, so nothing executed. '
                   '[G on that task](results/p2_r1/cells/HOLDOUT-A00_K1_G.json) actually completed read→calculate→write→validate→final. '
                   'This is a conditional serialization-bottleneck recovery.\n\n'
                   '[HOLDOUT-B01 / BF16 G](results/p2_r1/cells/HOLDOUT-B01_K0_G.json) validated before transfer and completed without observing validation of the changed state. '
                   'Its file and final numbers were correct but its observation contract failed. '
                   '[HOLDOUT-C00 / NVFP4 G](results/p2_r1/cells/HOLDOUT-C00_K1_G.json) stored schema-valid `published` instead of requested `review`. '
                   'G does not supply the correct status; this semantic error remains a failure.\n\n')
        text+='## <a id="conclusion"></a>'+('결론과 범위' if ko else 'Conclusion and scope')+'\n\n'
        if 'P2' in stages:
            text+=('정확한 유지·손실·회복 수는 위 paired 표에 있습니다. 문법에 맞는 출력은 실제 상태·인자·검증 관측까지 맞다는 보장이 아닙니다. '
                   '비용은 실패를 포함한 episode wall 합계와 생성 token입니다. 이 측정으로 serving speedup이나 동시 처리량을 주장하지 않습니다.\n\n' if ko else
                   'The paired table gives exact preservation, loss, and recovery counts. Syntactic validity does not guarantee correct values, state, or observed validation. '
                   'Costs sum episode walls and generated tokens including failures; this is not a serving-throughput or speedup benchmark.\n\n')
        text+=table(['Scope','Status'],[['P0 native cache / grammar','Measured; see P0 gates and kernel/finite receipts'],['P1 DEV / P2 HOLDOUT','Separate measured denominators and stored-trace audits'],['P3 COST / concurrent serving','NOT_RUN'],['Weight INT4 / training / other models','NOT_RUN'],['Git publication / Pages','NOT_RUN; separate approval required']])+'\n\n'
        text+=('Q1: 문법 제약은 NVFP4 조건에서 실제 완주를 늘렸지만 BF16의 완주수는 늘리지 않았습니다. '
               '형식 제약으로 상태값·최신 검증까지 보장하지 못했습니다. E는 N보다 점추정이 높지만 탐색 구간에0이 있고 독립 자유 정책 비교입니다. '
               'Q2: 같은 BF16 가중치여도 두 serving 구성의 결과는 달랐습니다. G에서 차이가 작아졌지만 비열등성·동등성이나 일반 업무 우위를 인증하지 않습니다. '
               '다음 실험은 별도 승인 후 동일 실패 관측의 N/E 진단이나 상태 변경 뒤 검증 계획을 다루어야 합니다. 이번에는 추가 실행하지 않았습니다.\n\n' if ko else
               'Q1: Grammar increased actual completions on NVFP4 but not BF16, and did not guarantee semantic values or current-state validation. '
               'E has a higher point estimate than N, with an interval including zero and independent free-policy histories. '
               'Q2: The serving configurations differ despite identical BF16 weights. G narrows the observed gap without establishing equivalence, noninferiority, or general business superiority. '
               'Future approved work could diagnose N/E at shared failure observations or post-update validation planning; neither is run here.\n\n')
        text+='## <a id="reproduce"></a>'+('재현과 참고자료' if ko else 'Reproduction and references')+'\n\n'
        text+='[CPU/GPU reproduction](REPRODUCE.md) · [Protocol](specs/protocol.json) · [Reuse map](REUSE_MAP.md) · [Notice](NOTICE.md) · [Roadmap](ROADMAP.md)\n\n'
        text+='[P0 source freeze](provenance/execution_freeze_p0_r2.json) · [P1 source freeze](provenance/execution_freeze_p1.json) · [Model identity](provenance/model_identity.json)\n\n'
        text+=('Codex 보조로 구현·문서화했고 별도 CPU 계산 코드로 검산했습니다. 이를 사람의 독립 심사나 외부 GPU 재현으로 표시하지 않습니다. '
               '공개 CPU 감사는 저장된 원문·SSE·token ID·사건을 재생합니다. 모델 가중치 없이 새 모델 추론을 재현한 것은 아닙니다.\n' if ko else
               'Implementation and documentation used Codex assistance; a separate CPU calculation checks the endpoint. This is not human peer review or external GPU replication. '
               'CPU auditing replays saved text, SSE, token IDs, and events; it does not regenerate model outputs without weights.\n')
        (output/('REPORT.ko.md' if ko else 'REPORT.md')).write_text(text,encoding='utf-8')
        s=stages['P2'];gs={(g['kv'],g['policy']):g for g in s['groups']}
        c=next(c for c in s['comparisons'] if c['comparison_id']=='K1:G-C0')
        overview=f'# {title}\n\n{intro}\n\n'+policy_table(stages['P2'])+'\n\n'
        overview+=(f"분모는 새 합성 HOLDOUT{s['registered_tasks']}입니다. NVFP4에서 문법 제약은 자유 생성보다 "
                   f"{c['counts'].get('recovery',0)}개를 더 완주했고 손실은 {c['counts'].get('loss',0)}개였습니다. "
                   f"BF16의 완주수는 네 정책 모두 {gs['K0','C0']['verified_success']}/{s['registered_tasks']}로 같았습니다. "
                   '문법 제약은 잘못된 상태값이나 오래된 검증으로 완료하는 행동까지 고치지는 못했습니다.\n\n' if ko else
                   f"Denominator: {s['registered_tasks']} new synthetic HOLDOUT tasks. On NVFP4, grammar recovered "
                   f"{c['counts'].get('recovery',0)} free-generation failures with {c['counts'].get('loss',0)} losses. "
                   f"All BF16 policies completed {gs['K0','C0']['verified_success']}/{s['registered_tasks']}. "
                   'Grammar did not eliminate wrong semantic values or stale-validation completion.\n\n')
        overview+=('A: 여러 상품의 개수와 한 개 가격으로 할인 적용 합계를 계산·저장합니다. B: 두 창고의 재고를 읽고 물건을 원자적으로 옮깁니다. '
                   'C: 문서의 현재 수정 번호를 읽고 상태·태그·담당자를 바꿉니다. 전부 소유한 가상 JSON 기록입니다.\n\n' if ko else
                   'A calculates and stores discounted totals from item counts and prices. B reads two stock records and transfers units atomically. '
                   'C reads a document revision and updates status, tags, and owner. All data are owned synthetic JSON records.\n\n')
        overview+=('직접 구현: 엄격 JSON 계약·4정책 실행기, 안전 fixture·관측 추적, 독립 endpoint 감사·비교·한영 탐색기. '
                   '엔진과 NVFP4 커널은 vLLM/FlashInfer upstream입니다.\n\n' if ko else
                   'Implemented here: strict JSON contracts, four-policy runner, safe fixtures and observation tracking, independent endpoint audit, comparison tools, and bilingual explorer. '
                   'Engine and NVFP4 kernels come from vLLM/FlashInfer upstream.\n\n')
        overview+='[Report](REPORT.ko.md) · [English report](REPORT.md) · [한국어 탐색기](demo/ko.html) · [English explorer](demo/en.html) · [CPU/GPU reproduction](REPRODUCE.md)\n\n'
        overview+='```bash\nPYTHONPATH=src python3 -B -m agent_contract_cost.verify --output /tmp/case017-review-new-output\n```\n\n'
        overview+=('출력은 아직 없는 새 경로를 사용합니다. 이 명령은 저장 응답을 재생하며 모델을 호출하지 않습니다.\n\n' if ko else
                   'Use a new, nonexistent output path. This command replays saved responses without calling a model.\n\n')
        overview+='### DEV / P1\n\n'+policy_table(stages['P1'])+'\n\n'
        overview+=f"P0 {p0['actual_model_request_attempts']} requests; P1 {sum(g['model_request_attempts'] for g in stages['P1']['groups'])}; P2 {sum(g['model_request_attempts'] for g in stages['P2']['groups'])}. Total {total_requests} API generation attempts, {total_tokens} output tokens; worker wall {total_worker:.3f}s.\n\n"
        overview+=('K0는 BF16 KV, K1은 native NVFP4 KV입니다. **가중치는 두 조건 모두 BF16**입니다. G는 공개 JSON schema로 생성 token을 제한하고, '
                   'N/E는 최초 계약 실패에만 한 번 재시도합니다. 같은 약4GiB pool의 저장 슬롯 비율3.556배는 전체 VRAM 절감이나 이번 업무의 속도 향상이 아닙니다. '
                   'P3·가중치 INT4·학습·동시 serving·원격 게시·Pages는 미실행입니다.\n\n' if ko else
                   'K0 is BF16 KV; K1 is native NVFP4 KV. **Weights are BF16 in both.** G constrains generated tokens with a public JSON schema; '
                   'N/E allow one retry at the first contract rejection. The3.556× token-slot ratio in an approximately4GiB pool is neither total VRAM reduction nor task speedup. '
                   'P3, weight INT4, training, concurrent serving, remote publication, and Pages are not run.\n\n')
        overview+='[Protocol](PROTOCOL.md) · [Reuse](REUSE_MAP.md) · [Notice](NOTICE.md) · [Roadmap](ROADMAP.md) · [Source freeze](provenance/execution_freeze_p2.json)\n'
        (output/('README.ko.md' if ko else 'README.md')).write_text(overview,encoding='utf-8')
    return {'stages':list(stages),'model_requests':total_requests,'output_tokens':total_tokens,'worker_wall_seconds':total_worker}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args(argv)
    print(json.dumps(build(a.output),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
