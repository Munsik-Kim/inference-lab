"""Display new measured serving/quality records without changing old cases."""
from html import escape
from pathlib import Path
import json
import statistics
from common import require, sha, read

C9='cases/009-q-serving-quality'
SOURCE_COPIES = {
    'diova-compare-core.py.txt':'packages/diova-compare/src/diova_compare/core.py',
    'diova-compare-tests.py.txt':'packages/diova-compare/tests/test_compare.py',
    'diova-compare-edge-tests.py.txt':'packages/diova-compare/tests/test_review_edges.py',
    'case009-serving.py.txt':C9+'/scripts/serving.py',
    'case009-quality.py.txt':C9+'/scripts/quality.py',
    'case009-report-en.md.txt':C9+'/REPORT.md',
    'case009-report-ko.md.txt':C9+'/REPORT.ko.md',
    'case009-notice.md.txt':C9+'/NOTICE.md',
}


def load_case009(root):
    c=root/C9
    paths=['results/derived/serving_summary.json','results/derived/quality_summary.json',
           'configs/serving_protocol.json','configs/quality_input_freeze.json','figures/serving-L128.svg','figures/serving-L1024.svg',
           'REPORT.md','REPORT.ko.md','NOTICE.md','scripts/serving.py','scripts/quality.py']
    if not (c/paths[0]).exists():return None
    require(all((c/p).is_file() and not (c/p).is_symlink() for p in paths),'Incomplete/unsafe Case009 display sources')
    s=read(c/paths[0]);q=read(c/paths[1])
    require(s['planned_cells']==60 and s['complete_cells']<=60,'Unexpected serving design')
    require([x['task'] for x in q['tasks']]==['arc_challenge','wikitext','gsm8k','mmlu'],'Quality task scope')
    sources={C9+'/'+p:sha(c/p) for p in paths}
    for p in ['packages/diova-compare/src/diova_compare/core.py','packages/diova-compare/tests/test_compare.py','packages/diova-compare/tests/test_review_edges.py']:
        sources[p]=sha(root/p)
    extra={}
    for key,path in [('posthoc','publication/posthoc/analysis.json'),('lifecycle','supplemental/lifecycle-v1/summary.json')]:
        if (c/path).exists():extra[key]=read(c/path);sources[C9+'/'+path]=sha(c/path)
    return {'evidence_kind':'NEW_MEASUREMENT','publication':'REVIEWED_FEATURE_BRANCH_CANDIDATE','source_identity':'reviewed Case009 snapshot; original archive and current file hashes, publication receipt separate',
      'sources':sources,'serving':s,'quality':q,**extra}


def table(head,rows):
    return '<div class="scroll" tabindex="0"><table><thead><tr>'+''.join('<th scope="col">'+escape(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def render_case009(data,lang):
    ko=lang=='ko';s=data['serving'];q=data['quality']
    title='긴 decode와 동시 요청에서 BF16·W4 비교' if ko else 'BF16 and W4 under longer decode and concurrent requests'
    h='<h1>'+title+'</h1><p>'+('기존 Qwen3-4B-Instruct-2507 저장본 두 개를 같은 vLLM 0.29.0 경로로 실행했습니다. CUDA Graph 성능곡선과 별도로 실행한 공식 품질 평가를 함께 봅니다.' if ko else 'Two existing Qwen3-4B-Instruct-2507 artifacts run through matched vLLM 0.29.0 paths. Read the CUDA Graph performance curves alongside separately executed official quality tasks.')+'</p>'
    h+='<p>'+('BF16은 원래 가중치 정밀도의 기준선입니다. W4는 기존 GPTQ 저장본에서 선택된 Linear 가중치를 4bit로 저장한 설정입니다. 두 설정 모두 활성값과 KV cache는 BF16을 사용합니다.' if ko else 'BF16 is the original weight-precision baseline. W4 is the existing GPTQ artifact with selected Linear weights stored at four bits. Both arms use BF16 activations and KV cache.')+'</p>'
    h+='<p class="scope">'+('완료한 비용 측정 조건' if ko else 'Completed serving cells')+f': {s.get("complete_cells",0)} / 60</p>'
    h+='<section class="panel"><h2>'+('조사한 질문과 설계' if ko else 'Question and design')+'</h2><p>'+('짧은 요청보다 decode가 길어지고 동시 요청이 늘면, 가중치 이동 감소가 요청 비용에 어떻게 나타나는지 조사합니다. 작은 Linear의 가중치 전송량은 줄어들 수 있지만 attention·KV·대기열 비용도 함께 발생합니다. 따라서 커널 논문의 속도비를 이 장치의 합격 기준으로 쓰지 않습니다.' if ko else 'The question is how reduced weight traffic changes serving cost as decode length and client concurrency increase. Attention, KV traffic and scheduling still contribute. Kernel-paper speedups are not an acceptance threshold for this device.')+'</p><p>'+('BF16과 W4에 같은 입력 ID·순서·4 GiB KV 예산을 적용하고, 새 서버 프로세스에서 실행 순서를 교차했습니다. 별도의 입력 128·출력 256, 동시성 1/16 eager 비교도 같은 두 모델로 측정합니다. 품질 평가는 공식 전체 split과 추출 규칙을 사용하며 시간 측정에 섞지 않습니다.' if ko else 'Matched IDs, order and a 4 GiB KV budget are used in fresh server processes with alternating arm order. A separate matched eager ablation covers input128/output256 at concurrency1/16. Quality uses official full splits and extraction rules outside the timing runs.')+'</p></section>'
    h+='<nav class="actions" aria-label="Contents">'+''.join(f'<a href="#{i}">{name}</a>' for i,name in [('serving','요청 비용' if ko else 'Serving cost'),('quality','품질 평가' if ko else 'Quality'),('scope','측정 범위' if ko else 'Scope'),('cli','CPU 도구' if ko else 'CPU tool')]+([('posthoc','사후 분석' if ko else 'Post-hoc'),('lifecycle','종료 진단' if ko else 'Lifecycle')] if 'lifecycle' in data else []))+'</nav>'
    h+='<section class="panel" id="serving"><h2>'+('전체 동시성 곡선' if ko else 'Complete concurrency curves')+'</h2>'
    h+='<p>'+('RTX 5080 · 출력 256토큰 · 동시 요청 상한 1/4/16/32 · 독립 서버 라운드 3회. 그래프는 라운드 평균, 막대는 라운드 최솟값–최댓값입니다. 신뢰구간은 아래의 짝지은 속도비에 별도로 표시합니다.' if ko else 'RTX 5080 · 256 output tokens · client concurrency 1/4/16/32 · three independent server rounds. Curves show round means and whiskers show round min–max; paired ratio intervals appear separately below.')+'</p>'
    for length in [128,1024]:
        h+=f'<figure><img class="serving-curve" src="../assets/serving-L{length}.svg" alt="'+('입력 '+str(length)+'토큰의 TPOT와 처리량 전체 곡선' if ko else f'TPOT and throughput across all concurrency values, {length} input tokens')+'"><figcaption>'+str(length)+(' 입력 토큰 · 영어 축 라벨' if ko else ' input tokens')+'</figcaption></figure>'
    rows=[]
    for r in s['contrasts']:
        def form(k):
            if k not in r:return 'INCOMPLETE'
            x=r[k];lo,hi=x['pointwise_95_interval'];return f'{x["mean_ratio"]:.3f} [{lo:.3f}, {hi:.3f}]'
        rows.append([f'{r["execution"]} / L{r["input_tokens"]} / C{r["client_concurrency"]}',form('tpot_ratio'),form('throughput_ratio')])
    h+=table(['조건' if ko else 'Condition','TPOT BF16/W4 [95%]','Tokens/s W4/BF16 [95%]'],rows)+'</section>'
    h+='<section class="panel" id="quality"><h2>'+('공식 품질 평가' if ko else 'Official quality evaluation')+'</h2><p>'+('동일한 입력·few-shot·parser를 사용합니다. benchmark 사이의 점수를 합산하지 않습니다. 선택지 점수의 KL은 전체 어휘 KL과 다릅니다.' if ko else 'Inputs, few-shot examples and parsers are matched. Benchmark scores are not combined. Choice-score KL is distinct from full-vocabulary KL.')+'</p>'
    rows=[]
    for t in q['tasks']:
        if 'groups' not in t:rows.append([t['task'],t['arms']['BF16'],t['arms']['W4']]);continue
        rows.append([t['task']+' · PROCESS EXIT',t['process_status']['BF16'],t['process_status']['W4']])
        if t['task']=='wikitext':
            a=next(iter(t['groups'].values()))['arms'];rows.append(['WikiText-2 word perplexity',f'{a["BF16"]["word_perplexity"]:.4f}',f'{a["W4"]["word_perplexity"]:.4f}']);continue
        groups=[('MMLU acc',t['overall_paired'])] if t['task']=='mmlu' else list(t['groups'].items())
        for name,r in groups:
            rows.append([name,*[f'{r[k]}/{r["n"]} ({100*r[k]/r["n"]:.2f}%)' for k in ['baseline_correct','candidate_correct']]])
            if 'acc_norm' in r:
                a=r['acc_norm'];rows.append(['ARC acc_norm',*[f'{a[k]}/{a["n"]} ({100*a[k]/a["n"]:.2f}%)' for k in ['baseline_correct','candidate_correct']]])
    h+=table(['Task / metric','BF16','W4'],rows)
    for t in q['tasks']:
        if t['task']=='gsm8k' and 'generation_budget' in t:
            b=t['generation_budget']['BF16'];w=t['generation_budget']['W4']
            count=f"BF16 {b['at_1024_token_cap']}/{b['n']} · W4 {w['at_1024_token_cap']}/{w['n']}"
            h+='<p>'+('GSM8K의 1,024토큰 상한 도달: ' if ko else 'GSM8K outputs reaching the 1,024-token cap: ')+escape(count)+('. 해당 출력도 점수에 포함하며, 두 추출 규칙은 같은 출력의 반복된 관점입니다.' if ko else '. These outputs remain in the scores. Both extraction filters view the same outputs.')+'</p>'
    if any(t['status']=='COMPUTED_WITH_PROCESS_FAILURE' for t in q['tasks']):
        h+='<p class="scope">'+('FAILED는 계산·파일 저장 뒤의 비정상 프로세스 종료를 뜻합니다. 표의 수치는 보존한 전체 문항 계산을 검산한 값이며, 해당 실행을 정상 완료로 바꾸지는 않습니다. 실패한 split은 재실행하지 않았습니다.' if ko else 'FAILED marks abnormal process exit after calculation and file writes. Reported values audit retained full-split calculations; they do not turn a failed execution into a successful one. Failed splits were not repeated.')+'</p>'
    h+='<p>'+('문항별 전환, 과제별 구간, MMLU 과목 결과, 생성 길이 제한은 아래 JSON에 보존합니다.' if ko else 'The JSON retains item-transition counts, task intervals, MMLU subjects and generation-cap records.')+'</p></section>'
    if 'posthoc' in data:
        d=data['posthoc'];m=d['mmlu'];w=d['wikitext'];g=d['gsm8k']['filter_views'];v=d['serving']['means_of_round_metrics']
        h+='<section class="panel" id="posthoc"><h2>'+('같은 기록의 사후 분석' if ko else 'Post-hoc analysis of the same records')+'</h2><p>POST_HOC_SAME_RECORDED_SCALARS</p>'
        rows=[['MMLU · '+('하락한 과목' if ko else 'subjects declining'),f"{m['subjects_declining']}/{m['n_subjects']}"],['MMLU · D '+('선택 비중' if ko else 'choice share'),f"{100*m['D_choice_counts']['BF16']/m['n_items']:.2f}% → {100*m['D_choice_counts']['W4']/m['n_items']:.2f}% (n={m['n_items']})"],['WikiText · '+('로그우도 하락 문서' if ko else 'documents with lower likelihood'),f"{w['lower_candidate_loglikelihood']}/{w['n_documents']}"],['WikiText · word PPL',f"+{100*w['word_PPL_relative_increase']:.2f}%"],['GSM8K · '+('평균 생성 토큰' if ko else 'mean generated tokens'),f"{g['BF16']['mean_generated_tokens']:.2f} → {g['W4']['mean_generated_tokens']:.2f}"],['L128/C1 · TTFT p50',f"{v['BF16']['L128-C1']['ttft_p50_ms']:.3f} → {v['W4']['L128-C1']['ttft_p50_ms']:.3f} ms"]]
        h+=table(['관측' if ko else 'Observation','기록' if ko else 'Record'],rows)
        h+='<p>'+('정답 위치 변화는 원인 규명이 아닙니다. GSM 길이비는 실제 실행 속도비가 아닙니다. 원래 과목별 구간과 graph·compile·fusion 묶음 비교는 그대로 유지합니다.' if ko else 'Position changes do not identify a cause. GSM length ratios are not runtime speedups. Original subject-cluster intervals and the graph/compile/fusion comparison bundle remain unchanged.')+'</p></section>'
    if 'lifecycle' in data:
        l=data['lifecycle'];h+='<section class="panel" id="lifecycle"><h2>'+('별도 짧은 종료 진단' if ko else 'Separate short lifecycle probes')+'</h2><p>'+escape(l['status'])+'</p>'
        h+='<p>'+escape(f"{l['clean']}/{l['attempts']} clean; {l['failed']} failed; {l['not_run']} budget cells not run; old full-split retries = {l['original_full_split_retries']}")+'</p><p>'+('D0는 원래 shutdown 흐름, D1은 객체 삭제·gc 시점을 추가로 비교했습니다. 두 설정의 작은 synthetic probe에서 과거 실패가 재현되지 않았습니다. 위 품질표의 원래 종료 실패를 수정한 결과는 아닙니다.' if ko else 'D0 retains the original shutdown flow; D1 additionally controls deletion and garbage collection. Small synthetic probes did not reproduce the historical failure. Original full-task exits in the quality table remain failed.')+'</p></section>'
    h+='<section class="panel" id="cli"><h2>diova-compare 0.1.1</h2><p>'+('같은 과제 안의 지표 누락을 거절하고 극소 양수 확률의 KL을 안정적으로 계산합니다. Torch·GPU 없이 공개 기록을 비교합니다.' if ko else 'Rejects mixed metric coverage within a task and computes tiny-positive-probability KL stably. Compare public records without Torch or a GPU.')+'</p><a href="../sources/diova-compare-core.py.txt">'+('수정 코드' if ko else 'Fixed source')+'</a> · <a href="../sources/diova-compare-edge-tests.py.txt">'+('회귀 테스트' if ko else 'Regression tests')+'</a></section>'
    h+='<section class="panel" id="scope"><h2>'+('측정 범위와 근거' if ko else 'Scope and evidence')+'</h2><p>'+('4 GiB의 같은 KV 예산을 사용했습니다. 긴 입력에서 동시성 32가 실제 active sequence 32를 뜻하지는 않습니다. 메모리는 desktop·startup·warmup을 포함한 1 Hz 장치 표본이며 Torch peak가 아닙니다. 서버 라운드가 3회뿐이고 WSL의 배경 작업이 있었습니다. 기존 Case 008의 8토큰 eager 측정은 그대로 남습니다.' if ko else 'Both arms use a 4 GiB KV budget. Client concurrency 32 need not mean 32 active sequences with long inputs. Memory is 1 Hz whole-device sampling, including desktop, startup and warmup, not a Torch peak. Only three process rounds and a WSL host with background work were observed. Case 008’s eight-token eager measurements remain unchanged.')+'</p><p>'+('배포 적합성: NOT_ASSESSED. 이 화면은 로컬 검토본의 저장된 측정입니다.' if ko else 'Deployment: NOT_ASSESSED. This is recorded evidence in a local review candidate.')+'</p><a href="../data/case009.json" download="case009.json">'+('측정·구간·출처 JSON 다운로드' if ko else 'Download measurements, intervals and source hashes')+'</a></section>'
    h+='<div class="actions">'+f'<a href="../sources/case009-report-{lang}.md.txt" download="REPORT.{lang}.md">'+('정식 보고서 다운로드' if ko else 'Download the full report')+'</a><a href="../sources/case009-serving.py.txt">'+('요청 측정 소스·영어' if ko else 'Serving runner source')+'</a><a href="../sources/case009-quality.py.txt">'+('공식 평가 연결 소스·영어' if ko else 'Official evaluation adapter source')+'</a></div>'
    return h
