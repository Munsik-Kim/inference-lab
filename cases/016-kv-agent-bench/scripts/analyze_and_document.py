#!/usr/bin/env python3
"""Raw-derived diagnostics and bilingual reports; never changes execution rows."""
import argparse
import collections
import csv
import json
import math
import pathlib
import statistics
import sys

CASE=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(CASE/'src'))
from kv_agent_bench.common import load,save,sha_file

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(v)for v in r)+' |\n'for r in rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=pathlib.Path,default=CASE);args=p.parse_args();case=args.case.resolve();root=case/'results/p1_r1';s=load(root/'summary.json');eps=[load(f)for f in sorted((root/'episodes').glob('*.json'))];tasks=load(case/'data/tasks.json')
    if len(eps)!=96 or s['status']!='COMPLETED_P1':raise ValueError('full report requires all 96 observed cells; preserve partial ledger instead')
    if any(e['availability']!='OBSERVED'for e in eps):raise ValueError('unobserved cells must not be imputed')
    audit=load(root/'independent_audit.json')
    if audit['status']!='PASS' or audit['episodes']!=96:raise ValueError('independent raw audit must finish before documentation')
    counts={kv:sum(e['VERIFIED_WORKFLOW_SUCCESS']is True for e in eps if e['kv']==kv)for kv in ('B16','F4')}
    diagnostics={}
    for kv in ('B16','F4'):
        e=[x for x in eps if x['kv']==kv];responses=[r for x in e for r in x['responses']];events=[v for x in e for v in x['events']]
        diagnostics[kv]={'episodes':48,'verified':counts[kv],'state_correct':sum(x['STATE_CORRECT']for x in e),'final_correct':sum(x['FINAL_ANSWER_CORRECT']for x in e),'claimed_done_with_wrong_state':sum(x['CLAIMED_DONE_WITH_WRONG_STATE']for x in e),
          'terminals':dict(collections.Counter(x['terminal_reason']for x in e)),'first_unmet':dict(collections.Counter(x['first_unmet_obligation']for x in e)),
          'event_errors':dict(collections.Counter(v['error']for v in events if v['error'])),'model_calls':len(responses),'tool_attempts':sum(x['tool_attempts']for x in e),'tool_events':len(events),
          'output_tokens':sum(r['output_tokens']for r in responses),'input_tokens':sum(len(r['prompt_token_ids'])for r in responses),
          'generated_tool_slots':sum(len(r['parsed_slots'])for r in responses),'admitted_tool_attempts':sum(x['tool_attempts']for x in e),'successful_callable_events':sum(v['status']=='SUCCESS'for v in events),'rejected_registry_or_argument_events':sum(v['status']!='SUCCESS'for v in events),
          'output_length_min_median_max':[min(r['output_tokens']for r in responses),statistics.median(r['output_tokens']for r in responses),max(r['output_tokens']for r in responses)],
          'request_length_finish_count':sum(r['server_finish_reason']=='length'for r in responses),'eos_observed_count':sum(r['eos_observed']for r in responses),
          'parse_stop_lengths':[x['responses'][-1]['output_tokens']for x in e if x['terminal_reason']=='PARSE_STOP'],
          'max_actual_prompt_tokens':max(len(r['prompt_token_ids'])for r in responses),
          'same_initial_hash_groups':12*2,'post_rollout_identical_observation_claim':False}
    family=[]
    for kv in ('B16','F4'):
        for menu in (8,32):
            for cap in (256,1024):
                for fam in 'ABCD':
                    e=[x for x in eps if(x['kv'],x['menu'],x['cap'],x['family'])==(kv,menu,cap,fam)]
                    family.append({'kv':kv,'menu':menu,'cap':cap,'family':fam,'n':len(e),'verified':sum(x['VERIFIED_WORKFLOW_SUCCESS']for x in e),'state':sum(x['STATE_CORRECT']for x in e)})
    save(root/'diagnostics.json',{'by_kv':diagnostics,'family':family,'successful_unique_dev_tasks':{kv:sorted({e['task_id'] for e in eps if e['kv']==kv and e['VERIFIED_WORKFLOW_SUCCESS']is True}) for kv in ('B16','F4')},'same_prompt_hash_verified_by_auditor':True,'source_summary_sha256':sha_file(root/'summary.json')})
    with (root/'table.csv').open('w',newline='')as f:
        fields=['kv','menu','cap','planned','completed','observed','verified','state','final','false_done','tokens','model_calls','tool_attempts'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:g[k]for k in fields}for g in s['groups'])
    with (root/'paired.csv').open('w',newline='')as f:
        fields=list(s['paired'][0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(s['paired'])
    cache={kv:load(case/f'results/p0_diagnostics_r3/{kv}_final_config_cache_after.json')for kv in ('B16','F8','F4')}
    measured={kv:load(root/f'{kv}_cache_after.json')for kv in ('B16','F4')}
    p0eps=[load(f)for run in ('p0_r2','p0_r5')for f in(case/f'results/{run}/episodes').glob('*.json')]
    attempts=[]
    for file in sorted((case/'results').glob('p0_r*/*_attempt*.json')):
        x=load(file)
        if 'kv'in x:attempts.append(x)
    completed_probes=[load(f)for pattern in ('*_raw_on.json','*_raw_off.json','*_incomplete_raw.json')for f in(case/'results/p0_diagnostics_r1').glob(pattern)]
    completed_probes += [load(f)for f in(case/'results/p0_diagnostics_r2').glob('*_profile_request.json')]
    cancelled_probes=[load(f)for run in ('p0_diagnostics_r2','p0_diagnostics_r3')for f in(case/'results'/run).glob('*_cancellation.json')]
    rejected_http=[load(f)for f in(case/'results/p0_diagnostics_r1').glob('*_failure.json')if'400 Bad Request'in load(f)['message']]
    save(case/'results/p0_overview.json',{'status':'TECHNICAL_SUPPORT_PASS','workflow_smoke':[{'kv':kv,'n':8,'verified':sum(e['VERIFIED_WORKFLOW_SUCCESS']is True for e in p0eps if e['kv']==kv)}for kv in ('B16','F8','F4')],
      'model_calls_smoke':sum(e['model_calls']for e in p0eps),'tool_attempts_smoke':sum(e['tool_attempts']for e in p0eps),'tool_events_smoke':sum(e['tool_events']for e in p0eps),'tokens_smoke':sum(e['output_tokens']for e in p0eps),
      'complete_instrumentation_requests':len(completed_probes),'instrumentation_complete_tokens':sum(r['usage']['completion_tokens']for r in completed_probes),'cancelled_generation_requests':len(cancelled_probes),'rejected_http_requests':len(rejected_http),'cancelled_tokens':'only received raw IDs observed; total engine output before abort is not exactly known',
      'all_accepted_user_generation_requests':sum(e['model_calls']for e in p0eps)+len(completed_probes)+len(cancelled_probes),'internal_engine_warmup_forwards':'not counted as user requests; load/compile wall included',
      'measured_server_attempt_wall_sum_seconds':sum(x['attempt_wall_seconds']for x in attempts),'attempts':attempts,
      'conservative_p0_budget_elapsed_before_p1':load(case/'results/p0_gates.json')['p0_budget_conservative_elapsed_seconds'],'timing_scope':'P0 conservative clock includes preparation pauses; not GPU busy. Per-server attempt sum and instrumentation clocks are available separately.'})
    group_rows=[[g['kv'],g['menu'],g['cap'],f'{g["verified"]}/12',f'{g["state"]}/12',f'{g["final"]}/12',g['false_done']]for g in s['groups']]
    paired_rows=[[x['comparison'],x['both_success'],x['loss'],x['recovery'],x['both_fail']]for x in s['paired']]
    cache_rows=[[k,{'B16':'BF16','F8':'FP8 E4M3','F4':'NVFP4 packed + FP8 block scales'}[k],f'{v["actual_unique_storage_bytes"]:,}',f'{v["cache_config"]["kv_cache_size_tokens"]:,}',36,28672]for k,v in cache.items()]
    cost_rows=[[k,diagnostics[k]['model_calls'],diagnostics[k]['tool_attempts'],diagnostics[k]['tool_events'],diagnostics[k]['output_tokens'],f'{s["attempts"][i]["attempt_wall_seconds"]:.3f}',f'{measured[k]["peak_allocated"]/2**30:.3f}',f'{measured[k]["peak_reserved"]/2**30:.3f}']for i,k in enumerate(('B16','F4'))]
    end_rows=[[k,n]for k,n in collections.Counter(e['terminal_reason']for e in eps).items()]
    unmet_rows=[[k,n]for k,n in collections.Counter(e['first_unmet_obligation']or'none'for e in eps).items()]
    baseline=next(g for g in s['groups']if(g['kv'],g['menu'],g['cap'])==('B16',8,1024));baseline_text='PASS'if s['baseline_usable'] else'NOT_MET'
    source=load(root/'code_freeze.json')['source_sha256'];main_title='Case016 — 작은 KV 캐시로 더 많은 업무를 정확히 끝낼 수 있을까?';en_title='Case016 — Can a Smaller KV Cache Complete More Jobs Correctly?'
    for ko in (True,False):
        language='ko'if ko else'en';suffix='.ko.md'if ko else'.md';title=main_title if ko else en_title
        intro=(f'실제 압축 KV를 사용하는 vLLM과 장문 업무 실행·감사 도구를 구현했습니다. 고정한 DEV 96개 조건에서 B16은 {counts["B16"]}/48, NVFP4는 {counts["F4"]}/48의 검증된 업무 성공을 기록했습니다. 같은 12개 과제를 메뉴·응답 한도 4개 조건으로 반복한 결과입니다. 기준선 확대 gate는 {baseline_text}입니다. 이번 범위 밖의 P2/P3는 미실행입니다.'if ko else f'Implemented real compressed-KV vLLM execution and a long-context workflow harness/auditor. On the frozen 96 DEV conditions, B16 completed {counts["B16"]}/48 verified workflows and NVFP4 {counts["F4"]}/48. These reuse 12 fixtures under 4 menu/cap combinations. The baseline expansion gate is {baseline_text}. P2/P3 are outside this run and are not executed.')
        readme=f'# {title}\n\n'+('여러 자료를 읽어 필요한 버전을 고르고, 금액 계산·기록 수정·검증까지 맡기는 합성 업무입니다.\n\n' if ko else 'A synthetic job reads several sources, chooses the required version, calculates an amount, updates a record and validates it.\n\n')+intro+'\n\n'+('먼저 알아둘 말: KV 캐시는 모델이 읽은 내용의 중간 계산 결과를 저장하는 공간입니다. 가중치는 모든 조건에서 BF16이고 KV 저장 형식만 비교합니다. 금액은 소수점 없는 최소단위 정수입니다. 예를 들어 개수×한 개 가격에 정수 할인 규칙을 적용합니다.\n\n'if ko else'A KV cache stores intermediate attention results for text already read. All model weights stay BF16; only KV storage changes. Money uses integer minor units: quantity × unit price with a stated integer discount rule.\n\n')
        readme+=('실제 구현: 긴 근거와 도구 메뉴 생성 → raw SSE/token 관측 → 격리된 도구 실행 → 상태·관측 순서·최종 답 채점 → 새 fixture에서 CPU 재생.\n\n'if ko else'Implemented: long evidence/tool menus → raw SSE/token capture → isolated tool execution → state/observation/final-answer scoring → CPU replay on fresh fixtures.\n\n')
        readme+=table(['KV','menu','cap','workflow','state','final','false done'],group_rows)+'\n'+('[결과 탐색기](demo/ko.html) · [상세 보고서](REPORT.ko.md) · [CPU 확인](REPRODUCE.ko.md) · [실행 코드](src/kv_agent_bench/episode_runner.py) · [원자료](results/p1_r1/)\n'if ko else'[Result explorer](demo/en.html) · [Full report](REPORT.md) · [CPU reproduction](REPRODUCE.md) · [Runner](src/kv_agent_bench/episode_runner.py) · [Raw records](results/p1_r1/)\n')
        readme+=('\n각 칸은 관측된12개 과제입니다. 파일 상태가 맞는 것과 관측 순서·최신 검증을 포함한 업무 완주는 다른 지표입니다. 로컬 검토본이며 GitHub/Pages에 게시하지 않았습니다.\n'if ko else'\nEach cell contains12 observed fixtures. Correct file state and a complete workflow with current observed validation are separate endpoints. This is a local review, not a GitHub/Pages publication.\n')
        (case/('README.ko.md'if ko else'README.md')).write_text(readme)
        toc=[('overview','한눈에 보기'if ko else'Overview'),('business','실제 업무'if ko else'The job'),('cache','KV와 기준선'if ko else'KV and baselines'),('experiment','실험'if ko else'Experiment'),('results','성공과 비용'if ko else'Success and cost'),('events','실패 사건'if ko else'Failure events'),('reproduce','직접 실행'if ko else'Reproduce'),('sources','상세 출처'if ko else'Sources')]
        body=f'# {title}\n\n{intro}\n\n'+ ' · '.join(f'[{label}](#{anchor})'for anchor,label in toc)+'\n\n'
        def section(i):return f'<a id="{toc[i][0]}"></a>\n\n## {toc[i][1]}\n\n'
        body+=section(0)+('이번 질문은 작은 KV가 장문 업무 완주에 어떤 영향을 주는지입니다. 품질이 좋아야 실험 완료인 것은 아닙니다. 모든 96개 조건은 관측 완료이며, 원기록과 실패를 유지합니다. 후속48개 EVAL은 생성·CPU 정답경로 검사만 했고 모델로 열지 않았습니다.\n\n'if ko else'The question is how smaller KV affects completion of long-context jobs. Completion of the experiment does not require a positive quality result. All 96 conditions were observed, with failures retained. Future 48 EVAL fixtures were generated and CPU-oracle-checked, never model-evaluated.\n\n')
        body+=section(1)+('네 종류는 구매·견적, 최신 버전, 지연 도착, 수정·취소입니다. 여러 자료에서 규칙에 맞는 자료를 고르고, 해당 자료 확인 → 금액 계산 → 기록 수정 → 최신 상태 검증 → 최종 JSON 순서로 진행합니다. 정답 경로는 도구4회와 생성5회로 예산 안에 있습니다.\n\n정수 최소단위는 소수점 계산을 피하기 위한 금액 표현입니다. `quantity`는 개수, `unit_price_minor`는 한 개 가격입니다. 할인은 정수비율이고 금액은 `floor((quantity*price*(100-discount)+50)/100)`입니다. UTC cutoff·버전·취소 우선 규칙은 초기 자료에 공개됩니다.\n\n'if ko else'The four families cover quotes, latest versions, delayed arrivals and corrections/cancellations. A valid path selects the source using the policy, reads it, calculates, writes the target, validates current state and then answers. The gold path needs 4 tools and 5 generations, within budget.\n\nMoney uses integer minor units to avoid floating-point rounding. `quantity` is the count and `unit_price_minor` is price per unit. The rule is `floor((quantity*price*(100-discount)+50)/100)`. UTC cutoff, version and cancellation precedence are visible in the initial evidence.\n\n')
        body+=('선택 규칙, 수량·할인, 후보 자료를8192token 입력의3600/5500/7350 위치에 둡니다. 도구8/32쌍은 길이와 근거 위치가 같습니다. 무관한 합성 padding으로 위치를 통제한 실험이며 자연 문서 모집단의 대표성은 측정하지 않았습니다. 초기 사실은 처음부터 사용 가능하고 명시한 자료 확인 외에 숨은 재조회 의무는 없습니다.\n\n'if ko else'Selection policy, quantity/discount and source candidates are placed at offsets3600/5500/7350 in8192-token inputs. Menus8/32 share lengths and evidence offsets. Irrelevant synthetic padding controls position; representativeness of natural-document workloads is not measured. Initial facts are available from generation0; there are no hidden reread obligations beyond the stated source verification.\n\n')
        body+=section(2)+table(['KV','storage','actual pool B','capacity tokens','layers','max context'],cache_rows)+'\n'
        body+=('목표 pool은 모두4,294,967,296B입니다. 실제 page 정렬 후의 저장량을 표에 기록했습니다. 같은 storage의 view를 중복 합산하지 않습니다. B16은147,456B/token, FP8은73,728B/token, NVFP4는 packed36,864+block-scale4,608=41,472B/token입니다. 저장토큰 용량은 처리량이 아닙니다. F8은 global scale1.0이 요청 전후 동일한 F8_DEFAULT_SCALE이며 P0에만 사용했습니다.\n\nNVFP4는 실제 uint8 packed data와 FP8 E4M3 block scale을 사용했습니다. GPU trace에 NVFP4 writer, FP4 cache를 읽는 FA2 prefill과 decode kernel이 있습니다. query/output이 BF16인 것은 BF16 cache fallback이 아닙니다. Kernel 이름은 `results/p0_diagnostics_r2/F4_kernel_profile.json`에 있습니다.\n\n'if ko else'The requested pool is4,294,967,296B for every arm; the table shows allocations after page alignment. Shared storage views are not double-counted. Per-token bytes are147,456 for B16,73,728 for FP8 and36,864 packed+4,608 block scales=41,472 for NVFP4. Capacity is not throughput. F8 uses unchanged global scales1.0 (F8_DEFAULT_SCALE), in P0 only.\n\nNVFP4 uses real uint8 packed data and FP8 E4M3 block scales. The GPU trace records its writer, FA2 prefill reading FP4 cache and the decode kernel. BF16 query/output does not imply BF16-cache fallback. Exact names are in `results/p0_diagnostics_r2/F4_kernel_profile.json`.\n\n')
        body+=section(3)
        body+=('P0는 KV별8개 SMOKE를 기술적으로 실행했습니다. 작은 NVFP4 검사에서 같은 압축 cache를 두 decoder로 읽은 차이는0이었고 BF16 원본 대비 상대 norm 오차는 약0.095였습니다. 별도 공식 test body는4layout PASS, 전체 upstream pytest suite는 NOT_RUN입니다. 실행 전 Case016 CPU 경계검사 34종을 실행했습니다. 실행 후 자료의 지연 도착·취소·ID/순서 성질을 확인하는 CPU 검토 3종도 별도로 통과했습니다.\n\n'if ko else'P0 technically executed8 SMOKE episodes per KV. The two decoders of the same compressed cache agreed exactly; relative norm error against original BF16 was about 0.095. Separately, the exact upstream test body passed 4 layouts; the full upstream pytest suite is NOT_RUN. There are 34 pre-run Case016 CPU boundary tests and 3 separately reported post-run CPU review properties.\n\n')
        body+=table(['KV','SMOKE observed','verified workflow'],[[kv,8,sum(e['VERIFIED_WORKFLOW_SUCCESS']is True for e in p0eps if e['kv']==kv)]for kv in ('B16','F8','F4')])+'\n'
        body+=('P1는 DEV12×KV2×메뉴2×cap2=96입니다. BF16 weights, TP1, FlashInfer, eager, async scheduling OFF, prefix caching OFF, chunked prefill2048, max-num-seqs4, context28672, clientC1을 공통 적용했습니다. generation 총8192, 요청8, 도구시도6, 요청180초, episode900초, temperature0, EOS준수입니다. 모델 결과를 본 뒤 과제·도구·예산을 변경하지 않았습니다. 환경·계측 오류와 수정은 amendment01~08로 구분합니다.\n\n'if ko else'P1 is12 DEV×2 KV×2 menus×2 caps=96. Common settings are BF16 weights,TP1,FlashInfer,eager,async scheduling OFF,prefix caching OFF,chunked prefill2048,max-num-seqs4,context28672 and clientC1. Budgets are8192 generated tokens,8 model calls,6 tool attempts,180s/request and900s/episode,temperature0 with EOS honored. Tasks/rules were not tuned after model outcomes. Setup/instrumentation fixes are separately recorded in amendments01–08.\n\n')
        body+='![All eight P1 cells](figures/p1_success.svg)\n\n'
        body+=('매 턴 전체 대화를 다시 보내며 prefix cache가 꺼져 있어 prefill이 반복됩니다. 각 HTTP 요청 내부에서만 autoregressive KV를 사용하고 종료 후 반납합니다. 한 episode 전체가 KV를 영구 점유하는 모형이 아닙니다.\n\n' if ko else 'Each turn resends the full history and repeats prefill with prefix caching OFF. Autoregressive KV is used within each HTTP request and released afterward; an episode does not permanently occupy its KV throughout the workflow.\n\n')
        body+=section(4)+table(['KV','menu','cap','verified/12','state/12','final/12','false done'],group_rows)+'\n'+table(['comparison','both success','loss','recovery','both fail'],paired_rows)+'\n'
        body+=f'Baseline usable: **{baseline_text}**; B16/menu8/cap1024 = **{baseline["verified"]}/12**, families {s["baseline_family_success"]}. '
        body+=('등록된9/12 및 각family2/3 기준은 후속 실행범위 관리 제안입니다. 기준선의 낮은 완주율 때문에 이 DEV의 paired 변화만으로 실용 우위나 동등성·비열등성을 판단하지 않습니다.\n\n'if ko else'The registered9/12 and at least2/3 per family criterion is an expansion-management proposal. The weak baseline limits interpretation of these DEV paired changes as practical superiority, equivalence or noninferiority.\n\n')
        body+=table(['KV','model requests','tool attempts','events','output tokens','worker attempt s','peak allocated GiB','peak reserved GiB'],cost_rows)+'\n'
        body+=f'P1 worker clock: **{s["worker_wall_seconds"]:.3f}s**. '+('로드·실패·I/O·계측을 포함하며 GPU busy나 serving speedup이 아닙니다. C1 혼합 DEV block의 makespan/correct-job 장부는 원 summary의 attempts에 있습니다. 성공0의 seconds/tokens-per-correct는 null입니다. 취소 probe의 수신 token은 기록됐지만 서버가 중단 전 만든 전체 token 수는 정확히 관측하지 못했습니다.\n\n'if ko else'This includes load, failures, I/O and metrics; it is neither GPU busy nor a serving speedup. C1 mixed-DEV makespan/correct-job ledgers are in summary.attempts. Seconds/tokens per correct job are null when success is0. Received tokens from cancel probes are preserved, but total engine generation before abort is not exactly observed.\n\n')
        body+=section(5)+table(['terminal reason','episodes'],end_rows)+'\n'+table(['first unmet obligation','episodes'],unmet_rows)+'\n'
        body+=('도구시도는 예산 내에서 호스트가 처리한 slot 수입니다. 생성됐지만 한도 때문에 실행에 진입하지 못한 slot은 원응답에 남습니다. 사건 수에는 registry·인자 거절도 포함하며, 실제 callable 성공 수는 diagnostics에서 따로 확인합니다.\n\n' if ko else 'Tool attempts count slots admitted within the host budget. Generated over-budget slots remain in raw responses. Event counts include registry/argument rejections; successful callable events are separately counted in diagnostics.\n\n')
        body+=('도구결과 valid:true는 공개 schema·불변식 통과이며 정답 금액이나 올바른 버전 선택을 보장하지 않습니다. B16 DEV-B00/menu8/cap1024는 필요한 버전3 대신2를 선택해348142를 저장했고 valid:true를 받았습니다. 독립 정답은253559이며 실패 판정은 유지됩니다. 이 사례는 전체 raw로 연결되며 다른 조건을 대표하는 유일한 사례로 선택한 것이 아닙니다.\n\n같은 response 안의 모든 call은 같은 생성 전 관측 목록을 가집니다. write→validate는 순차 실행을 허용하지만, 조회결과·계산결과가 필요한 다음 행동은 그 결과를 받은 뒤 생성되어야 합니다. first-unmet는 공개 DAG의 정해진 순서이고 모델 내부의 유일한 원인 진단이 아닙니다. 자유 rollout의 후속 대화가 다르면 동일 관측이라고 부르지 않습니다.\n\n'if ko else'Public valid:true checks schema/invariants, not the gold amount or correct version selection. B16 DEV-B00/menu8/cap1024 selected version2 rather than3, wrote348142 and received valid:true. The independent gold is253559, so failure is retained. The full record is linked; this is not a selected sole representative of all conditions.\n\nAll calls in one response share the same pre-generation observations. Write→validate can execute sequentially, but read/calculation-dependent actions must be generated after those results were received. First-unmet follows the registered DAG order, not a unique diagnosis of internal model causality. Diverged free rollouts are not claimed to have identical later observations.\n\n')
        body+=('[전체96개 raw](results/p1_r1/episodes/) · [가족별·파싱·길이 진단](results/p1_r1/diagnostics.json) · [paired CSV](results/p1_r1/paired.csv)\n\n'if ko else'[All 96 raw records](results/p1_r1/episodes/) · [Family/parse/length diagnostics](results/p1_r1/diagnostics.json) · [Paired CSV](results/p1_r1/paired.csv)\n\n')
        body+=section(6)+('[실행 명령과 CPU 감사 범위](REPRODUCE.ko.md). 원 SSE를 재조립하고 새 fixture에서 상태·관측·endpoint를 다시 계산합니다. scalar audit는 모델 추론 자체를 재현하지 않습니다. 입력과 원 모델을 재다운로드하거나 quantizer를 실행하지 않습니다.\n\n'if ko else'[Commands and CPU audit scope](REPRODUCE.md). Reassemble raw SSE and replay state/observations/endpoints on fresh fixtures. A scalar audit does not reproduce model inference. No model download or quantizer is needed for the CPU path.\n\n')
        body+=section(7)+f'Model revision: `cdbee75f17c01a7cc42f958dc650907174af0554`. P1 execution source SHA256: `{source}`. Git base: `0c86afa4728cbf901f6c3df64802810db6581a00`.\n\n'
        body+=table(['scope','status'],[['vLLM/NVFP4 storage + 36 layers','observed'],['P1 DEV96','completed'],['EVAL48 /16K /C2,C4 /P2,P3','NOT_RUN'],['weight W4 /D16 /training','NOT_RUN'],['Git stage/commit/push/PR/Pages','NOT_RUN'],['full model logits independently checked finite','NOT_RUN; small cache decode finite checked']])+'\n'
        body+=('P2/P3를 시작하려면 기준선이 왜 실패했는지 확인하고 별도 승인·새 버전·새 동결 계획이 필요합니다. 이번 결과를 고치기 위한 즉석 prompt/도구/한도 sweep은 하지 않았습니다.\n\n'if ko else'P2/P3 require review of baseline failures, separate approval and a newly frozen version/plan. No prompt/tool/budget sweep was added to improve this run.\n\n')
        body+='[vLLM0.32 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0), [NVFP4 SM8x/SM12x support](https://github.com/vllm-project/vllm/pull/46963), [SM12x XQA decode](https://github.com/vllm-project/vllm/pull/60452), [Qwen function calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html). [Source audit](provenance/source_audit.json), [reuse map](REUSE_MAP.md), [NOTICE](NOTICE.md).\n\n'
        body+=('설치한 공식 engine/kernel/model과 자체 구현한 fixture·client·실행·채점·감사를 구분합니다. Codex 보조와 별도 scalar 계산 검토를 사용했으며 사람의 독립 심사나 외부 GPU 재현으로 표시하지 않습니다.\n'if ko else'Upstream engine/kernels/model are distinguished from the implemented fixtures/client/runner/scoring/audit. Codex assistance and separate scalar calculations are disclosed; they are not human peer review or external GPU reproduction.\n')
        (case/('REPORT.ko.md'if ko else'REPORT.md')).write_text(body)
    print(json.dumps({'status':'PASS','groups':8,'episodes':96,'verified':counts,'reports':['REPORT.ko.md','REPORT.md']}))

if __name__=='__main__':main()
