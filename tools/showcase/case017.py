"""Case017: raw-backed display with an explicit expansion-gate deviation."""
from html import escape
from pathlib import Path
import importlib.util
import functools
import io
import re
import tarfile
from common import read, require, sha

C17='cases/017-agent-contract-cost-bench'
TITLES={'ko':'도구 호출 규칙과 업무 완주 비용','en':'Agent Contract–Cost Bench'}
SECTIONS=('overview','workflow','policies','results','cost','episodes','protocol-deviation','reproduce','references')
SOURCE_COPIES={
    **{'case017-'+kind+'-'+lang+'.md.txt':C17+'/'+name+('.ko.md' if lang=='ko' else '.md')
       for kind,name in [('guide','README'),('report','REPORT'),('reproduce','REPRODUCE'),('deviation','PROTOCOL_DEVIATION')] for lang in ('ko','en')},
    'case017-runner.py.txt':C17+'/src/agent_contract_cost/runner.py',
    'case017-parser.py.txt':C17+'/src/agent_contract_cost/contracts.py',
    'case017-audit.py.txt':C17+'/src/agent_contract_cost/audit.py',
    'case017-tests.py.txt':C17+'/tests/test_runner.py',
    'case017-public-checks.py.txt':C17+'/publication/run_checks.py',
    'case017-wording.py.txt':C17+'/publication/wording.py',
    'case017-protocol.json.txt':C17+'/specs/protocol.json',
    'case017-episode-ledger.csv.txt':C17+'/publication/summary/episode_ledger.csv',
    'case017-policy-results.csv.txt':C17+'/publication/summary/policy_results.csv',
    'case017-notice.md.txt':C17+'/NOTICE.md',
    'case017-mapping.json.txt':C17+'/publication/provenance/original_mapping.json',
}
EVIDENCE_ASSET='data/case017-original-evidence.tar.xz'
EVIDENCE_SOURCE=C17+'/publication/data/original-evidence.tar.xz'
FIGURES={'assets/case017-p2-'+lang+'.svg':C17+'/publication/summary/p2-completions-'+lang+'.svg' for lang in ('ko','en')}

@functools.lru_cache(maxsize=2)
def check_evidence_download(blob):
    """One explicit source-matched binary; inspect members instead of UTF-8 decoding XZ."""
    require(blob.startswith(b'\xfd7zXZ\x00'),'Case017 evidence XZ signature')
    from pathlib import PurePosixPath
    sensitive=re.compile(r'/home/[A-Za-z0-9_.-]+/|/mnt/[a-z]/Users/|[A-Za-z]:[\\/]+Users[\\/]|(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,})|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\s*\n')
    names=set()
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:xz') as tar:
        for entry in tar:
            p=PurePosixPath(entry.name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in entry.name and ':' not in entry.name,'Case017 evidence path')
            require(entry.isfile() and entry.name not in names,'Case017 regular unique evidence member')
            require(entry.size<=16*1024*1024,'Case017 member size')
            text=tar.extractfile(entry).read().decode('utf-8')
            require(not sensitive.search(text),'Case017 private path/credential in evidence')
            names.add(entry.name)
    require(len(names)==418,'Case017 expected evidence member coverage')
    return len(names)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def load_case017(root):
    case=root/C17
    if not (case/'README.md').is_file():return None
    module('case017_public_verify',case/'publication/verify.py').verify(case)
    data=module('case017_public_derive',case/'publication/derive.py').project(case)
    require(data==read(case/'publication/summary/project.json'),'Case017 display differs from lossless raw records')
    data['display_source_files']={p:sha(root/p) for p in [*SOURCE_COPIES.values(),*FIGURES.values(),EVIDENCE_SOURCE,
        C17+'/publication/summary/project.json',C17+'/publication/PUBLIC_MANIFEST.json']}
    return data

def a(url,label):return '<a href="'+escape(url,quote=True)+'">'+escape(label)+'</a>'
def table(headers,rows,label):
    return '<div class="scroll" tabindex="0" role="region" aria-label="'+escape(label,quote=True)+'"><table><thead><tr>'+''.join('<th scope="col">'+escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def group(d,kv,policy):return next(g for g in d['groups'] if (g['stage'],g['kv'],g['policy'])==('P2',kv,policy))
def cost_change(d,key):
    a,b=group(d,'K1','C0'),group(d,'K1','G')
    return 100*((b[key]/b['verified'])/(a[key]/a['verified'])-1)
def recovery(d):return next(r for r in d['comparisons'] if r['stage']=='P2' and r['comparison_id']=='K1:G-C0')
def successes(d,lang):
    return table(['KV','C0','G','N','E'],[[kv+' · '+('BF16 KV' if kv=='K0' else 'NVFP4 KV'),*[str(group(d,kv,p)['verified'])+'/'+str(group(d,kv,p)['n']) for p in ('C0','G','N','E')]] for kv in ('K0','K1')], 'P2 verified workflow completions')
def wording(data,lang):
    root=Path(__file__).resolve().parents[2]
    return module('case017_public_wording',root/C17/'publication/wording.py').public_text(data,lang)

def deviation(data,lang):
    return '<p class="case017-deviation">'+escape(wording(data,lang)['deviation'])+' '+a('case017.html#protocol-deviation','경위·원안 근거' if lang=='ko' else 'Chronology and original intent')+'</p>'

def home_text(data,lang):
    """Short entry copy; detailed wording and retained case documents stay intact."""
    free,grammar=group(data,'K1','C0'),group(data,'K1','G')
    n=free['n']
    gate=data['gate'];observed=gate['observed_dev_verified'];required=gate['required_dev_verified'];dev_n=gate['denominator']
    if lang=='ko':
        return {
            'intro':'상품 합계 기록·창고 이동·문서 정보 변경을 시험하고, 저장된 값·수정 후 받은 검증 결과·최종 답을 함께 검사하는 평가기를 구현했습니다.',
            'result':f"BF16 가중치를 고정한 신규 합성 {n}과제에서, NVFP4 KV의 문법 제약은 자유 출력보다 완주 {free['verified']}/{n}→{grammar['verified']}/{n}, 상태 정확 {free['state_correct']}/{n}→{grammar['state_correct']}/{n}를 기록했습니다. 실패를 포함한 실행 비용과 남은 오류도 공개합니다.",
            'deviation':f'DEV에서 BF16 KV·자유 출력은 {observed}/{dev_n}로 확대 기준 {required}/{dev_n}에 미달했지만, 로컬 gate의 기준 누락으로 P2가 진행됐습니다. 이 이탈을 공개한 탐색적 측정입니다.',
        }
    return {
        'intro':'We built a benchmark that checks saved state, validation observed after updates, and final answers on synthetic order, inventory and document tasks.',
        'result':f"With BF16 weights fixed on {n} new synthetic tasks, grammar-constrained output with native NVFP4 KV completed {grammar['verified']}/{n} versus {free['verified']}/{n} with free output; correct saved states rose from {free['state_correct']}/{n} to {grammar['state_correct']}/{n}. Costs including failures and remaining errors are reported alongside these observations.",
        'deviation':f'BF16 KV/free output completed {observed}/{dev_n} DEV tasks, below the {required}/{dev_n} expansion criterion. The local gate omitted this threshold and proceeded to P2; we disclose the deviation and treat P2 as exploratory.',
    }

def home_case017(data,lang):
    t=lambda k,e:k if lang=='ko' else e
    copy=home_text(data,lang)
    return '<section class="panel" id="contract-cost-study"><p class="eyebrow">CASE 017 · AGENT CONTRACT &amp; COST</p><h2>'+a('case017.html#overview',TITLES[lang])+'</h2><p>'+escape(copy['intro'])+'</p><p>'+escape(copy['result'])+'</p><p class="case017-deviation">'+escape(copy['deviation'])+'</p><div class="actions">'+a('case017.html#results',t('결과','Results'))+' · '+a('case017.html#episodes',t('실제 사건','Recorded episodes'))+' · '+a('case017.html#reproduce',t('CPU 재생','CPU replay'))+' · '+a('case017.html#protocol-deviation',t('확대 기준 이탈 경위','Protocol deviation'))+'</div></section>'

def render_case017(data,lang):
    t=lambda k,e:k if lang=='ko' else e
    copy=wording(data,lang)
    labels=t(['한눈에 보기','어떤 일을 시험했나요?','KV와 네 출력 규칙','완주·상태·최종 답','완주와 추가 비용','실제 사건 탐색','확대 기준 누락','직접 확인·실행','출처와 구현 기여'],
             ['Overview','What work was tested?','KV and four output policies','Completion, state and final answer','Completions and added cost','Recorded episode explorer','Omitted expansion criterion','Check and reproduce','Sources and contributions'])
    out='<header class="study-intro"><p class="eyebrow">DIOVA · CASE 017</p><h1>'+escape(TITLES[lang])+'</h1><p class="lede">'+escape(copy['subtitle'])+'</p><p>'+escape(copy['intro'])+'</p><p>'+escape(copy['implementation'])+'</p></header><div class="study-layout"><aside class="study-toc"><nav aria-label="'+t('사례 목차','Case contents')+'"><ol>'+''.join('<li>'+a('#'+s,l)+'</li>' for s,l in zip(SECTIONS,labels))+'</ol></nav></aside><article class="study-article">'
    def section(i,body):return '<section class="study-section" id="'+SECTIONS[i]+'"><h2>'+labels[i]+'</h2>'+body+'</section>'
    out+=section(0,'<p>'+escape(copy['kv_intro'])+'</p><p>'+escape(copy['policy_intro'])+'</p>'+successes(data,lang)+'<p>'+escape(copy['headline'])+'</p><p>'+escape(copy['paired'])+'</p>'+deviation(data,lang))
    out+=section(1,'<ol class="case017-flow">'+''.join('<li>'+escape(x)+'</li>' for x in t(['기록과 요청 읽기','계산하거나 수정하기','현재 상태의 검증 결과 받기','실제 값으로 최종 답하기'],['Read records and request','Calculate or update','Observe current-state validation','Report actual final values']))+'</ol>'+table([t('합성 업무','Synthetic workflow'),t('모델이 해야 할 일','Required work')],t([
       ['A · 상품 합계','상품 개수·한 개 가격·할인으로 합계 계산 후 기록. 금액은 정수 최소단위입니다.'],['B · 창고 이동','두 창고의 재고와 수정 번호(revision)를 읽고 정해진 개수를 한 번에 이동합니다.'],['C · 문서 정보','문서 수정 번호를 읽고 요청한 상태·태그·담당자로 바꿉니다.']],
       [['A · Product totals','Read quantities, per-item prices and discounts; calculate and record an integer minor-unit total.'],['B · Stock transfer','Read two stock records/revisions and atomically move the requested quantity.'],['C · Document metadata','Read the revision and update requested status, tags and owner.']]),'Three owned synthetic workflows')+'<p>'+t('실제 주문·결제·회사 시스템과 연결하지 않습니다. 검증 도구는 공개된 기록 규칙만 확인하며 숨겨진 정답을 알려주지 않습니다.','These fixtures connect to no real orders, payments or company systems. Validators check public invariants, not hidden task correctness.')+'</p>')
    out+=section(2,'<p>'+t('Qwen3-4B 가중치는 BF16으로 고정했습니다. KV 캐시는 모델이 앞선 토큰을 처리하며 저장한 중간 계산값입니다. K0는 BF16, K1은 native NVFP4 packed 값과 FP8 block scale을 저장합니다.','Qwen3-4B weights remain BF16. KV cache stores intermediate attention values for preceding tokens. K0 uses BF16; K1 uses native NVFP4 packed values and FP8 block scales.')+'</p>'+table(['Policy',t('출력·복구 규칙','Output/retry rule')],t([
      ['C0 · 자유 출력','형식을 안내하고 첫 호출 규칙 위반에서 종료'],['G · 문법 제약','매 응답에 공개 JSON schema, 재시도 없음'],['N · 중립 재시도','첫 계약 오류에 중립 안내로 한 번 재생성'],['E · 오류 설명 재시도','같은 기회에 공개 오류 코드·위치·기대 타입 안내']],
      [['C0 · Free','Stop at the first contract rejection'],['G · Grammar','Public JSON schema every response, no retry'],['N · Neutral retry','One neutral retry after the first contract rejection'],['E · Error retry','One retry with the public error code, path and type']]),'Common output contract')+'<p>'+t('정답 ID·금액·행동 순서는 문법에 넣지 않습니다. 네 정책 모두 같은 단일 JSON·엄격 parser·도구를 사용합니다. 지원되지 않은 uniqueItems는 모든 정책에서 host가 검사합니다.','Grammar contains no correct ID, amount or action sequence. All policies share a single-JSON contract, strict parser and tools. Unsupported uniqueItems is checked by the host in all policies.')+'</p><details><summary>'+t('저장량·설정·동결 소스','Storage, settings and frozen source')+'</summary>'+table(['KV','B/token','Pool bytes','Token slots'],[[r['kv'],r['bytes_per_token'],r['pool_bytes'],r['slots']] for r in data['capacity']],'Native KV capacity')+'<p>'+t('약3.556배는 같은 약4GiB pool의 token 슬롯 비율입니다. 한 요청 context는28672이며, 전체 VRAM 감소·단일100K 문맥·속도 향상을 뜻하지 않습니다. C1, prefix cache OFF, BF16 weights, vLLM0.32.0/FlashInfer, 공통GRAPH입니다.',
        'Approximately3.556× describes token slots in approximately4GiB pools. Per-request context is28672; this is not total VRAM reduction, a single100K context or a speedup. C1, prefix cache OFF, BF16 weights, vLLM0.32.0/FlashInfer and common GRAPH mode are used.')+'</p>'+a('../sources/case017-protocol.json.txt',t('원 protocol','Original protocol'))+'</details>')
    groups=[g for g in data['groups'] if g['stage']=='P2']
    out+=section(3,table(['KV','Policy',t('완주 V','Complete V'),t('상태 S','State S'),t('최종 답 F','Final F'),'Fresh*'],[[g['kv'],g['policy'],g['verified'],g['state_correct'],g['final_correct'],g['fresh_validation']] for g in groups],'P2 separate endpoint counts')+'<p>'+t('모두 24과제 중 개수입니다. V는 저장·검증·최종 답까지 맞춘 완주, S는 저장된 값, F는 최종 답의 정확성입니다. Fresh*는 현재 저장 상태의 검증 결과를 답하기 전에 받았는지 원 사건에서 계산한 사후 지표입니다. Fresh만으로 정답은 아닙니다. C00_K1_G는 현재 검증을 받았지만 review 대신 published를 저장했습니다.',
        'Every count is out of 24. V is verified completion, S is correct saved state and F is a correct final answer. Fresh* is a posthoc check of whether validation of the current saved state was observed before answering. Fresh alone does not prove correctness: C00_K1_G observed current validation but stored published instead of review.')+'</p><figure><img src="../assets/case017-p2-'+lang+'.svg" alt="'+t('P2 각 정책의 완주 수, 분모24, 축0부터','P2 completions per policy, denominator24, zero-origin axis')+'"><figcaption>'+t('완주 수는 높을수록 좋습니다. 상태·최종 답·Fresh의 의미는 표에서 구분합니다.','Higher completion counts are better. The table separately defines state, final-answer and Fresh outcomes.')+'</figcaption></figure><p>'+t('G 창고 실패16회: 고유7과제×두 KV의 최신 검증 누락14회, B05×두 KV의 SKU 접두사 누락·이동 미실행2회입니다. BF16 G는 완주 수가 같아도 상태 정확21→23으로 늘었습니다. NVFP4 N은 B03 창고 이동을 유일하게 완주했습니다.',
        'The16 G warehouse failures comprise14 stale-validation cases (seven tasks ×two KV configurations) and two B05 cases with a missing SKU prefix and no transfer. BF16 G leaves completion unchanged but raises correct states21→23. NVFP4 N alone completed the B03 warehouse transfer.')+'</p><details><summary>'+t('P1·family·paired 구간','P1, families and paired intervals')+'</summary>'+table(['KV','C0','G','N','E'],[[kv,*[str(next(g for g in data['groups'] if (g['stage'],g['kv'],g['policy'])==('P1',kv,p))['verified'])+'/8' for p in ('C0','G','N','E')]] for kv in ('K0','K1')],'P1 DEV denominator8')+table(['KV','Policy','A /8','B /8','C /8'],[[g['kv'],g['policy'],*[g['family_verified'][f] for f in ('A','B','C')]] for g in groups],'P2 by synthetic family')+table([t('비교','Comparison'),t('회복','Recovery'),t('손실','Loss'),'pp','95% pp'],[[r['comparison_id'],r['recovery'],r['loss'],f"{r['difference_pp']:.2f}",str([round(x,2) for x in r['ci_95_pp']])] for r in data['comparisons'] if r['stage']=='P2' and r['comparison_id'] in ('K1:G-C0','K1:E-N')],'Exploratory paired changes')+'<p>'+t('family별 task를 재표집하며 한 task의8조건을 함께 유지한 bootstrap10000회/seed6102013입니다. E−N 구간은0을 포함합니다. 신규 합성24과제는 독립 회사가 아니며, 독립 자유 실행의 E−N 차이는 오류 설명 정보만의 인과효과가 아닙니다.',
        'The original10000-draw bootstrap, seed6102013, resamples tasks within families while retaining all eight conditions. E−N includes zero. These24 synthetic tasks are not independent companies; E−N compares free policies rather than isolating feedback information causally.')+'</p></details>')
    out+=section(4,'<p>'+escape(copy['cost'])+'</p>'+table(['KV','Policy','Requests','Input tokens','Output tokens','Episode wall s'],[[g['kv'],g['policy'],g['requests'],g['input_tokens'],g['output_tokens'],f"{g['wall_seconds']:.3f}"] for g in groups],'All24 episode costs including failures')+'<p>'+t('NVFP4 C0→G: 입력/완주14062.4→14571.9(+3.62%), 출력/완주352.9→396.5(+12.36%), episode 초/완주5.60→6.23(+11.27%). 실패 비용을 포함한24개 전체 비용을 완주 수로 나눴습니다. 성공 episode만의 평균 latency·과금·전력·P3 처리량이 아닙니다.',
        'NVFP4 C0→G: input tokens/completion14062.4→14571.9(+3.62%), output352.9→396.5(+12.36%), episode seconds/completion5.60→6.23(+11.27%). These divide all24 costs, including failures, by completions; they are not successful-episode mean latency, billing, energy or P3 throughput.')+'</p><p>'+t('P0/P1/P2 요청22/233/775=1030, 출력678/11054/38853=50585. Worker wall1035.148295초는 서버 로드·컴파일·종료도 포함하며 episode wall과 범위가 다릅니다. 입력·출력 토큰을 같은 계산 비용으로 간주하지 않습니다.',
        'P0/P1/P2 requests22/233/775=1030; output678/11054/38853=50585. Worker wall1035.148295s includes server load, compilation and shutdown, unlike episode walls. Input/output tokens are not interchangeable compute costs.')+'</p>')
    explorer='<p>'+t('아래7개는 선택한 실제 사건입니다. 요약과 모델 원문을 구분하고, 같은 turn 번호나 실행 직전 상태만으로 같은 관측이었다고 간주하지 않습니다. 전체256개 원기록은 별도 압축 원자료에 보존합니다.',
        'These are seven selected measured episodes. Summaries and model raw text are separate. Equal turn numbers or pre-execution state do not establish equal observations. All256 complete records are retained in the compressed evidence.')+'</p><div class="case017-filters" id="case017-filters" hidden><label>'+t('실제 사건 선택','Choose a recorded episode')+'<select id="case017-episode"><option value="">'+t('모두 보기','Show all')+'</option>'+''.join('<option value="'+escape(e['episode_id'],quote=True)+'">'+escape(e['episode_id'])+'</option>' for e in data['selected_episodes'])+'</select></label><button type="button" id="case017-reset">'+t('전체 사건','All episodes')+'</button><p id="case017-count" role="status" aria-live="polite"></p></div>'
    notes={
       'HOLDOUT-A00_K1_C0':t('여분 } 때문에 첫 조회가 실행되지 않았습니다.','An extra } prevented the first read.'),
       'HOLDOUT-A00_K1_G':t('같은 과제에서 계산·기록·현재 검증·final까지 완료했습니다.','Completed calculation, writing, current validation and final on the same task.'),
       'HOLDOUT-B01_K0_G':t('version0 검증 뒤 version1 이동을 하고 새 검증 없이 final. 상태·최종 숫자는 맞습니다.','Validated version0, transferred to version1, then issued final without new validation; state and final numbers are correct.'),
       'HOLDOUT-C00_K1_G':t('review 요청에 published를 저장했습니다. valid:true는 공개 불변식 검사이며 요청 정답이 아닙니다.','Stored published for a review request. valid:true checks public invariants, not the requested answer.'),
       'HOLDOUT-B03_K1_N':t('중립 재시도 후 이동→현재 검증→final로 완주했습니다.','After neutral retry, transfer→current validation→final completed.'),
       'HOLDOUT-B03_K1_G':t('파싱은 정상이지만 이동 전에 검증해 최신 관측이 없습니다.','Parsing succeeds, but validation precedes transfer and is stale.'),
       'HOLDOUT-B03_K1_E':t('N과 최초 입력·여분 } 실패가 같지만 이후 목적지→출발지 조회·이동 전 검증으로 갈라집니다.','Shares N’s first input and extra-} failure, then reads destination→source and validates before transfer.')}
    for e in data['selected_episodes']:
        s=e['scores'];eid=e['episode_id']
        explorer+='<details class="case017-episode" id="episode-'+escape(eid,quote=True)+'"><summary>'+escape(eid)+' · V/S/F/Fresh='+str(s['verified'])+'/'+str(s['state_correct'])+'/'+str(s['final_correct'])+'/'+str(s['fresh_validation'])+'</summary><p>'+escape(notes[eid])+'</p><h3>'+t('요청 원문','Original task request')+'</h3><pre>'+escape(e['request'])+'</pre>'
        explorer+='<p>Recorded episode_id: <code>'+escape(e['recorded_episode_id'])+'</code></p><details><summary>'+t('수정 전후 JSON','Before/after JSON')+'</summary><pre>'+escape(json_dump({'initial_state':e['initial_state'],'final_state':e['final_state']}))+'</pre></details>'
        for turn in e['turns']:
            explorer+='<section class="case017-turn"><h3>'+t('생성 ','Generation ')+str(turn['generation_turn'])+'</h3><p>'+t('생성 전 받은 사건: ','Events observed before generation: ')+escape(str(turn['available_result_event_ids']))+' · state version '+str(turn['state_version'])+'</p><details><summary>'+t('모델 원문 열기','Open model raw text')+'</summary><pre class="case017-raw">'+escape(turn['raw_text'])+'</pre></details>'
            if turn['feedback'] is not None:explorer+='<p>'+t('다음 user 피드백: ','Next user feedback: ')+'<code>'+escape(turn['feedback'])+'</code></p>'
            explorer+='<details><summary>'+t('파서·실제 사건·상태·해시','Parser, actual event, state and hashes')+'</summary><pre>'+escape(json_dump({k:turn[k] for k in ('parse','event','input_token_sha256','observation_sha256','state_digest','raw_utf8_sha256')}))+'</pre></details></section>'
        explorer+='<p>'+t('원기록 포인터: ','Original record pointer: ')+'<code>'+escape(s['source_file'])+'</code> · SHA256 <code>'+escape(s['source_sha256'])+'</code></p></details>'
    out+=section(5,explorer)
    out+=section(6,deviation(data,lang)+'<p>'+t('장부상 HOLDOUT 생성·봉인·P2 freeze는 첫 기록 응답보다 앞섰고 공통35개 파일이 동일했습니다. 이는 local 자기기록이며 gate 준수와 별도입니다. 원 source·protocol·보고서·점수는 소급 수정하지 않았습니다. 이번에 원안의6/8 문구를 사용자로부터 추가 전달받았으며, local Codex가 실험 전에 전문을 받았다는 증거는 아닙니다.',
        'Local records place HOLDOUT generation, sealing and P2 freeze before the first recorded response;35 common files match. This is self-recorded evidence, separate from gate compliance. Original source, protocol, reports and scores were not retroactively changed. The user newly supplied the original6/8 excerpt; it is not proof of full pre-experiment delivery to local Codex.')+'</p>'+a('../sources/case017-deviation-'+lang+'.md.txt',t('경위·시점·미확인 사항 전체','Full chronology, evidence and unknowns')))
    command='python3 -B cases/017-agent-contract-cost-bench/publication/run_checks.py --output /tmp/case017-public-check-new --tests'
    out+=section(7,'<p>'+t('저장소 clone 후 Python 표준 라이브러리로 실행합니다. 모델·개인 경로 없이 압축 원문을 새 디렉터리에 복원해256개 endpoint를 재생합니다. 저장 IDs/hash를 확인하며 새 tokenization이나 GPU 추론 재현은 하지 않습니다.',
        'From a repository clone, Python standard-library commands restore compressed raw records into a fresh directory and replay all256 endpoints without a model or personal paths. Saved IDs/hashes are checked; no new tokenization or GPU inference is reproduced.')+'</p><pre>'+escape(command)+'</pre><div class="actions">'+a('../sources/case017-reproduce-'+lang+'.md.txt',t('CPU/GPU 재현 범위','CPU/GPU reproduction scope'))+' · '+a('../'+EVIDENCE_ASSET,t('손실 없는 원자료 다운로드','Download lossless evidence'))+' · '+a('../sources/case017-episode-ledger.csv.txt','256-row CSV')+' · '+a('../sources/case017-mapping.json.txt',t('복원 경로·hash','Restoration mapping/hashes'))+'</div><p>'+t('P3·추가 모델 실험·배포는 이번 로컬 준비에서 실행하지 않았습니다.','P3, additional model experiments and deployment were not run in this local preparation.')+'</p>')
    out+=section(8,'<p>'+t('vLLM·FlashInfer·xgrammar와 Qwen은 upstream입니다. 직접 구현한 것은 parser·정책·합성 fixture·관측 추적·endpoint 감사·표시 도구입니다. Codex 지원과 별도 AI 보조 CPU 검산을 명시합니다. 사람 심사·외부 GPU 재현으로 표현하지 않습니다. ChatGPT 보완 CSV 검토는256행 산술 범위이고 전체 fixture 재생은 local receipt의 범위입니다.',
        'vLLM, FlashInfer, xgrammar and Qwen are upstream. Direct contributions are the parser, policies, synthetic fixtures, observation tracking, endpoint auditor and display tooling. Codex and separate AI-assisted CPU checking are disclosed; they are not human peer review or external GPU reproduction. ChatGPT addendum-CSV review covers256-row arithmetic; full fixture replay is evidenced by local receipts.')+'</p><div class="actions">'+a('../sources/case017-report-'+lang+'.md.txt',t('정식 보고서','Full report'))+' · '+a('../sources/case017-parser.py.txt','Parser')+' · '+a('../sources/case017-runner.py.txt','Runner')+' · '+a('../sources/case017-audit.py.txt','Auditor')+' · '+a('../sources/case017-notice.md.txt','NOTICE')+'</div>')
    return out+'</article></div>'

def json_dump(v):
    import json
    return json.dumps(v,ensure_ascii=False,indent=2)
