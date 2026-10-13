"""Case013 display: recorded variant quality and blocked packed correspondence."""
from html import escape
from pathlib import Path
import importlib.util
from common import read, require, sha

C13='cases/013-dualquant-reference-quality'
FIGURES={f'assets/case013-{name}':C13+'/figures/'+name for name in ('ppl.svg','dev-success.svg','packed-gates.svg')}
SECTIONS=('overview','implementation','results','analysis','cost','reproduce','report')
SOURCE_COPIES={
    'case013-report-en.md.txt':C13+'/REPORT.md', 'case013-report-ko.md.txt':C13+'/REPORT.ko.md',
    'case013-reproduction-en.md.txt':C13+'/REPRODUCTION.md', 'case013-reproduction-ko.md.txt':C13+'/REPRODUCTION.ko.md',
    'case013-notice.md.txt':C13+'/NOTICE.md','case013-solver.py.txt':C13+'/source/paper_int4.py',
    'case013-codec.py.txt':C13+'/source/reference_int4.py','case013-runner.py.txt':C13+'/source/paper_model_runner.py',
    'case013-audit.py.txt':C13+'/analysis/audit.py','case013-tests.py.txt':C13+'/tests/test_records.py',
    'case013-method-contract.md.txt':C13+'/METHOD_CONTRACT.md','case013-evaluation.md.txt':C13+'/EVALUATION_PROTOCOL.md',
    'case013-failures.md.txt':C13+'/FAILURE_ANALYSIS.md','case013-packed-probe.md.txt':C13+'/PACKED_PROBE.md',
    'case013-roadmap.md.txt':C13+'/ROADMAP.md',
}

def load_case013(root:Path):
    case=root/C13
    if not (case/'README.md').exists():return None
    spec=importlib.util.spec_from_file_location('case013_display_derive',case/'publication/derive.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    data=read(case/'summary/project.json')
    require(data==mod.derive(case),'Case013 projection differs from saved records')
    names=[C13+'/summary/project.json',C13+'/provenance/PUBLIC_MANIFEST.json',*SOURCE_COPIES.values(),*FIGURES.values(),
           *(C13+'/'+name for name in data['source_identity'])]
    return {**data,'sources':{name:sha(root/name) for name in names},
            'evidence_kind':'RECORDED_INT4_REFERENCE_VARIANT_NOT_FORMAL_SERVING'}

def a(url,label):return '<a href="'+escape(url,quote=True)+'">'+escape(label)+'</a>'
def table(headers,rows,label):
    return '<div class="scroll" role="region" tabindex="0" aria-label="'+escape(label)+'"><table><thead><tr>'+''.join('<th scope="col">'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def figure(name,alt,caption):
    return '<figure><img style="max-width:100%;height:auto" src="../assets/case013-'+name+'" alt="'+escape(alt,quote=True)+'"><figcaption>'+escape(caption)+'</figcaption></figure>'

def home_case013(data,lang):
    ko=lang=='ko'
    return ('<section class="panel" id="dualquant-study"><p class="eyebrow">CASE 013 · INT4 REFERENCE QUALITY</p><h2>'
        +('가중치 오차가 줄면 실제 답변도 좋아질까?' if ko else 'Does lower weight error improve real answers?')+'</h2><p>'
        +('가중치를 실제 4-bit로 저장하고 행·열 스케일을 유지하는 실행기를 만들었습니다. 같은 입력에서 문장 점수, 장문 답, 도구 호출을 비교합니다.' if ko else 'Built real 4-bit weight storage and a runner that preserves row/column scales. Compare language scores, long-input answers and tool use on matched inputs.')+'</p><p>'
        +('작은 모델의 확률 점수는 좋아졌지만, 4B에서는 도구 작업 성공이 줄었습니다. 가중치 오차 감소와 실제 답변 개선은 같은 결과가 아니었습니다.' if ko else 'Probability scores improved in the small model, while tool-task success decreased in 4B. Lower weight error did not consistently improve real answers.')+'</p><p class="scope">'
        +('Qwen 두 모델 · RTX 5080 · 복원 BF16 품질 비교 · packed 전체 모델 수치 기준 미통과' if ko else 'Two Qwen models · RTX 5080 · Reconstructed BF16 quality path · Full packed-model numeric gate not met')+'</p><div class="actions">'
        +a('case013.html#overview','쉽게 살펴보기' if ko else 'Overview')+' · '
        +a('case013.html#implementation','구현 코드' if ko else 'Implementation')+' · '
        +a('case013.html#results','전체 결과' if ko else 'Results')+'</div></section>')

def render_case013(data,lang):
    ko=lang=='ko';txt=lambda en,korean:korean if ko else en
    labels=['한눈에 보기','구현','실험과 결과','해석','시간·메모리','CPU 검산','보고서·출처'] if ko else ['Overview','Implementation','Experiment and results','Interpretation','Time and memory','CPU checks','Report and sources']
    out='<header><p class="eyebrow">DIOVA · CASE 013</p><h1>'+txt('Does Lower Weight Error Improve Real Answers?','가중치 오차가 줄면 실제 답변도 좋아질까?')+'</h1><p class="lede">'+txt('Stored INT4 codes and scales, restored complete models, and checked language scores, long answers and tool actions on the same inputs.','INT4 code와 scale을 저장하고 전체 모델을 복원했습니다. 같은 입력에서 문장 점수, 장문 답과 실제 도구 행동을 확인합니다.')+'</p></header>'
    out+='<nav aria-label="'+txt('Page contents','페이지 목차')+'"><ol>'+''.join('<li>'+a('#'+key,label)+'</li>' for key,label in zip(SECTIONS,labels))+'</ol></nav>'
    def section(i,body):return '<section class="panel study-section" id="'+SECTIONS[i]+'"><h2>'+labels[i]+'</h2>'+body+'</section>'
    out+=section(0,'<p>'+txt('Question: can more accurate compressed weights improve actual answers and tool use?','질문: 더 정확하게 압축한 가중치가 실제 답변과 도구 사용도 개선할까요?')+'</p><p>'+txt('Both models had lower weight error with D16 than RTN. PPL improved in 0.6B but slightly worsened in 4B; free-agent success also fell in 4B.','D16의 가중치 오차는 RTN보다 두 모델 모두 작았습니다. 0.6B의 PPL은 좋아졌지만 4B에서는 조금 나빠졌고, 자유 도구 작업의 성공도 줄었습니다.')+'</p><p>'+txt('PPL is a next-symbol probability score: lower is better. Agent success requires the correct final answer and any required actions.','PPL은 다음 기호에 준 확률 점수로, 작을수록 좋습니다. 에이전트 성공은 최종 답과 요구한 실제 실행 조건을 함께 확인합니다.')+'</p>')
    out+=section(1,'<ol><li>'+txt('Original weights → group64 row/column scale updates','원래 가중치 → group64 행·열 스케일 갱신')+'</li><li>'+txt('INT4 codes + scales → packed files → reload','INT4 code + scale → packed 파일 → 재로딩')+'</li><li>'+txt('Same token inputs → BF16 / RTN / D16 / D4 → scores and outputs','같은 token 입력 → BF16 / RTN / D16 / D4 → 점수와 출력')+'</li></ol><p>'+txt('D16 rescales inputs as well as weights. D4 adds activation INT4 to the same D16 weight artifact. Embeddings, attention, norms and KV precision stay on the original paths.','D16은 가중치뿐 아니라 입력에도 열 스케일을 적용합니다. D4는 같은 D16 가중치에 activation INT4를 추가합니다. Embedding·attention·norm·KV 정밀도는 원래 경로를 유지합니다.')+'</p><p>'+txt('This is a public-equation reference variant, not the authors’ complete private-library program. The quality path reconstructs BF16 weights for BF16 matrix operations.','공개 수식의 reference variant이며 저자의 비공개 라이브러리 전체 실행이 아닙니다. 품질 경로는 BF16 가중치를 복원해 BF16 행렬 연산을 수행합니다.')+'</p><div class="actions">'+a('../sources/case013-solver.py.txt',txt('Solver and adapter','변환과 adapter'))+' · '+a('../sources/case013-codec.py.txt',txt('Packed codec','Packed codec'))+' · '+a('../sources/case013-runner.py.txt',txt('Model and tool runner','모델·도구 실행기'))+'</div>')
    quality=table(['Model','Method','PPL ↓','KL ↓'],[[x['model'],x['method'],f"{x['ppl']:.3f}",f"{x['forward_KL_BF16_to_candidate']:.3f}"] for x in data['quality']],txt('All four methods in both models','두 모델의 전체 네 방법'))
    quality+=figure('ppl.svg',txt('Separate Qwen model panels of PPL for four methods','네 방법의 PPL을 Qwen 모델별로 나눈 두 panel'),txt('Lower PPL is better; descriptive subset scores, no confidence intervals.','PPL은 낮을수록 좋습니다. 일부 데이터의 관측 점수이며 신뢰구간은 없습니다.'))
    amap={(x['cohort'],x['method'],x['mode']):x for x in data['agents']}
    atable=table(['Method',txt('Base free','기본 자유 실행'),txt('16K free','16K 자유 실행')],[[m,*[f"{amap[(c,m,'FREE')]['successes']}/{amap[(c,m,'FREE')]['n']}" for c in ['base','extension16k']]] for m in ['BF16','RTN','D16','D4']],txt('Free tool tasks','자유 도구 작업'))
    atable+=figure('dev-success.svg',txt('Exact long-task and separate replay/free DEV success counts','장문 exact와 별도 replay/free DEV의 성공 수'),txt('k/n DEV counts; length variants share task IDs and are not independent samples.','DEV의 k/n입니다. 길이 변경본은 같은 task ID를 공유하며 독립 표본으로 합치지 않습니다.'))
    lmap={(x['method'],x['length']):f"{x['successes']}/{x['n']}" for x in data['long']}
    ltable=table(['Method','8K exact','16K exact','32K exact'],[[m,*[lmap.get((m,l),txt('Not run: policy','미실행: 정책')) for l in [8192,16384,32768]]] for m in ['BF16','RTN','D16','D4']],txt('Exact long-input task scores','장문 exact 점수'))
    out+=section(2,'<p>'+txt('Each model/method scores 65,504 positions in 32 × 2,048-token WikiText-2 subset windows. The same windows are reused across methods; this is not full benchmark PPL.','각 모델/방법은 WikiText-2 일부 32 × 2,048-token 구간에서 65,504위치를 채점합니다. 방법 간 같은 구간을 재사용하며 전체 benchmark PPL은 아닙니다.')+'</p>'+quality+'<h3>'+txt('Free tool use','직접 도구를 고르는 작업')+'</h3>'+atable+'<p>'+txt('Eight unique DEV tasks; the 16K follow-up reuses four of them. Replay with supplied tool results passed in every condition. It is not free execution.','고유 DEV task는 8개이며 16K 후속은 그중 4개의 길이 변경본입니다. 주어진 도구 결과를 읽는 replay는 모든 조건에서 정답이었습니다. 직접 도구를 실행하는 점수와 구분합니다.')+'</p><details><summary>'+txt('Exact long-input answers and missing cells','장문 exact 답과 미실행 조건')+'</summary>'+ltable+'<p>'+txt('Twelve unique tasks. Candidate 32K cells were skipped by a conservative memory policy after measuring 16K workspace; these are not OOM failures.','고유 task는 12개입니다. 후보 32K는 16K workspace를 관측한 뒤 보수적 메모리 정책으로 미실행했으며 OOM 실패가 아닙니다.')+'</p></details>')
    out+=section(3,'<p>'+txt('A small change in a language score can coexist with errors in tool names or action sequences. D16 used CamelCase tool names on two fixture-edit tasks, repeated rejected calls and claimed completion without the required edit.','작은 문장 점수 변화와 도구 이름·행동 오류가 함께 나타날 수 있습니다. D16은 두 파일 수정 작업에서 도구 이름을 CamelCase로 쓰고 거절된 호출을 반복한 뒤 실제 수정 없이 완료를 주장했습니다.')+'</p><p>'+txt('Post-hoc reading also separates a correct numeric expression from the frozen exact-answer contract. Primary scores were not relaxed afterwards. These small DEV observations are not formal noninferiority results.','사후 분석에서는 계산값이 맞는 표현과 미리 정한 exact 답변 계약을 구분했습니다. 이후 주 점수 기준을 완화하지 않았습니다. 작은 DEV 관측이며 formal 비열등성 결과가 아닙니다.')+'</p>')
    costs=table(['Model','Method',txt('Artifact bytes','Artifact bytes'),txt('NLL wall (s)','NLL wall (초)')],[[x['model'],x['method'],'N/A' if x['method']=='BF16' else f"{x['artifact_payload_bytes']:,}",f"{x['quality_wall_seconds']:.3f}"] for x in data['quality']],txt('Recorded storage and quality-loop costs','기록된 저장량과 품질 loop 비용'))
    out+=section(4,figure('packed-gates.svg',txt('Different tensor-relative errors; the full model exceeds its 0.02 limit','서로 다른 tensor 상대 오차이며 전체 모델은 기준0.02를 넘음'),txt('Linear scale; normalize each tensor by its own reference energy. Not a layer-error trajectory.','선형 축이며 각 tensor의 reference energy로 정규화합니다. Layer별 오차 궤적이 아닙니다.'))+'<p>'+txt('Projection artifact bytes include codes and scales, not the full checkpoint or runtime VRAM. NLL-loop wall includes measurement transfer and some I/O, not serving latency.','Projection artifact bytes는 code와 scale을 포함하며 전체 checkpoint나 실행 VRAM이 아닙니다. NLL loop wall은 계측 전송·일부 I/O를 포함하며 serving latency가 아닙니다.')+'</p><details><summary>'+txt('Recorded bytes and quality-loop times','기록된 bytes와 품질 loop 시간')+'</summary>'+costs+'</details><p>'+txt('Actual packed Marlin GEMM and one block passed. Full-model relative error was ','실제 packed Marlin GEMM과 한 block은 통과했지만 전체 모델 상대 오차는 ')+f"{100*data['packed']['relative_rmse']:.2f}%"+txt(' against a ','로, 사전 기준 ')+f"{100*data['packed']['threshold']:.0f}%"+txt(' limit. Serving speedup and capacity gains remain unestablished.','를 넘었습니다. Serving 속도·최대 문맥 이득은 아직 확인하지 못했습니다.')+'</p>')
    command='python3 -B '+C13+'/analysis/audit.py --output /tmp/case013-audit.json\npython3 -B -m unittest discover -s '+C13+'/tests -v'
    out+=section(5,'<p>'+txt('With NumPy installed, recalculate recorded NLL/PPL, predicted-ID flips, exact answers and required tool actions. No model or GPU is loaded.','NumPy로 저장된 NLL/PPL·예측 ID flip·exact 답·필요한 도구 실행 조건을 다시 계산합니다. 모델·GPU를 로드하지 않습니다.')+'</p><pre>'+escape(command)+'</pre><p>'+a('../sources/case013-reproduction-'+lang+'.md.txt',txt('Dependencies and rerun requirements','의존성과 재실행 요건'))+' · '+a('../sources/case013-audit.py.txt',txt('Independent scalar audit','독립 scalar 검산'))+' · '+a('../sources/case013-tests.py.txt',txt('CPU contract tests','CPU 계약 테스트'))+'</p>')
    out+=section(6,'<p>'+a('../sources/case013-report-'+lang+'.md.txt',txt('Full report','정식 보고서'))+' · '+a('../data/case013.json',txt('Numbers and source hashes','수치와 source hash'))+' · '+a('../sources/case013-notice.md.txt',txt('References and contributions','출처·기여'))+'</p><p>'+a('../sources/case013-method-contract.md.txt',txt('Paper/code/study contract','논문·코드·실행 계약'))+' · '+a('../sources/case013-evaluation.md.txt',txt('Scoring protocol','평가 기준'))+' · '+a('../sources/case013-failures.md.txt',txt('Failures and controls','실패·성공 사례'))+' · '+a('../sources/case013-packed-probe.md.txt',txt('Packed technical record','Packed 기술 기록'))+' · '+a('../sources/case013-roadmap.md.txt',txt('Proposed follow-up','미실행 후속 계획'))+'</p><p>'+txt('DualQuant is the authors’ method. DIOVA provides the independent INT4 variant, serialization, evaluation and inspection tools. New model runs during publication: zero.','DualQuant는 원저자의 방법입니다. DIOVA는 독립 INT4 variant·저장·평가·탐색 도구를 제공합니다. 이번 공개 정리에서 새 모델 실행은 0입니다.')+'</p>')
    return out+'<p>'+a('#overview',txt('Back to overview','한눈에 보기로 돌아가기'))+'</p>'
