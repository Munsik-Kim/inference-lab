"""Small Case019 display: facts and saved projections only, no numerical SDK."""
from html import escape
import importlib.util
from common import read,sha,require

C19='cases/019-silent-alignment-dynamics'
TITLES={'ko':'손실은 그대로인데 학습은 진행될까?','en':'Learning Before the Loss Moves'}
SECTIONS=('overview','initialization','gap','intervention','method','reproduce','references')
SOURCE_COPIES={
    **{'case019-'+kind+'-'+lang+'.md.txt':C19+'/'+stem+('.ko.md' if lang=='ko' else '.md')
       for kind,stem in [('guide','README'),('report','REPORT'),('reproduce','REPRODUCE')] for lang in ['ko','en']},
    'case019-notice.md.txt':C19+'/NOTICE.md',
    'case019-source-map.md.txt':C19+'/SOURCE_MAP.md',
    'case019-facts.json.txt':C19+'/publication/facts.json',
    'case019-distribution.json.txt':C19+'/publication/distribution.json',
    'case019-mapping.json.txt':C19+'/publication/provenance/original_mapping.json',
    'case019-public-checks.py.txt':C19+'/publication/run_checks.py',
    'case019-interventions.py.txt':C19+'/src/silent_alignment/p2/core.py',
    'case019-paired.csv.txt':C19+'/publication/summary/p2/fixed_time_paired.csv',
    'case019-events.csv.txt':C19+'/publication/summary/p2/event_groups.csv',
}
FIGURES={
    'assets/case019-initialization.svg':C19+'/publication/figures/p1-01_rho_direction_modes_loss.svg',
    'assets/case019-gap.svg':C19+'/publication/figures/p1-02_gap_average_and_worst.svg',
    'assets/case019-R2.svg':C19+'/publication/figures/p2-01_R2_tail_and_persistence.svg',
    'assets/case019-effects.svg':C19+'/publication/figures/p2-04_J_G_A_paired_points.svg',
}

def load_case019(root):
    case=root/C19
    if not case.exists():return None
    spec=importlib.util.spec_from_file_location('case019_public_identity',case/'publication/verify.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);identity=mod.verify(case)
    f=read(case/'publication/facts.json');p=read(case/'publication/summary/site_projection.json')
    require(f['counts']['p1_primary']==81 and f['counts']['p2_continuations']==128,'Case019 phase counts')
    require(f['distribution']['asset_url']==identity['release_url'] and f['new_GPU_model_requests']==0,'Case019 Release location/source mismatch')
    sources=[*SOURCE_COPIES.values(),*FIGURES.values(),C19+'/publication/PUBLIC_MANIFEST.json',C19+'/publication/summary/site_projection.json']
    return {'facts':f,'projection':p,'public_identity':identity,'display_source_files':{n:sha(root/n) for n in sources}}

def link(url,label):return '<a href="'+escape(url,quote=True)+'">'+escape(label)+'</a>'
def table(headers,rows,label):
    return '<div class="scroll" tabindex="0" role="region" aria-label="'+escape(label)+'"><table><thead><tr>'+''.join('<th scope="col">'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def home_case019(data,lang):
    t=lambda k,e:k if lang=='ko' else e
    return '<section class="panel case019-entry" id="case019-entry"><h2>Case019 · '+TITLES[lang]+'</h2><p>'+t(
        '두 층 선형망과 검산 도구를 독립 구현해, 손실이 거의 변하지 않는 동안 방향이 정렬되는지 확인했습니다. 같은 예측 크기에서 방향을 바꾼 checkpoint 실험으로 초기 손실 차이와 이후 학습 진행도 나눴습니다.',
        'We independently implemented a two-layer linear network and audit tools to observe alignment before visible loss changes, then split immediate loss differences from subsequent progress after direction interventions at the same prediction size.')+'</p><p>'+t(
        '작은 초기화의 시간 분리, 큰 초기화의 순서 역전, 좁은 경계의 느린 방향과 개입 뒤의 반대 사례를 함께 보여주는 유한 차원 CPU 실험입니다.',
        'Finite-dimensional CPU results include small-initialization time separation, reversed ordering at larger initialization, slow boundary directions and intervention paths that later reverse order.')+'</p><div class="actions">'+link('case019.html#overview',t('결과와 그림','Results and figures'))+' · '+link('case019.html#intervention',t('방향 개입','Direction interventions'))+' · '+link('case019.html#reproduce',t('CPU 기록 검산','CPU recorded replay'))+'</div></section>'

def render_case019(data,lang):
    ko=lang=='ko';t=lambda k,e:k if ko else e;f=data['facts']
    names=dict(zip(SECTIONS,t(['한눈에 보기','초기화와 시간 분리','경계의 느린 방향','크기를 보존한 방향 개입','지표와 방법','CPU 기록 검산','구현과 출처'],
                            ['Overview','Initialization and time separation','Slow boundary direction','Direction interventions at fixed size','Metrics and methods','CPU recorded replay','Implementation and sources'])))
    def section(key,body):return '<section class="panel" id="'+key+'"><h2>'+escape(names[key])+'</h2>'+body+'</section>'
    def figure(asset,alt,caption):return '<figure><img loading="lazy" src="../assets/'+asset+'" alt="'+escape(alt)+'"><figcaption>'+escape(caption)+'</figcaption></figure>'
    def graph(id,label):return '<div class="c19-dynamic" hidden><svg id="'+id+'" viewBox="0 0 760 280" role="img" aria-label="'+escape(label)+'"></svg></div>'
    def option(v,label):return '<option value="'+v+'">'+escape(label)+'</option>'
    observations=t(['작은 초기화에서는 정렬과 손실 변화 시점이 갈라졌고, 큰 초기화에서는 손실 변화가 먼저였습니다.',
                    '좁은 경계에서는 평균에 작게 드러나는 느린 방향이 오래 남았습니다. R2의 긴 지연과 재하강을 별도 계산으로 확인했습니다.',
                    '같은 예측 크기에서도 방향을 바꾼 뒤 경로가 달라졌습니다. 초기 이득이 줄고, 교란 뒤 손실 순서가 뒤집히는 조건도 있었습니다.'],
                   ['Smaller initialization separated alignment and visible-loss times; with larger initialization, loss moved first.',
                    'A narrow boundary left a slow direction partly hidden by the average. Separate calculations confirmed R2’s long tail and re-exit.',
                    'Directions changed paths at the same prediction size. Some early gains shrank, and some perturbed paths later overtook the original.'])
    out='<div class="case019-layout"><nav class="case019-toc" aria-label="'+t('이 페이지 목차','On this page')+'">'+''.join(link('#'+k,names[k]) for k in SECTIONS)+'</nav><article class="case019"><h1>Case019 — '+TITLES[lang]+'</h1>'
    out+=section('overview','<p class="c19-lead">'+t(
        '손실이 거의 변하지 않는 동안에도 모형의 방향은 목표에 가까워질 수 있습니다. 두 층 선형망을 CPU로 독립 구현하고, 학습 중 예측 크기를 보존한 채 방향을 바꾸어 이후 손실 경로를 비교했습니다.',
        'A model’s directions can approach a target while loss barely moves. We independently implemented a two-layer linear network on CPU, then changed directions at saved checkpoints while preserving prediction size to compare subsequent loss paths.')+'</p><p>'+t(
        '이 페이지는 whitened 두 층 선형망의 gradient flow를 유한 차원에서 측정한 결과입니다. 원문의 방향·크기 시간 분리 일부를 수치로 재현하고, 경계·방향 개입은 자체 통제 실험으로 구분했습니다.',
        'These finite-dimensional experiments solve gradient flow in a whitened two-layer linear network. Alignment–amplitude separation is a partial numerical reproduction of the paper; boundary controls and checkpoint interventions are our controlled experiments.')+'</p><ol>'+''.join('<li>'+escape(v)+'</li>' for v in observations)+'</ol><div class="actions">'+link('#initialization',t('그림과 관측','Figures and observations'))+' · '+link('#intervention',t('무엇을 바꿨나','What changed'))+' · '+link('#reproduce',t('기록 검산하기','Replay recorded data'))+'</div>')
    first=[r for r in f['timing'] if r['g']==.4]
    init=table(t(['rho','평균 정렬 절반 시점','손실 10% 변화 시점','R 반복'],['rho','Mean alignment half-time','10% learnable-loss time','R starts']),
               [[str(r['rho']),f"{r['T_align_half']['median']:.3f}",f"{r['T_visible10']['median']:.3f}",'8'] for r in first],t('같은 R 8개의 궤적별 사건 시각 중앙값','Median of per-trajectory event times for the same R8'))
    init+='<p>'+t('표는 각 궤적의 사건 시각을 먼저 구한 중앙값입니다. 그림의 pointwise median 곡선에서 threshold를 다시 구한 값이 아닙니다. 시간 t는 실제 초가 아닌 gradient-flow 단위입니다.',
                  'The table first measures events per trajectory, then takes their median. It is not a threshold applied to a pointwise median curve. Time t uses gradient-flow units, not wall seconds.')+'</p>'
    init+='<label for="c19-rho">'+t('저장된 초기화 선택','Select a recorded initialization')+'</label><select id="c19-rho">'+''.join(option(str(v),'rho='+str(v)) for v in [.01,.0001,.000001])+'</select>'+graph('c19-rho-plot',t('R8 pointwise median 손실과 오른쪽 평균 방향 오차','R8 pointwise median loss and right mean directional error'))
    init+=figure('case019-initialization.svg','Original finite-dimensional rho comparison',t('자체 계산 그림 · g=.4 · 같은 R 8개의 방향/손실 곡선과 aligned 대조군. 원문 저자 그림이 아닙니다.','Our finite-dimensional figure, g=.4: same R8 paths and aligned controls; not an author figure.'))
    out+=section('initialization',init)
    r2=f['R2'];m=r2['large_rho_mode32']
    gap='<p>'+t(f"R2(g=.025,rho=1e-6)의 가장 느린 방향은 평균에 비해 오래 남았습니다. worst 오차 절반 시점은 {r2['worst_half']['time']:.3f}, 저장 bracket은 83.0–83.2였습니다. 전체 원 81개 경로의 별도 유한-rho graph와 선택 R2의 원 U/V RK4가 이를 확인했습니다.",
                   f"The slowest direction of R2(g=.025,rho=1e-6) persisted much longer than the average suggests. Worst-half occurred at {r2['worst_half']['time']:.3f}, within the saved 83.0–83.2 bracket. A separate finite-rho graph for all 81 original paths and selected original U/V RK4 checks corroborated it.")+'</p>'
    gap+='<label for="c19-gap">'+t('저장된 경계 gap','Recorded boundary gap')+'</label><select id="c19-gap">'+''.join(option(str(g),'g='+str(g)) for g in [.4,.1,.025])+'</select><label for="c19-gap-summary">'+t('같은 R 8개 요약 또는 R2','Same R8 summary or disclosed R2')+'</label><select id="c19-gap-summary">'+option('median',t('R8 pointwise median','R8 pointwise median'))+option('R2',t('대표 R2 한 경로','One disclosed R2 trajectory'))+'</select>'+graph('c19-gap-plot','Right mean and worst squared directional errors')
    gap+='<p>'+t(f"큰 rho=.01의 같은 R2 mode32는 {m['up_entries'][0]['time']:.3f}에 기준을 넘고 {m['down_exits'][0]['time']:.3f}에 다시 내려왔습니다. gradient-flow 시간 10단위 동안 grid상 유지한 첫 구간은 {m['first_sustained10']['start']:.3f}–{m['first_sustained10']['end']:.3f}였습니다. P1의 최초 crossing과 P2의 사후 유지 진단은 나란히 남겼습니다.",
                   f"At larger rho=.01, R2 mode32 entered the threshold at {m['up_entries'][0]['time']:.3f}, exited at {m['down_exits'][0]['time']:.3f}, and first retained it over 10 gradient-flow units on the saved grid at {m['first_sustained10']['start']:.3f}–{m['first_sustained10']['end']:.3f}. Original first crossing and posthoc persistence remain separate.")+'</p>'+figure('case019-R2.svg','R2 original modes and corroborating graph/reference checks',t('R2 한 경로 · 첫 진입·재이탈·10단위 유지와 긴 tail. 수치 참조는 별도 계산이며 새로운 dataset 반복이 아닙니다.','One R2 trajectory: entry, re-exit, 10-unit retention and long tail. Numerical references are separate calculations, not new datasets.'))
    out+=section('gap',gap)
    arm=table(t(['arm','바뀐 것과 보존한 것'],['Arm','What changes and what is preserved']),[
        ['CONTINUE',t('저장 U/V 그대로 새 적분','New integration from unchanged saved U/V')],
        ['GAUGE',t('숨은 좌표 공동 회전: Z 그대로, Gram은 직교 conjugacy','Joint hidden rotation: same Z; Grams follow orthogonal conjugacy')],
        ['ALIGN',t('target SVD를 아는 oracle 정렬: 실제 두 Gram·예측 singular value 보존','Oracle target-SVD alignment; preserves both actual Grams and prediction singular values')],
        ['ROTATE',t('target 고정 ambient 회전: 실제 두 Gram·예측 singular value 보존','Ambient rotation with fixed target; preserves both actual Grams and prediction singular values')]],'Four checkpoint intervention definitions')
    arm+='<p>'+t('기존 16개 P1 경로의 t*=1/5를 나눠 32 parent, 네 arm의 128개 continuation을 풀었습니다. 같은 R 8개·target과 하나의 고정 회전 triple을 공유합니다. ALIGN의 즉시 손실 하한은 oracle 구성의 성질이고, 그 뒤 경로는 실제 적분 결과입니다.',
                 'Sixteen original P1 paths provide 32 parents at t*=1/5 and 128 continuations. They share R8, targets and one fixed rotation triple. ALIGN’s immediate loss bound follows from its oracle construction; subsequent paths come from actual integrations.')+'</p><label for="c19-parent">'+t('저장된 parent 조건','Recorded parent condition')+'</label><select id="c19-parent">'+''.join(option(f'{g}/{st}',f'g={g}, t*={st}') for g in [.4,.025] for st in [1,5])+'</select>'+graph('c19-arm-loss','Four-arm R8 median learnable loss residual')+graph('c19-arm-worst','Four-arm R8 median worst directional error')
    arm+='<p class="c19-legend">CONTINUE — solid · GAUGE — dashed · ALIGN — dotted · ROTATE — dash-dot</p><dl><dt>J</dt><dd>'+t('방향을 바꾼 직후의 손실 차이','Loss difference immediately after transformation')+'</dd><dt>G(τ)</dt><dd>'+t('그 뒤 CONTINUE 대비 추가 누적 감소량의 차이','Difference in subsequent cumulative loss decrease versus CONTINUE')+'</dd><dt>A(τ)=J+G(τ)</dt><dd>'+t('해당 시점 총 차이. 양수이면 arm 손실이 더 낮음','Total difference at that time; positive means lower arm loss')+'</dd></dl><p>'+t('분모 D는 parent의 원래 학습 가능 손실로 모든 arm에 같습니다. G는 학습속도 향상률이 아닙니다.','D is the parent’s original learnable loss, shared across arms. G is not a learning-speed improvement rate.')+'</p><label for="c19-tau">'+t('저장된 비교 시점 τ','Recorded comparison time τ')+'</label><select id="c19-tau">'+''.join(option(str(v),'τ='+str(v)) for v in [5,20,155])+'</select><div id="c19-effects" aria-live="polite"></div>'
    ex=f['example'];arm+='<p>'+t(f"예를 들어 g=.025,t*=5,τ=5에서 8개 짝 차이의 평균 J/D={ex['mean_J_over_D']:.8f}, G/D={ex['mean_G_over_D']:.8f}, A/D={ex['mean_A_over_D']:.8f}였습니다. 총 차이는 원래 학습 가능 손실의 약 {ex['percentage_original_learnable_loss_difference']:.4f}%이며 ‘더 빠른 비율’이 아닙니다.",
                  f"For g=.025,t*=5,τ=5, the means of eight paired differences are J/D={ex['mean_J_over_D']:.8f}, G/D={ex['mean_G_over_D']:.8f}, A/D={ex['mean_A_over_D']:.8f}: about {ex['percentage_original_learnable_loss_difference']:.4f}% of original learnable loss, not a speedup percentage.")+'</p><p>'+t(
        'ROTATE는 τ=5의 32 parent 모두에서 손실이 더 높았지만, g=.025,t*=1,τ=20은 5개가 더 낮고 3개가 더 높았습니다. t*=5에서는 2개가 더 낮고 6개가 더 높았습니다. g=.4에서는 장기 차이가 사라졌고 g=.025에서는 유한 관측 기간의 잔여 차이가 남았습니다.',
        'ROTATE had higher loss for all 32 parents at τ=5, but at g=.025,t*=1,τ=20 five had lower and three higher loss. At t*=5, two had lower and six higher. Long-term differences vanished at g=.4; finite-horizon residual differences remained at g=.025.')+'</p>'+figure('case019-effects.svg','Paired J,G,A seed values and median marks','Same-parent R8 paired points; raw CSV contains both gaps and all fixed times.')
    out+=section('intervention',arm)
    out+=section('method','<p>'+t('손실 L=.5‖UVᵀ−C‖²은 고정 목표와의 제곱 오차입니다. rank 제한 바닥 L_min을 뺀 residual을 표시합니다. rho는 V₀의 진폭 √(2rho)와 구분하고, signed mode lᵢᵀZrᵢ는 정렬된 singular value와 구분합니다.',
        'Loss L=.5‖UVᵀ−C‖² is squared error to a fixed target. Displayed residual subtracts the rank-constrained floor L_min. rho differs from initial amplitude √(2rho); signed mode lᵢᵀZrᵢ differs from a sorted singular value.')+'</p><p>'+t(
        'mean 방향 오차는 principal direction들의 평균, worst는 가장 큰 오차입니다. half 도달은 완전 정렬이 아니며 미도달은 null로 둡니다. boundary-pair는32·33번째 값만 .6±g/2로 바꿔 나머지62개와 basis를 유지하지만, 경계 amplitude·target norm·L_min도 변합니다.',
        'Mean directional error averages principal directions; worst follows the largest error. Halving is not complete alignment; non-reached values remain null. Boundary-pair changes only values 32/33 to .6±g/2, keeping the other 62 and bases fixed while boundary amplitudes, target norm and L_min also change.')+'</p><p>'+t(
        'P0 17개 pilot, P1 random 72+고유 aligned 9, P2 128개 branch는 서로 다른 계산 분모입니다. 같은 target·R·parent를 공유하므로 독립 dataset226개로 합치지 않습니다. 방향 개입은 시작 loss/gradient도 바꾸며, 이 유한 선형계의 경로를 관측했습니다.',
        'P0’s 17 pilots, P1’s 72 random plus nine unique aligned paths, and P2’s 128 branches have distinct computational denominators. Shared targets/R/parents prevent pooling them as 226 independent datasets. Direction interventions also change starting loss/gradient; observations concern this finite linear system.')+'</p>')
    command='python3 -I -B cases/019-silent-alignment-dynamics/publication/run_checks.py --output /tmp/case019-quick-new --mode quick --tests'
    out+=section('reproduce','<p>'+t('고정 Python3.11.15/NumPy1.26.4/SciPy1.17.1/Matplotlib3.10.9 CPU 환경에서 실행합니다. quick는 공개 작은 slice, full은 공개 소스와 별도 evidence의 저장자료를 검산합니다. 새 full ODE sweep은 실행하지 않습니다.',
        'Run in the pinned Python3.11.15/NumPy1.26.4/SciPy1.17.1/Matplotlib3.10.9 CPU environment. Quick checks the public slice; full checks stored records from source plus separate evidence. No new full ODE sweep is run.')+'</p><pre><code>'+escape(command)+'</code></pre><p>'+link('../sources/case019-reproduce-'+lang+'.md.txt',t('환경·quick/full 명령','Environment and quick/full commands'))+' · '+link('../sources/case019-distribution.json.txt',t('배포 구조와 asset 정보','Distribution and asset metadata'))+'</p><p>'+t(
        f"전체 evidence는 약 {f['distribution']['asset_bytes']/1e6:.1f} MB의 전용 Release asset입니다. 사용자가 직접 다운로드하고 bytes·SHA256을 확인한 파일을 --evidence로 지정합니다. 사이트는 큰 자료를 자동 다운로드하거나 복사하지 않습니다.",
        f"Full evidence uses a roughly {f['distribution']['asset_bytes']/1e6:.1f} MB dedicated Release asset. Download explicitly, verify bytes/SHA256 and pass the file with --evidence. This site neither downloads nor copies that large payload.")+'</p><p>'+link(f['distribution']['asset_url'],t('전체 기록 evidence 다운로드','Download full recorded evidence'))+'</p><p><code>SHA256 '+escape(f['distribution']['asset_sha256'])+'</code></p><details><summary>'+t('검산 범위와 원형 이력','Replay scope and historical records')+'</summary><p>'+t(
        '전체 기록은 P0 17×1002/136 checkpoint, P1 81×1702/891, P2 128×1677/1920을 검산합니다. 저장하지 않은 full factor 경로는 재구성하지 않고 GAUGE error series와 남은 checkpoint를 확인합니다. P0 최초 실패, P1 launcher 오류, P2 사후 R2 선택과 helper 작성 시점은 원형 이력으로 남아 있습니다.',
        'Full replay checks P0 17×1002/136 checkpoints, P1 81×1702/891 and P2 128×1677/1920. Unsaved full factor paths are not reconstructed; retained GAUGE error series/checkpoints are inspected. Original P0 failure, P1 launcher error and P2 posthoc R2/helper timing remain historical records.')+'</p>'+link('../sources/case019-mapping.json.txt','Original byte mapping')+'</details>')
    out+=section('references','<p>'+t('직접 구현한 ODE·두 층 선형 환경·초기화·지표·대조군·개입·검산 도구와, NumPy/SciPy/Matplotlib이 제공한 선형대수·수치 적분·그림을 구분합니다. Codex 보조 구현이며 저장소 기존 라이선스를 따릅니다.',
        'Our implementation covers the ODE, initialization, metrics, controls, interventions and audits. NumPy/SciPy/Matplotlib provide linear algebra, numerical integration and figures. Development used Codex assistance under the existing repository license.')+'</p>'+link('https://arxiv.org/abs/2610.29864v1','Martin, Bach, Biroli · Silent Alignment v1')+' · '+link('../sources/case019-source-map.md.txt','Source/statement map')+' · '+link('../sources/case019-report-'+lang+'.md.txt',t('상세 보고서','Detailed report'))+' · '+link('../sources/case019-notice.md.txt','Notice'))
    out+='<noscript><p>'+t('JavaScript가 꺼져 있습니다. 위 결과·정적 그림·표·원문과 검산 안내는 그대로 읽을 수 있습니다. 선택기는 저장된 조건의 보조 탐색 기능입니다.',
        'JavaScript is disabled. Results, static figures/tables, sources and replay instructions remain readable. Selectors provide optional exploration of recorded conditions.')+'</p></noscript></article></div>'
    return out
