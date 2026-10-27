"""Build bilingual local reports from audited facts and tables."""
import argparse,csv,json,math,shutil
from pathlib import Path
from common import read,write,identity,csvwrite

def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def text(path,body):
    with Path(path).open('x',encoding='utf-8') as f:f.write(body)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);a=p.parse_args();r=a.root;pub=r/'public_candidate';loc=read(r/'references/locations.json')
    A=read(r/'outputs/p4a_v2/p4a_facts.json');B=read(r/'outputs/p4b_v1/p4b_facts.json');C=read(r/'outputs/p4c_v1/p4c_facts.json');av=read(r/'outputs/p4a_v2/p4a_verification.json');bv=read(r/'outputs/audit_B_v1/receipt.json');cv=read(r/'outputs/p4c_v1/p4c_verification.json')
    e=next(x for x in A['groups'] if x['split']=='original_EVAL_posthoc' and x['length']==512)
    drift=rows(r/'outputs/p4a_v2/fixed_reference_drift.csv');drifts=[]
    for split in ['DEV','EVAL']:
        for n in [128,512]:
            for method in ['D1_R0','D1_R1','P_R0','P_R1']:
                d=[float(x['KL']) for x in drift if x['original_split']==split and x['length']==str(n) and x['candidate']==method]
                drifts.append({'split':'original_'+split+'_posthoc','length':n,'reference':'D0_R0','candidate':method,'documents':len(d),'mean_KL':math.fsum(d)/len(d)})
    group=rows(r/'outputs/p4c_v1/operator_grouped.csv');actual=[]
    for layer in [0,25,27]:
        d={x['comparison']:float(x['relative_RMSE']) for x in group if int(x['layer'])==layer and x['kind']=='ACTUAL_DEV'}
        ob=next(x for x in C['observations'] if x['layer']==layer)
        actual.append({'layer':layer,**d,'P_closer_documents':len(ob['P_closer_to_exact64']),'D1_closer_documents':len(ob['D1_closer_to_exact64']),'equal_documents':8-len(ob['P_closer_to_exact64'])-len(ob['D1_closer_to_exact64'])})
    pairs=rows(r/'outputs/p4b_v1/rounding_cancellation_pairs.csv');bagg=[]
    for split in ['original_DEV_posthoc','original_EVAL_posthoc']:
        own=[x for x in pairs if x['analysis_split']==split]
        bagg.append({'split':split,'cells':len(own),'individual_error_decreases':sum(float(x['individual_error_change_R1_minus_R0'])<0 for x in own),**{k:math.fsum(float(x[k]) for x in own) for k in ['R0_individual_error_sum','R1_individual_error_sum','R0_pair_gap_SSE','R1_pair_gap_SSE','R0_cross','R1_cross']}})
    led=[json.loads(x) for x in (r/'records/worker_ledger.jsonl').read_text().splitlines()];gpu=[x for x in led if x['kind']=='GPU'];gwall=math.fsum(x['parent_wall_seconds'] for x in gpu)
    peak=max(read(r/f'records/gpu-q{layer}-b{block}/data/operator_receipt.json')['memory']['peak_allocated'] for layer in [0,25,27] for block in [0,1,2])
    facts={'scope':'POSTHOC_FOLLOWUP','original_primary_verdict':'INCONCLUSIVE','A':A,'B':B,'C':C,'D0_fixed_drift':drifts,'local_energy_aggregate':bagg,'actual_projection_RMSE':actual,'new_GPU_workers':len(gpu),'new_GPU_parent_wall_seconds':gwall,'maximum_projection_allocated_bytes':peak,'original_model_revision':'c1899de289a04d12100db370d81485cdf75e47ca','A_fsum_max_difference':av['max_KL_JS_vector_fsum_difference'],'B_scalar_energy_max_difference':bv['maximum_absolute_energy_difference'],'C_scalar_dot_max_difference':cv['max_scalar_matmul_absolute_difference']}
    write(pub/'facts.json',facts);csvwrite(pub/'fixed_D0_drift_summary.csv',drifts);csvwrite(pub/'actual_projection_summary.csv',actual);csvwrite(pub/'local_energy_summary.csv',bagg)
    shutil.copytree(a.replay/'figures',pub/'figures_v2')
    grid='| D1 reference / P candidate | P_R0 | P_R1 |\n|---|---:|---:|\n'+''.join(f"| D1_R{i} | {e['cells'][str(i)+'0']:.10f} | {e['cells'][str(i)+'1']:.10f} |\n" for i in [0,1])
    jsgrid='| D1 / P | P_R0 | P_R1 |\n|---|---:|---:|\n'+''.join(f"| D1_R{i} | {e['JS_cells'][str(i)+'0']:.10f} | {e['JS_cells'][str(i)+'1']:.10f} |\n" for i in [0,1])
    contrasts='| Fixed reference | Mean KL before − after P change | Post-hoc 95% interval | Positive / negative documents |\n|---|---:|---|---:|\n'
    for mode,key in [('D1_R0','delta_P_given_D1R0'),('D1_R1','delta_P_given_D1R1')]:
        v=e[key];contrasts+=f"| {mode} | {v['mean']:+.10f} | [{v['ci_low']:+.10f}, {v['ci_high']:+.10f}] | {v['positive_documents']} / {v['negative_documents']} |\n"
    tab='| Layer | D1 / dense64 | P / dense64 | D1 / exact64 | P / exact64 | P / D1 | P closer / D1 closer to exact64 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for x in actual:tab+=f"| {x['layer']} | {x['D1_vs_dense64']:.10f} | {x['P_vs_dense64']:.10f} | {x['D1_vs_exact64']:.10f} | {x['P_vs_exact64']:.10f} | {x['P_vs_D1']:.10f} | {x['P_closer_documents']} / {x['D1_closer_documents']} |\n"
    count='| Block | Plan | Completed | Unit |\n|---|---:|---:|---|\n| P4A | 32 documents × 2 lengths | 64 | document–length rows; 320 old logits files |\n| P4A | 64 × 16 × 4 | 4096 | KL/JSD pair–position cells |\n| P4B | 32 × 2 layers × 2 Q/K × 2 R | 256 | local tensor rows, 128 paired cells |\n| P4C | 3 projections × 3 blocks | 9 | fresh operator processes |\n| P4C | 15 fixtures × 3 × 3 × 2 cores | 270 | main core calls |\n| P4C | 3 × 4 controls + 3 profiles | 15 | additional GEMM calls |\n| Input preparation | 24 native + 24 formula controls | 48 | input-norm calls; 24 beta transports |\n| Full-model input extraction | at most 8 if needed | 0 | full forwards |\n| New EVAL / P4D / generation | NOT_RUN | 0 | outside scope |\n'
    histories=f"""Original source bundle: `5876ea3c7a89ae77eda9f237059dc6101feeea834fe00df47ca9f715122841e7`.
Original Git base/HEAD: `bb2ab39ba5e9f6cea7e4c7d122fa6b26e143e688`.
Follow-up AB freeze: `{read(r/'records/freeze_AB.json')['bundle_sha256']}`.
AB computation v2: `{read(r/'records/freeze_AB_v2.json')['bundle_sha256']}`.
Final C freeze: `{read(r/'records/freeze_C_final.json')['bundle_sha256']}`.
These are separate source/config/input bundles, not new Git commits.
"""
    appendix=f"""
{count}

[Protocol AB](../protocol.json), [protocol C](../protocol_C.json), [freeze records](../records/),
[original paths and identity](../references/locations.json), [input hash references](../references/input_identity.json),
[CPU replay receipt]({a.replay/'replay_receipt.json'}), [current budget](../records/budget.json),
[preservation receipt](../outputs/close_audit/preservation_receipt.json), [verification](../outputs/close_audit/verification.json).

{histories}
"""
    ko=f"""**Case021 후속 — 더 정확하게 계산하면 두 실행 결과도 더 가까워질까?**

같은 INT4 숫자를 사용해도, 중간값을 반올림하는 위치와 계산 경로에 따라 결과가 달라집니다. 이번에는 **각 계산이 국소 참조에 얼마나 가까운지**와 **두 경로가 서로 얼마나 가까운지**를 따로 확인했습니다.

관측한 결과는 세 가지입니다. P만 FP32 중간 계산으로 바꾸면 D1_R0에서는 멀어지고 D1_R1에서는 가까워졌습니다. 기존 EVAL의 96개 국소 Q/K 조합 중 34개는 각 경로의 참조 오차가 줄어도 두 경로 간 거리가 늘었습니다. projection에서는 모든 구조화 입력의 D1/P 출력이 같았지만 실제 DEV activation에서는 작은 차이가 남았습니다.

아직 분리하지 못한 것은 실제 activation에서 남는 차이를 만드는 kernel 내부의 정확한 반올림·합산 순서입니다. 이번 결과로 원래 전체 모델 판정 **INCONCLUSIVE**를 바꾸지 않습니다. 기존 결과를 본 뒤 설계한 후속 분석이며 새 검증 문서를 사용하지 않았습니다.

GPU에서는 입력 준비 1개와 projection worker 9개를 실행했고, 새 전체 모델 forward는 0회입니다. 저장 logits와 local Q/K는 CPU로 다시 계산했습니다. [재현 안내](REPRODUCE.ko.md), [공통 수치](facts.json), [영문 보고서](FOLLOWUP_REPORT.md)에서 이어서 확인할 수 있습니다.

**무엇을 비교했는가**

D0는 원 D16 artifact의 `(codes*gamma)*alpha_inv` 순서를 유지한 dense 참조입니다. D1은 combined scale을 BF16로 만든 뒤 BF16 weight를 미리 복원합니다. P는 같은 signed codes와 BF16 scale을 Marlin으로 실행합니다. D1은 원래 미압축 모델의 정답이 아닙니다. 이번 P4C의 입력 Z에는 beta가 이미 한 번 적용되어 있으므로 core에서 다시 곱하지 않습니다.

R0는 기존 Q/K norm–RoPE 경로입니다. R1은 같은 norm weight·epsilon·native cos/sin 값으로 중간 계산을 FP32에 유지하고 마지막 Q/K를 BF16로 만듭니다. norm은 벡터 크기를 정규화하는 계산이고 RoPE는 위치를 반영하는 회전입니다. 원 R0/R1 경계는 DISTINCT이며, 이번에는 두 요소를 따로 바꾸지 않았습니다.

**P4A: 어느 참조에 가까워지는가**

아래 KL은 기존 EVAL24의 512-token 입력에서 고정 16위치를 평균하고, 문서를 같은 비중으로 평균한 값입니다. KL의 첫 인수는 D1이며 자연로그 단위 nats입니다. 전체 vocabulary 151936개를 사용했습니다.

{grid}

각 행 안에서는 D1을 고정하고 P만 R0→R1으로 바꿉니다. 양수는 그 고정 참조에 더 가까워졌다는 뜻입니다.

{contrasts}

구간은 문서 단위 paired bootstrap 10000회, seed61027025의 **사후 기술적 구간**입니다. 기존 EVAL을 새로운 미관측 확인시험으로 부르지 않습니다. `K00−K10`은 첫 인수까지 바뀌므로 같은 참조에 대한 개선량이 아닙니다.

대칭 JSD도 별도로 계산했습니다. 확률 혼합은 `(p+q)/2`이고 logaddexp로 계산했습니다.

{jsgrid}

D0_R0에 대한 drift는 [별도 표](fixed_D0_drift_summary.csv)에 있습니다. D1과 가까워지는 것과 D0_R0에 가까워지는 것은 다른 질문입니다. DEV8과 128 길이는 [전체 facts](facts.json)와 원 CSV에 별도 행으로 보존했으며, 두 길이를 독립 문서로 합치지 않았습니다. category와 모든 문서 원점도 남겼습니다.

![교차 정밀도 비교](figures_v2/01_cross_precision.png)

그림의 점은 문서별 16위치 평균이고 빨간 선은 문서 평균입니다. 왼쪽 아래는 같은 참조에서 P만 바꾼 차이입니다. 표시값은 nats에 1000을 곱한 값이며 신뢰구간 막대가 아닙니다.

원 주 비교 `K00−K11`은 {A['original_interval_replay']['mean']:.10f}, 원 seed61027023의 95% 구간은 [{A['original_interval_replay']['ci_low']:.10f}, {A['original_interval_replay']['ci_high']:.10f}]로 재생됐습니다. 0을 포함하므로 원 **INCONCLUSIVE**, tau_abs=1e-10을 그대로 유지합니다. 후속 seed의 구간은 원 구간을 대체하지 않습니다. 과거 32-token D0–P 약0.02605433과 원 0.02 gate 미통과도 유지합니다.

**P4B: 개별 오차 감소와 경로 간 거리 감소는 다를 수 있다**

기존 local replay에서 D1/P는 공통 D0_R0 block input을 받았습니다. 층25/27, Q/K 각각의 16위치 full head vector를 사용했습니다. R0/R1 projection의 raw bit 동등성 256건, cos/sin·position·norm weight·epsilon을 확인했습니다. Q는 16head, K는 8head이고 head 폭은 128입니다.

실제 projection 값을 독립 FP64 norm–RoPE 식에 넣은 결과를 `u_D,u_P`로 둡니다. 마지막 BF16 cast 전의 국소 참조입니다. `d=u_P-u_D`, `e_D=y_D-u_D`, `e_P=y_P-u_P`, `n=e_P-e_D`이면 `y_P-y_D=d+n`이고, `gap²=d²+n²+2 dot(d,n)`입니다. 음의 cross term은 이 좌표에서 두 차이가 일부 상쇄됐다는 뜻입니다.

EVAL의 96개 문서·층·Q/K 조합 모두 개별 참조 오차의 합이 줄었습니다. 그중 pair-gap은 34개 증가, 57개 감소, 5개 동일했습니다. 증가한 34개 중 24개는 R0의 cross term이 음수이고 R1보다 더 음수였습니다. 나머지 10개도 삭제하지 않았습니다. DEV의 대응값은 32개 중 증가14/감소17/동일1, 그 증가14개 중 해당 cross 패턴8개입니다. 반대 패턴인 ‘개별 오차 증가·pair-gap 감소’는 이번 좌표에서 0개였습니다.

![국소 에너지 분해](figures_v2/02_rounding_cancellation.png)

왼쪽 위 방향의 점이 개별 오차는 줄지만 경로 간 거리는 늘어난 경우입니다. 오른쪽 막대는 같은 좌표의 에너지 합이며, 평균적으로는 R1 pair-gap이 줄어드는 반대 관측도 함께 보입니다. 이 국소 항등식은 전체 모델 KL의 인과 기여율이 아닙니다. 선택된 Q/K만으로 attention이나 전체 key history를 재구성하지 않았습니다.

**P4C: 실제 projection과 두 참조**

층0/25/27의 q_proj만 사용했습니다. 모두 I1024/O2048, group64, bias 없음입니다. M512를 고정해 +1/-1 one-hot은 모든 1024축을 각각 검사했고, group 안/밖 두 항과 group64의 ±1/8 입력을 각각 512행 검사했습니다. group64는 16group 모두 32행씩입니다. 마지막 one-hot zero padding은 이 shape에서 0행입니다. 실제 입력은 기존 DEV8의 512-token D0_R0 block input에서 native input norm과 beta를 적용해 얻었습니다. 전체 모델은 재실행하지 않았습니다.

두 FP64 참조는 `Y_exact64 = Z @ (code×actual_BF16_scale).T`와 `Y_dense64 = Z @ actual_D1_BF16_weight.T`입니다. 전자는 고정 code/scale 선형 식이고 후자는 실제 D1 weight 값의 선형 식입니다. 어느 것도 모델 능력의 정답이 아닙니다. scale 역배치·signed/offset nibble·group 축·beta 한 번 적용을 검산했습니다. 실제 code×BF16 scale이 FP32에서 정확히 표현됨을 확인했고 직접 nearest-even BF16 반올림과 D1 weight 전체 bit를 대조했습니다. scalar dot 900좌표(각각 두 참조)와 FP64 행렬식도 일치했습니다.

3개 projection 모두 구조화 7fixture에서는 D1/P가 같았습니다. 실제 DEV8에서는 모두 차이가 있었습니다. 이 조건은 `MULTI_TERM_DIFFERENCE_OBSERVED`로 기록하지만, 두 항·64항 구조화 대조도 같았으므로 ‘여러 항이라는 사실 하나’나 특정 내부 누적 순서를 원인으로 확정하지 않습니다.

아래 relative RMSE는 DEV8의 같은 좌표 SSE와 reference energy를 각각 합친 뒤 제곱근을 취한 값입니다. 마지막 열만 문서별 두 참조 오차의 방향을 셉니다.

{tab}

P가 exact64에 가까운 문서는 층0에서0/8, 층25에서7/8, 층27에서4/8입니다. 반대 사례가 각각8/8,1/8,4/8이며 숨기지 않았습니다. 차이가 작은 국소 결과이며 전체 logits나 원 미압축 checkpoint보다 낫다는 뜻은 아닙니다.

![projection과 두 oracle](figures_v2/03_projection_oracles.png)

큰 점은 SSE/energy를 합친 값, 작은 점은 fixture 원점입니다. y축은 0 근처 선형 구간이 있는 symlog로 0을 보존합니다. 서로 다른 입력 종류는 값 범위와 norm이 달라 자연어 성능처럼 직접 비교할 수 없습니다. 첫 process block의 전체 BF16 출력만 보존했고 나머지 2block의 180개 출력 hash는 모두 같았습니다. 독립 모델 표본180개가 아닙니다.

**검산·비용·보존 범위**

NumPy와 독립 math.fsum KL/JSD의 최대 차이는 {av['max_KL_JS_vector_fsum_difference']:.3e}입니다. local scalar 식의 에너지 최대 차이는 {bv['maximum_absolute_energy_difference']:.3e}, projection scalar dot의 최대 차이는 {cv['max_scalar_matmul_absolute_difference']:.1f}였습니다. 원 tolerance를 유지했고 음수 잔차를 clamp하지 않았습니다. 출력 경로 덮어쓰기, source/input 변경, BF16 tie와 double-round, signed/offset/group, beta 중복, 잘못된 position/head, energy0와 KL 방향 등을 fixture로 확인했습니다.

GPU gate nonzero/예외/timeout에서 worker와 시작 journal이 생성되지 않는 회귀를 먼저 통과했습니다. 실제 gate는 compute PID 없음·utilization0·free VRAM6144MiB 이상을 한 supervisor의 spawn 성공 분기로 묶었습니다. 전후 표본만으로 연속 독점을 보장하지 않습니다. 새 GPU parent wall은 {gwall:.6f}s/1200s이고 projection allocated peak는 {peak/2**20:.3f}MiB/2048MiB였습니다. device VRAM이나 GPU busy time과 다른 수치입니다. 현재 CPU wall·process time·RSS와 원 campaign 합계는 [장부 집계](../records/budget.json)에 있습니다.

첫 P4A CPU attempt는 348.003570s에서 timeout으로 끝났습니다. 실패 source·freeze·빈 output·장부를 보존하고 동일 정규화를 bounded cache로 재사용하는 v2를 새 source freeze로 실행했습니다. 입력·수식·seed·tolerance를 바꾸지 않았습니다. 실제 projection 출력 전 C 계약을 동결했고, 최종 C freeze 당시 P4A/B 결과는 이미 알려져 있었습니다. [관측 시점](../records/observation_timeline.json)을 참고하십시오.

원 Case021의 일반 파일 931개, 원 Git 상태, 실제 사용한 원 raw 1096개, artifact3개와 선택 runtime source/binary를 종료 시 대조합니다. 모델은 작은 config/tokenizer, 존재/size와 사용한 norm vector7개를 현재 확인한 범위입니다. 전체 모델의 현재 hash를 새로 확인했다고 주장하지 않습니다. 기존 환경·다른 서비스는 변경하지 않았습니다. Python3.12.14/Torch2.13.0+cu130/Transformers5.17.0/vLLM0.29.0 GPU 환경과 Python3.11.15/NumPy1.26.4/Matplotlib3.10.9 CPU 환경을 읽기 전용으로 사용했습니다. 설치·업그레이드·다운로드는 0회입니다.

TF32와 FP16/BF16 reduced-precision reduction은 껐고 Marlin FP32 reduce=true, atomic=false입니다. 실제 RTX5080/sm120/driver621.01에서 Marlin kernel trace를 확인했습니다. 직접 작성한 것은 fixture·core 연결·감독기·FP64 감사·표시 코드입니다. Marlin kernel·Torch GEMM/allocator·HF 모형을 새로 개발한 것으로 표현하지 않습니다. Codex가 구현·분석·한영 문서화를 보조했습니다.

새 HOLDOUT, norm/RoPE 개별 분리 실행(P4D), RL, 다른 모델, 생성, 서빙, 속도 sweep, Git 게시와 사이트 배포는 NOT_RUN입니다. [P4D](P4D_DESIGN_ONLY.md)는 설계 메모만 남겼습니다. 선택할 설정이 생기더라도 새 문서에서 별도 확인해야 하며 이번 결과가 배포 권고는 아닙니다. one-hot 전체 출력은 변환 weight를 드러낼 수 있어 로컬 증거로만 보관하고 공개 후보 manifest에서는 제외했습니다.
{appendix}
"""
    en=f"""**Case021 follow-up — Does a more accurate local calculation make two execution paths agree?**

The same stored INT4 numbers can produce different results when rounding and execution differ. This follow-up separates closeness to a local FP64 expression from agreement between two execution paths.

Three observations matter. Changing only P to R1 moved it farther from D1_R0 but closer to D1_R1. In 34 of 96 existing EVAL local Q/K cells, individual reference errors decreased while the D1–P gap grew. All structured projection fixtures produced identical D1/P outputs, while actual DEV activations retained small differences.

The exact internal rounding or accumulation sequence responsible for those activation differences remains unresolved. The original full-model verdict stays **INCONCLUSIVE**. This is a post-hoc study of already observed DEV/EVAL documents, not a new holdout test.

We ran one GPU input-preparation worker and nine projection workers, with **zero new full-model forwards**. Logits and local Q/K were reanalyzed on CPU. See [reproduction](REPRODUCE.md), [shared facts](facts.json), and the [Korean report](FOLLOWUP_REPORT.ko.md).

**Execution and reference contracts**

D0 preserves the original D16 `(codes*gamma)*alpha_inv` dense reconstruction. D1 forms a BF16 combined scale and reconstructs BF16 dense weights. P executes the same signed codes and BF16 scales through Marlin. D1 is not the original uncompressed checkpoint. P4C cores receive Z after the single beta multiplication; they do not apply beta again.

R0 is native Q/K norm–RoPE. R1 retains the intermediate norm and rotary calculations in FP32, reuses the same norm weights, epsilon and native cos/sin values, and casts final Q/K to BF16. The original boundary is DISTINCT. Norm and RoPE were changed as a bundle; this follow-up does not separate their effects.

**P4A: the fixed reference matters**

The following directed KL values use all 151936 vocabulary entries, average 16 fixed positions within each 512-token document, then average the existing 24 EVAL documents equally. Units are natural-log nats; D1 is the first argument.

{grid}

Within each row only P changes from R0 to R1. Positive differences mean closer agreement with that unchanged reference.

{contrasts}

These are post-hoc descriptive paired-document percentile intervals with 10000 resamples, seed61027025. `K00−K10` changes the first argument and is not an improvement against one fixed reference. Symmetric JSD, using m=(p+q)/2 and stable logaddexp, is also retained:

{jsgrid}

[D0_R0 drift](fixed_D0_drift_summary.csv) answers a different question. Getting closer to D1 is not the same as getting closer to D0_R0. DEV8, the auxiliary 128-token prefixes, categories and every document point remain in the tables. Prefix lengths are not pooled as independent documents.

![Cross precision](figures_v2/01_cross_precision.png)

Dots are document means over the 16 positions; red lines are means across documents. Display units are nats multiplied by1000. These are not confidence-interval bars.

The original diagonal contrast replays to {A['original_interval_replay']['mean']:.10f}, with original seed61027023 interval [{A['original_interval_replay']['ci_low']:.10f}, {A['original_interval_replay']['ci_high']:.10f}]. It includes zero. The original **INCONCLUSIVE** verdict and tau_abs=1e-10 remain unchanged; the follow-up bootstrap seed does not replace the original interval. The historical 32-token D0–P relative RMSE near0.02605433 still exceeds its original0.02 gate.

**P4B: less individual error can mean a larger pair gap**

The existing local replay gives D1/P the same D0_R0 block input. We checked all256 R0/R1 projection bit comparisons and the original cos/sin, positions, norm weights and epsilon. Layers25/27 each provide Q with16heads and K with8heads, head width128, at16 selected positions.

Let u_D,u_P be the independent FP64 norm–RoPE expressions for the actual projection values, before the last BF16 cast. With d=u_P−u_D and n=(y_P−u_P)−(y_D−u_D), the identity is `y_P−y_D=d+n` and `gap²=d²+n²+2 dot(d,n)`. A negative cross term indicates cancellation at these coordinates, not an allocation of full-model causes.

All96 EVAL cells reduce the sum of individual reference errors. Pair-gap SSE increases in34, decreases in57 and is unchanged in5. Of the34 increasing cells,24 have a negative R0 cross term that is more negative than R1; the other10 are retained. DEV has32 cells:14 increases,17 decreases,1 unchanged, with8 of the14 matching that cross-term pattern. The opposite pattern of larger individual error but smaller pair gap occurs in0 cells here.

![Local decomposition](figures_v2/02_rounding_cancellation.png)

Left/up points show reduced individual error but increased pair distance. The right panel preserves the contrary aggregate observation: total local pair-gap SSE falls with R1. The decomposition is exact on the selected coordinates; it does not identify the cause of full-model KL. We do not reconstruct attention from incomplete selected Q/K history.

**P4C: three real projections and two FP64 expressions**

The fixed q_proj layers0/25/27 all have I1024/O2048, group64 and no bias. M512 is fixed. Positive and negative one-hot probes cover all1024 input axes each. Within-group pairs, across-group pairs and64-term ±1/8 rows each use512 rows; all16 groups receive32 rows in the64-term fixture. This shape needs zero padding rows:0. Actual Z comes from the existing eight DEV D0_R0 block inputs via native input norm and one beta transport. No model-wide forward was needed.

The two references are `Y_exact64=Z@(code*actual_BF16_scale).T` and `Y_dense64=Z@actual_D1_BF16_weight.T`. They describe this code/scale representation and this dense materialization, not model quality. We checked inverse scale layout, signed/offset nibbles, group axes and beta once. Every code×BF16-scale product was exactly representable in FP32, and direct nearest-even BF16 rounding matched every actual D1 weight bit. Scalar math.fsum dots checked900 row/column locations for both references.

All seven structured fixtures are D1/P-identical for all three projections. All eight actual DEV inputs show differences. The recorded label is `MULTI_TERM_DIFFERENCE_OBSERVED`, but even the two-term and64-term structured controls agree. The number of terms alone, or one particular internal accumulation order, is therefore not established as the cause.

Below, relative RMSE pools SSE and reference energy over the eight DEV inputs before taking the square root. Only the final column counts document-level directions.

{tab}

P is closer to exact64 on0/8 documents at layer0,7/8 at layer25 and4/8 at layer27. The corresponding opposite counts are8/8,1/8 and4/8. These small local differences do not establish better full logits, capabilities or fidelity to an uncompressed checkpoint.

![Projection oracles](figures_v2/03_projection_oracles.png)

Large markers pool SSE/energy; small markers show each fixture. The symlog axis has a linear region around zero and retains exact zeros. Input families differ in value range and norm. First-block full BF16 outputs are retained; all180 subsequent output hashes match. Repeated processes are not independent models.

**Verification, costs and preservation**

NumPy versus independent math.fsum KL/JSD differs by at most {av['max_KL_JS_vector_fsum_difference']:.3e}. Independent scalar local energy differs by at most {bv['maximum_absolute_energy_difference']:.3e}; projection scalar-dot versus matrix results differ by at most {cv['max_scalar_matmul_absolute_difference']:.1f}. Tolerances were not relaxed and negative residuals were not clamped. Fixtures cover overwriting, input/source mutation, BF16 ties and double rounding, nibble/axis errors, duplicate beta, position/head errors, zero energy and KL direction.

Before GPU work, fake gate nonzero, exception and timeout paths proved zero worker creation and zero start-journal writes. The actual supervisor couples a passing gate directly to spawn: no compute PID, utilization0 and at least6144MiB free VRAM. Endpoint samples do not prove uninterrupted exclusivity. New GPU parent wall is {gwall:.6f}s of1200s; projection allocated peak is {peak/2**20:.3f}MiB of2048MiB. Neither metric is device-wide VRAM or GPU busy time. [Budget](../records/budget.json) separates CPU wall/process time/RSS and the cumulative campaign.

The first P4A CPU attempt timed out after348.003570s. Its source, freeze, empty output and debit remain. Version2 reuses bounded cached normalizations with unchanged inputs, formulas, seed and tolerance under a new source freeze. The final C freeze occurred after P4A/B observations but before projection outputs; see [timeline](../records/observation_timeline.json).

The preservation audit checks all931 original Case files, original Git state,1096 actually referenced raw files,3 used artifacts and selected runtime source/binaries. Current model checks cover small config/tokenizer files, shard existence/size and7 used norm vectors, not a fresh whole-shard hash. Existing environments and services were not modified. GPU runtime: Python3.12.14, Torch2.13.0+cu130, Transformers5.17.0, vLLM0.29.0. CPU analysis: Python3.11.15, NumPy1.26.4, Matplotlib3.10.9. Installs, upgrades and downloads:0.

TF32 and FP16/BF16 reduced-precision reductions are disabled; Marlin FP32 reduce=true and atomic=false. Profiler evidence shows the actual Marlin kernel on RTX5080/sm120/driver621.01. New work consists of fixtures, core connections, supervision, FP64 audits and presentation. Marlin, Torch GEMM/allocator and the HF model remain reused implementations. Codex assisted coding, analysis and bilingual documentation.

New holdout, separate norm/RoPE execution(P4D), RL, other models, generation, serving, speed sweeps, Git publication and site deployment are NOT_RUN. [P4D](P4D_DESIGN_ONLY.md) is design only. Any future setting choice needs a separate check on new documents. Full one-hot output arrays stay local because they can reveal transformed weights; they are excluded from the public-candidate manifest.
{appendix}
"""
    text(pub/'FOLLOWUP_REPORT.ko.md',ko);text(pub/'FOLLOWUP_REPORT.md',en)
    # The two reports use the same machine-derived numeric tables.
    for block in [grid,jsgrid,contrasts,tab,count]:assert block in ko and block in en
    write(pub/'report_facts_check.json',{'shared_numeric_tables':5,'Korean_English_same_facts':True,'facts_identity':identity(pub/'facts.json'),'new_GPU_execution_in_report_builder':0})
    print(json.dumps({'status':'COMPLETE','reports':2,'facts':str(pub/'facts.json')}))
if __name__=='__main__':main()
