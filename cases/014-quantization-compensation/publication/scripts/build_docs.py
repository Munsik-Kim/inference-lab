"""Generate unified bilingual entry/report documents from the verified display data."""
import argparse,json,math,re
from pathlib import Path

def english_spacing(s):
    s=re.sub(r'\b(on|from|all|with|at|by|over|in|of|is|has|per|for|and|lower|crosses)(?=\d)',r'\1 ',s)
    s=re.sub(r'(\d)(?=(?:books?|documents?|targets?|projections?|methods?|tokens?|comparisons?|paired|fixed|stored|seeds?|files?|intervals?|conditions?|groups?|resamples?|points?|bytes?|whole|models?|strengths?|polic(?:y|ies)|positions?|independent)\b)',r'\1 ',s)
    return s.replace('This project ridge/RTN reference','This ridge/RTN reference')

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args();c=a.case;d=json.loads((c/'publication/summary/project.json').read_text());p2=d['p2'];primary=p2['primary'];M={(x['split'],x['method']):x for x in p2['methods']};R=p2['resources']
    title={'ko':'같은 INT4 저장량으로 모델 품질 개선하기','en':'Improving Model Quality at the Same INT4 Storage Budget'}
    intro={'ko':'양자화된 모델의 실제 입력 경로를 따라 가중치를 보정하고, 같은 INT4 형식으로 저장하는 도구를 구현했습니다. Qwen3-0.6B의196개 Transformer projection에 고정된 약한 보정을 적용해, 새 PG19 64책의256-token 평가 구간에서 RTN 대비 PPL을'+f"{abs(100*M['EVAL-ID','COMP_QUARTER']['PPL_relative_change']):.2f}%"+' 낮췄습니다. Projection 저장량은 그대로 유지했습니다.',
    'en':'Implemented a path-aware compensation pipeline that refits projection weights and stores them in the same INT4 format. On Qwen3-0.6B, fixed quarter-strength compensation reduced perplexity by '+f"{abs(100*M['EVAL-ID','COMP_QUARTER']['PPL_relative_change']):.2f}%"+' versus RTN on256-token excerpts from64 fresh PG19 books, with unchanged projection storage.'}
    scope={'ko':'ArXiv에서는 같은 이득이 확인되지 않았습니다. 이 결과는 저장본을 BF16으로 복원한 품질 평가이며, 실행 속도·VRAM 절감은 별도로 측정하지 않았습니다.','en':'The gain did not transfer to the ArXiv evaluation. Quality was measured after dense BF16 reload; serving speed and VRAM savings were not measured.'}
    def table(split):
        rows=['| Method | NLL ↓ (nats/token) | PPL ↓ | KL(BF16∥method) ↓ | Argmax flip |','| --- | ---: | ---: | ---: | ---: |']
        for method in ['BF16','RTN','COMP_QUARTER','COMP_HALF']:
            r=M[split,method];rows.append(f"| {method} | {r['NLL']:.6f} | {r['PPL']:.6f} | {r['mean_KL']:.6f} | {r['flip_rate']*100:.3f}% |")
        return '\n'.join(rows)
    paired=['| Cohort | Candidate−RTN | ΔNLL (nats/token) | 95% paired CI | PPL change |','| --- | --- | ---: | --- | ---: |']
    for r in p2['contrasts']:
        if r['baseline']=='RTN':paired.append(f"| {r['split']} | {r['candidate']} | {r['mean_document_delta_NLL']:+.6f} | [{r['CI95'][0]:+.6f}, {r['CI95'][1]:+.6f}] | {r['PPL_relative_change']*100:+.3f}% |")
    paired='\n'.join(paired)
    p1table=['| Target | CAL alpha | EVAL selected ΔSSE | EVAL alpha0.25 ΔSSE | T_inherited | T_cancellation | T_requant |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in d['p1']['targets']:p1table.append(f"| {r['target'].replace('model.layers.','')} | {r['selected_alpha']:g} | {r['selected_SSE_change']*100:+.2f}% | {r['quarter_SSE_change']*100:+.2f}% | {r['T_inherited']:.2e} | {r['T_cancellation']:.2e} | {r['T_requant']:.2e} |")
    p1table='\n'.join(p1table);ch=['| Target | OLD_FIT | SELECT_SSE | SELECT_NLL |','| --- | ---: | ---: | ---: |']
    for t,v in sorted(d['p1b']['choices'].items()):ch.append(f"| {t.replace('model.layers.','')} | {v['OLD_FIT']:g} | {v['SELECT_SSE']:g} | {v['SELECT_NLL']:g} |")
    ch='\n'.join(ch);bmac=[x for x in d['p1b']['macro_contrasts'] if x['split']=='FRESH-EVAL-ID' and x['candidate']=='SELECT_SSE' and x['baseline'] in ['OLD_FIT','BASE_ZERO','FIXED_QUARTER']]
    btable=['| SELECT_SSE versus | ΔSSE / alpha0 SSE, target mean | ΔNLL, target mean | NLL95% pointwise CI |','| --- | ---: | ---: | --- |']
    for x in bmac:btable.append(f"| {x['baseline']} | {100*x['D_macro']:+.3f} baseline-SSE pp | {x['NLL_macro_doc_mean_difference']:+.6f} | [{x['NLL_macro_CI95'][0]:+.6f}, {x['NLL_macro_CI95'][1]:+.6f}] |")
    btable='\n'.join(btable)
    for lang in ['ko','en']:
        ko=lang=='ko';suffix='.ko' if ko else '';other='README.md' if ko else 'README.ko.md';rep='REPORT'+suffix+'.md';txt=lambda k,e:k if ko else e
        readme=f"""# Case 014 — {title[lang]}

[{'English' if ko else '한국어'}]({other}) · [Pages](https://munsik-kim.github.io/inference-lab/{lang}/case014.html#overview)

{intro[lang]}

**{scope[lang]}**

| {'실제 비교' if ko else 'Measured comparison'} | RTN PPL | QUARTER PPL | {'변화' if ko else 'Change'} |
| --- | ---: | ---: | --- |
| PG19 ·64 {'책의 구간' if ko else 'book excerpts'} | {M['EVAL-ID','RTN']['PPL']:.3f} | {M['EVAL-ID','COMP_QUARTER']['PPL']:.3f} | {M['EVAL-ID','COMP_QUARTER']['PPL_relative_change']*100:+.2f}%; NLL64/64 {'개선' if ko else 'lower'} |
| ArXiv ·16 content groups | {M['SHIFT','RTN']['PPL']:.3f} | {M['SHIFT','COMP_QUARTER']['PPL']:.3f} | {M['SHIFT','COMP_QUARTER']['PPL_relative_change']*100:+.2f}% {'점추정, NLL 구간0포함' if ko else 'point estimate; NLL CI crosses0'} |

**{'동일 INT4 projection 저장량' if ko else 'Identical INT4 projection storage'}:247,733,136 bytes/method.**
{'원 BF16의 PG19 PPL은' if ko else 'Original BF16 PG19 PPL is'} {M['EVAL-ID','BF16']['PPL']:.3f}.

[{'결과 비교하기' if ko else 'Compare results'}]({rep}#results) ·
[{'구현과 재현' if ko else 'Implementation and reproduction'}](REPRODUCE.md) ·
[{'P0→P2 연구 과정' if ko else 'P0→P2 research journey'}](STAGE_MAP.md)

## {'구현한 기능' if ko else 'Implemented pipeline'}

{'실제 입력 확인 → 약한 보정 적합 → INT4 저장 → 새 프로세스 재로딩 → 새 문서의 다음 토큰 점수 비교.' if ko else 'Capture actual inputs → fit weak compensation → store INT4 → reload in a new process → compare next-token scores on unused documents.'}

- [Native capture / own-trajectory runner](followups/p2_core/source/run_p2.py)
- [CPU FP64 full-input fitting](followups/p2_core/source/p2_fit.py)
- [INT4 codec](scripts/codec.py) · [Independent decoder](followups/p2_core/source/independent_codec.py)
- [Three-term decomposition](scripts/decompose.py) · [Raw-to-display exporter](publication/scripts/derive.py)
- [CPU tests](followups/p2_core/tests/test_p2.py) · [Public checks](publication/scripts/run_checks.py)

## {'범위와 출처' if ko else 'Scope and sources'}

{'P0의 대수·codec, P1의 독립 표적 개입, P1b의 독립 강도 선택, P2의 자체 경로 전층 보정을 한 사례에 묶었습니다. 각 단계의 부정 결과와 원형·선택·판정은 보존했습니다.' if ko else 'One case connects P0 algebra/codec, P1 independent target interventions, P1b independent strength selection and P2 whole-model own-trajectory compensation. Negative results, original files, choices and decisions remain preserved.'}

{'구현은 Guo의 선행 연구를 참고한 ridge/RTN reference입니다. 공식 QEP 전체 재현이나 새 quantizer가 아닙니다. 자체 구현과 upstream 역할은' if ko else 'This project ridge/RTN reference builds on Guo’s prior work; it is not complete official QEP reproduction or a new quantizer. See'} [PORTFOLIO](PORTFOLIO.md), [NOTICE](NOTICE.md), [source audit](references/source_audit.json).
"""
        (c/('README'+suffix+'.md')).write_text(english_spacing(readme) if not ko else readme)
        labels=['배경','질문과 가설','이론과 구현 계약','방법','실험','실험 결과','결과 분석','결론','재현과 참고자료'] if ko else ['Background','Questions and hypotheses','Theory and implementation contract','Methods','Experiments','Results','Analysis','Conclusion','Reproduction and sources']
        anchors=['background','questions','contract','methods','experiments','results','analysis','conclusion','references']
        body=f"# Case 014 — {title[lang]}\n\n"+txt('양자화 오차 보정의 상쇄·재양자화·일반화 분석','Quantization Compensation: Cancellation, Re-quantization, and Generalization')+f"\n\n[{('English' if ko else '한국어')}]({'REPORT.md' if ko else 'REPORT.ko.md'}) · [Stage map](STAGE_MAP.md) · [CPU/GPU reproduction](REPRODUCE.md)\n\n{intro[lang]}\n\n{scope[lang]}\n\n"+' · '.join(f'[{i+1}. {t}](#{key})' for i,(t,key) in enumerate(zip(labels,anchors)))+'\n'
        sections=[]
        sections.append(txt('작은 가중치 오차가 언제 실제 출력 개선으로 이어지는지 조사했습니다. 앞선 층에서 전달되는 입력 오차를 보정하더라도 기존 상쇄·재양자화·새 문서의 변화가 이득을 바꿀 수 있습니다. P1의 실패가 P1b의 독립 선택을, 그 관측이 P2의 고정 약한 전층 비교를 동기화했습니다. 전체가 처음부터 정해진 단일 사전등록 실험은 아닙니다.',
          'The study asks when correcting inherited input error improves actual outputs. Existing cancellation, re-quantization and new documents can change the benefit. P1 motivated P1b independent selection; those observations motivated P2 fixed weak full-model comparisons. This sequence was not one experiment preregistered from the outset.'))
        sections.append(txt('Q1: 상속오차 개선이 유리한 상쇄를 깨뜨리는가? Q2: 저장 scale/code를 다시 계산하면 방향이 달라지는가? Q3: 보정량과 강도의 선택이 새 문서에 전이되는가? P2의 별도 주 질문은 같은 형식의 전층 QUARTER가 전체 RTN보다 새 PG19 NLL을 낮추는가입니다. HALF는 보조이며, 결과를 보고 주 후보를 바꾸지 않았습니다.',
          'Q1: can inherited-error correction destroy useful cancellation? Q2: does recalculating stored scales/codes reverse a benefit? Q3: do fitted corrections and selected strengths transfer to unused documents? P2 separately registers whole-model QUARTER minus RTN on fresh PG19 gold NLL as primary. HALF is auxiliary and does not replace the primary after observing results.'))
        formula='''W[out,in], X/Z[in,tokens], Wstar=W+alpha*K
A=W(X-Z); C=alpha*KZ
B=(What-Wstar)Z; B0=(What0-W)Z
R=A-C-B; R0=A-B0
Delta = ||R||²-||R0||²
T_inherited = ||C||²-2<A,C>
T_cancellation = 2<B0,C>
T_requant = ||B-B0||²-2<R0-C,B-B0>
Delta = T_inherited+T_cancellation+T_requant'''
        sections.append('```text\n'+formula+'\n```\n\n'+txt('CAL에서 full-input Gram G=ZZᵀ, λ=0.001 trace(G)/d_in의 CPU FP64 Cholesky·triangular solve로 K를 적합합니다. 명시 inverse·block-diagonal 근사는 없습니다. Ridge H는 일반적으로 projector가 아닙니다. 평가에서는 CAL의 K를 고정하며 AP_eval로 다시 적합하지 않습니다.\n\n고정-B 반사실 A−C−B0와 실제 고정밀 Wstar control은 다른 대상입니다. 저장 What의 FP32 표현값을 FP64로 올린 대수와 실제 BF16 출력 차이·weight cast·arithmetic residual을 분리합니다. P2의 B0는 각 방법의 현재 Z 위의 국소 RTN 대조이며, 전체 RTN 모델 오차가 아닙니다.',
          'CAL fitting uses the full-input Gram G=ZZᵀ and λ=.001 trace(G)/d_in with CPU FP64 Cholesky/triangular solves. No explicit inverse or block-diagonal approximation is used. Ridge H is generally not a projector. CAL K stays fixed on evaluation; AP_eval is not refitted.\n\nThe fixed-B counterfactual A−C−B0 differs from actually executing a high-precision Wstar control. Algebra uses saved FP32 What lifted to FP64; BF16 module errors, weight casts and arithmetic residuals are separate. P2 B0 is local RTN on each method’s current Z, not the separate whole-RTN model error.'))
        sections.append(txt('공통 encoder는 signed INT4[−8,7], input group64, output-row/group FP32 absmax/7, nearest-even, 진짜 zero group만 scale1, low-nibble-first, header36B입니다. pack→디스크→독립 decode→BF16 설치를 확인했습니다. Runtime에 K를 별도 GEMM으로 더하지 않습니다. Embedding/lm_head tie·norm·attention·RoPE 등 공통 경로는 유지합니다.\n\nP2는 매 그룹마다 native 전체 CausalLM forward로 실제 입력을 캡처합니다. q/k/v → 실제 attention → o → residual/norm → gate/up → SiLU 곱 → down 순서입니다. 같은 입력 그룹만 Gram factor를 공유하며 QUARTER/HALF는 각자 원형에서 시작해 새로운 Z·K를 만듭니다.',
          'The common encoder is signed INT4[−8,7], input group64, per-output-row/group FP32 absmax/7, nearest-even, unit scale only for true-zero groups, low-nibble-first, header36B. Pack→disk→independent decode→BF16 installation is checked. K is not a runtime GEMM. Common embedding/lm_head ties, norms, attention and RoPE are preserved.\n\nP2 captures actual inputs with the native entire CausalLM forward after each group: q/k/v → actual attention → o → residual/norm → gate/up → SiLU product → down. Only shared-input groups share Gram factors. QUARTER/HALF independently start from original weights and create their own Z/K.'))
        sections.append(txt('단계별 문서·선택·원형은 [STAGE_MAP](STAGE_MAP.md)에 있습니다. P1: CAL32/DEV16/EVAL32/SHIFT16,6표적×5alpha. P1b: 기존 K6개와 저장본30개를 유지하고 새 SELECT32/EVAL64/SHIFT16. P2: 원 CAL32 재사용, 별도의 새 PG19 64책·ArXiv16 content group,196개 projection 전체 비교입니다. 문서와 원문 창의 과거 사용은 확인 가능한 manifest 합집합에서 제외했습니다. Pile는 article ID가 없어 content-group 기준입니다. 사전학습 미노출을 주장하지 않습니다.\n\n모든 LM 점수는 chat/BOS/EOS 추가 없는256 input tokens의 다음255원문 tokens입니다. KL은 고정16위치의 전체 어휘 BF16∥후보, argmax flip은255위치입니다. P2는 BF16/batch1/SDPA/use_cache=False/전체256 head/TF32·reduced reduction OFF입니다. 모든 artifact 동결 당시 fresh forward0회였고, 그 후 평가했습니다.',
          'Stage documents, choices and originals are mapped in [STAGE_MAP](STAGE_MAP.md). P1 uses CAL32/DEV16/EVAL32/SHIFT16 with6 targets×5 strengths. P1b keeps6 K matrices and30 artifacts, using new SELECT32/EVAL64/SHIFT16. P2 reuses original CAL32 but evaluates separate fresh PG19 64 books and ArXiv16 content groups with all196 projections. Known prior inputs are excluded by manifest union; Pile has content-group identity without article IDs. Pretraining non-exposure is not established.\n\nAll LM metrics use256 input tokens without chat/BOS/EOS and255 original next-token targets. KL uses16 fixed full-vocabulary BF16∥candidate positions; argmax flips use255. P2 uses BF16/batch1/SDPA/use_cache=False/full256 head/TF32 and reduced reduction OFF. Every artifact was frozen at zero fresh forwards, followed by evaluation.'))
        results=txt('### P2 — 全층 모델의 대표 결과\n','### P2 — Whole-model results\n').replace('全층','전층')+'\nPG19 ·64 book excerpts ·16,320 valid next tokens/method\n\n'+table('EVAL-ID')+'\n\nArXiv ·16 content groups ·4,080 valid next tokens/method\n\n'+table('SHIFT')+'\n\n'+paired+'\n\n'+txt('주 비교의 PPL 감소는 RTN 대비12.92%이며 정답률·속도 증가가 아닙니다. 64/64는 각 책의 고정 구간 NLL이 낮았다는 뜻입니다. PG19 QUARTER의 RTN 추가 NLL 손실 회복은58.17%이며 상세 진단 분모는 RTN−BF16 gap입니다. 원 BF16을 능가한 결과는 아닙니다.\n\n원래 seed6101523으로10,000회 문서 paired bootstrap을 재계산했습니다. 구간은 이 모델·CAL32·고정 강도에 조건화합니다. HALF/SHIFT는 보조 pointwise입니다. HALF−QUARTER의 새 게시 사후 점추정은 PG19−0.003356, ArXiv+0.092972 nats/token입니다. 독립 검토 code/seed가 없어 해당 과거 CI를 붙이지 않았습니다.',
          'The12.92% primary PPL reduction is relative to RTN, not accuracy or speed.64/64 means lower NLL on each fixed book excerpt. QUARTER recovers58.17% of RTN’s extra NLL gap above BF16 on PG19; it does not beat BF16.\n\nThe original seed6101523 and10,000 paired document bootstrap resamples are recomputed. Intervals condition on this model/CAL32/fixed strengths. HALF/SHIFT are auxiliary pointwise. Publication post-hoc HALF−QUARTER point estimates are−.003356 on PG19 and+.092972 nats/token on ArXiv. Reviewer code/seed were unavailable, so unsupported historical CIs were not attached.')
        results+='\n\n![PG19](figures/p2_r1/01_pg19.svg)\n\n![ArXiv](figures/p2_r1/02_arxiv.svg)\n\n'+txt('### P0→P1→P1b의 관측\n\nP0의 고정 오차 유리수 예,2bit4-code 예, 무한 격자 dither는 서로 다른 모형입니다. dither는 ridge0·clipping 없음·조건당2,000개 저장 seed의 양성 대조입니다. 이번 게시에서는 저장 SSE/SE를 검산했으며 Monte Carlo를 다시 돌리지 않았습니다.\n\nP1의 CAL 선택은 EVAL 국소 SSE를6/6에서 높였고, 사후 확인한0.25는6/6에서 낮췄습니다. 아래 합산 항의 단위는 해당 표적의32문서 local squared energy입니다. 연속 보정 전이 손실이 큰 양의 항이며 상쇄를 주원인으로 단정하지 않습니다.',
          '### P0→P1→P1b observations\n\nP0 rational fixed-error examples,2bit4-code examples and infinite-lattice dither are distinct models. Dither is a ridge0/no-clipping positive control with2,000 stored seeds per condition. Publication recalculates saved SSE/SE; Monte Carlo is not rerun.\n\nP1 CAL-selected strengths increased EVAL local SSE in6/6 targets; post-hoc quarter strength reduced it in6/6. The following sums are local squared energies over32 documents for that target. Continuous transfer loss is the large positive term; cancellation is not established as the main cause.')+'\n\n'+p1table+'\n\n'+txt('P1b는 기존 K와 저장본을 유지하고 선택 문서를 분리했습니다. 여섯 독립 개입의 비율 평균이며 하나의 전층 모델 결과가 아닙니다.','P1b separates selection documents while retaining K and artifacts. These are mean ratios over six independent interventions, not one whole-model result.')+'\n\n'+ch+'\n\n'+btable+'\n\n'+txt('고정0.25 대비 추가 SSE 이득은 block4/20 o_proj의0.5 선택에 집중됐고 나머지 네 표적은 같은0.25 artifact입니다. 추가 NLL 이득의 구간은0을 포함합니다. ArXiv에서는 두 selector 모두 alpha0보다 평균 SSE/NLL이 악화했습니다.','Extra local SSE benefit over fixed quarter is concentrated in block4/20 o_proj selecting.5; the other four targets alias the same quarter artifact. The additional NLL CI crosses0. Both selectors worsen mean SSE/NLL against alpha0 on ArXiv.')
        sections.append(results)
        sections.append(txt('국소 SSE를 줄이는 것과 실제 다음 토큰 NLL을 낮추는 것은 다른 목표입니다. P1b의 여섯 효과를 더해 P2를 예측하지 않았습니다. P2는 새 자체 경로 K를 사용한 실제 전층 모델에서 PG19 이득을 직접 확인했으며, ArXiv의 손실·불확실성도 남았습니다.\n\nFP32 대수 부호와 BF16 실측 부호, 보수적 bound 인증은 구분했습니다. P1은 저장된2,304비교 모두 실측 방향이 같고2,195개가 bound 인증됐습니다. P1b fresh EVAL은1,536개 중1,535개 방향 일치·1,396개 인증입니다. 인증되지 않았다는 것은 실측 방향이 없거나 반대라는 뜻이 아닙니다. 사후 선택 민감도를 새 학습·보장으로 승격하지 않았습니다.',
          'Reducing local SSE and lowering gold next-token NLL are different objectives. Six P1b effects were not summed to predict P2. P2 directly observes PG19 gains in actual full models fitted on their own trajectories, alongside ArXiv losses/uncertainty.\n\nAlgebraic sign, measured BF16 sign and conservative-bound certification remain distinct. P1 has measured agreement in all2,304 saved comparisons and2,195 bound certifications. P1b fresh EVAL has1,535/1,536 measured agreement and1,396 certifications. Lack of certification does not mean unmeasured or reversed direction. Post-hoc selection sensitivity is not new training or a guarantee.'))
        sections.append(txt('같은 INT4 저장 형식의 고정 약한 자체 경로 보정은 이번 PG19 전층 비교에서 유효했습니다. ArXiv에서는 같은 이득을 확보하지 못했으므로 도메인 의존을 함께 보고합니다. 좋은 결과를 얻기 위한 추가 선택·실험 없이 P2를 완료했습니다.',
          'Fixed weak own-trajectory compensation works at the same INT4 format in this PG19 whole-model comparison. The same gain is not established on ArXiv, so domain limits accompany the result. P2 completed without extra selection or experiments to seek a favorable outcome.')+'\n\n| Axis | Scope |\n| --- | --- |\n| Capture/fit/serialization/reload |196 projections/model,588 historical files,392 K fits; implementation completed |\n| PG19 whole-model quality |Registered QUARTER−RTN LOWER_NLL |\n| ArXiv |QUARTER higher point estimate/CI crossing0; HALF higher NLL |\n| Native packed serving / VRAM savings / agent / other model |NOT_EVALUATED |\n| Publication |New GPU inference/K fits/requantization0; CPU scalar/interval/unit checks only |\n')
        sections.append(txt('원 연구 상위 문서6개는 교체 전 바이트를 보존했습니다. 원형579파일 중578개는 공개 사본과 byte-identical이며 환경 기록1개는 개인 model path 필드만 이전 공개 ZIP의 sanitized 값으로 대체했습니다. 원 raw는 로컬에서 그대로 보존했습니다. [대응표](publication/provenance/ORIGINAL_MAP.json)와 [단계 복원](publication/scripts/restore_stage.py)으로 추적할 수 있습니다.',
          'Six original top-level documents were preserved byte-for-byte before replacement. Of579 research files,578 are byte-identical in retained public locations; one environment receipt uses the earlier review ZIP’s sanitized model-path field. The private original is unchanged. See the [mapping](publication/provenance/ORIGINAL_MAP.json) and [stage restorer](publication/scripts/restore_stage.py).')+'\n\n[CPU/GPU reproduction](REPRODUCE.md) · [Stage map](STAGE_MAP.md) · [Publication checks](publication/verification/cpu-recalculation.json) · [Raw-derived display](publication/summary/project.json)\n\n'+txt('Guo의 [v1 논문](https://arxiv.org/html/2610.16020v1), [기존 source audit](references/source_audit.json), [공식 코드 식별자](https://github.com/tellsgo/quantization-compensation/tree/1e5dea93dfd9466ce0569fc8dde55ef8c2fcb9e0)를 구분합니다. 본문 절은 과거 HTML 대조 기록이며 PDF 페이지 별도 rendering은 미실행입니다. 저자의 HALF/새 문서 선택 전이는 선행 결과이며 이번 새 발견으로 주장하지 않습니다. 자체 capture·fit orchestration·codec/reload·분해·평가·auditor를 공개합니다. Qwen 모델·PyTorch·Transformers·NumPy와 보정 아이디어는 upstream/선행 연구입니다. Codex 보조와 별도 CPU 계산 경로 검산을 기록하며 사람의 독립 심사·외부 GPU 재현·공식 저자 승인으로 표시하지 않습니다.',
          'Distinguish Guo’s [v1 paper](https://arxiv.org/html/2610.16020v1), the [historical source audit](references/source_audit.json) and [official code identity](https://github.com/tellsgo/quantization-compensation/tree/1e5dea93dfd9466ce0569fc8dde55ef8c2fcb9e0). Section checks are historical HTML checks; PDF pages were not separately rendered. HALF and new-document selection transfer are prior author results, not claimed discoveries here. Project contributions are capture, fit orchestration, codec/reload, decomposition, evaluation and audit tools. Qwen/PyTorch/Transformers/NumPy and compensation ideas are upstream/prior work. Codex assistance and separate CPU calculation paths are disclosed, without claims of independent human peer review, external GPU reproduction or official author approval.'))
        for i,(label,key,text) in enumerate(zip(labels,anchors,sections),1):body+=f'\n<a id="{key}"></a>\n## {i}. {label}\n\n{text}\n'
        body+='\n\n'+txt('저장 장부: ','Storage ledger: ')+f"RTN=QUARTER=HALF247,733,136B; headers7,056 + scales27,525,120 + codes220,200,960B. BF16 projection tensors880,803,840B; unchanged common tensors311,296,000B (ties counted once). Core reduction{(1-247733136/880803840)*100:.3f}%.\n\n"+txt('Offline 구축: ','Offline construction: ')+f"RTN{R['stage_seconds_inclusive']['RTN_build']:.3f}s / QUARTER{R['stage_seconds_inclusive']['COMP_QUARTER_build']:.3f}s / HALF{R['stage_seconds_inclusive']['COMP_HALF_build']:.3f}s. "+txt('Capture·FP64 solve·진단·I/O 포함, serving 시간 아님.','Includes capture/FP64 solve/diagnostics/I/O, not serving time.')+'\n'
        (c/rep).write_text(english_spacing(body) if not ko else body)
    (c/'STAGE_MAP.md').write_text('''# Stage map / 단계와 원형

One active Case014 connects four separately designed stages. P1b/P2 were designed after prior observations.

| Stage | Input / unit | Candidate / selection | Original documents and records |
| --- | --- | --- | --- |
| P0 | Rational fixtures;4-code chain;2000 stored noise seeds per dither condition | Three distinct mathematical models, no LM quality | [Original P0 report](publication/original_pre_p1/REPORT.ko.md), [raw](results/local_r1/p0_full.json) |
| P1 p1_r2 | PG19 CAL32/DEV16/EVAL32;ArXiv16;256-token windows |6 independent targets×5alphas;CAL raw SSE selector | [Original P1 report](publication/originals/pre_publication/REPORT.ko.md), [cells](results/p1_r2/cells/), [freeze](results/p1_r2/fit_frozen.json) |
| P1b p1b_r2 | New SELECT32/PG19 EVAL64/ArXiv16;known prior inputs excluded |Same6 K/30artifacts;SSE/NLL selectors;7policies can alias | [Report](followups/p1b_selection/REPORT.ko.md), [cells](results/p1b_r2/cells/), [selection](results/p1b_r2/selection_frozen.json) |
| P2 p2_r1 |Original CAL32;separate fresh PG1964/ArXiv16 |Whole196projection models;new own-path K for fixed.25/.5;QUARTER primary | [Report](followups/p2_core/REPORT.ko.md), [metrics](results/p2_r1/metrics/), [freeze](results/p2_r1/artifact_frozen.json) |

Each model/method uses original next-token targets, not teacher argmax labels. Different stages/cohorts are not pooled.
The six P1/P1b hybrid interventions are not six independent models or six times as many documents.

[Research-tree mapping](publication/provenance/ORIGINAL_MAP.json) preserves source sizes/hashes and retained paths.
[Historical archive mapping](publication/provenance/STAGE_ARCHIVES.json) restores exactly the original file set for each stage.
Root documents changed by integration are retained before editing. Personal model-path metadata has an explicit sanitized-copy mapping.
Historical NOT_RUN_PUBLICATION/storage-policy decisions remain historical records, not current publication status.

Restore into a new external folder before using an original closed-set verifier. Links in relocated byte-preserved documents retain
their original relative meaning in the restored view. The publication wrapper performs these restorations automatically.

```bash
python3 -B cases/014-quantization-compensation/publication/scripts/restore_stage.py --stage P1 --output "$NEW_VIEW"
```

Requested named independent human/AI review attachments were not present in the scoped local inputs. The historical executors'
receipts and new Codex-assisted CPU audits are labeled separately in [review availability](publication/reviews/availability.json).
''')
    (c/'REPRODUCE.md').write_text('''# Public reproduction / 공개 검산

## A. Model-free records → tables → intervals

Run from the repository root. Use a new external output directory. The first command uses only Python stdlib;
the full command needs NumPy2.3.5 and Torch2.13.0 CPU operations. A hosted CI uses the pinned CPU-only requirements
in publication/requirements-cpu.txt. Existing research environments were not installed into or upgraded.

```bash
OUT=$(mktemp -d)
python3 -B cases/014-quantization-compensation/publication/scripts/derive.py --output "$OUT/display.json"
python3 -B cases/014-quantization-compensation/publication/verify.py --output "$OUT/publication.json"
```

With an existing suitable CPU Python selected as CPU_PY:

```bash
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OUT=$(mktemp -d)
"$CPU_PY" -B cases/014-quantization-compensation/publication/scripts/run_checks.py --output "$OUT/case014" --tests
```

This restores closed P0/P1/P1b/P2 views and checks 2880/3360/320 raw rows, original CAL/SELECT choices,
policy aliases, NLL/PPL/KL/flip aggregates,120 P1 intervals,462 P1b intervals and20 P2 NLL/PPL interval fields
with the original bootstrap seeds. Existing15+10+9 tests are rerun; publication tests are a separate suite.
P0 dither SSE/SE are recalculated from saved noise outcomes; the2000-seed simulation itself is not rerun.
The public bundle identifies external weights/artifacts by hash; it does not decode private payloads or rerun GPU logits.

## B. Historical GPU experiment replay

This publication performs no model forward, K fit or re-quantization. The original model is
Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca; RTX5080/SM120,
Torch2.13.0+cu130/Transformers5.17.0. Full local shard SHA256 is recorded in the model manifests.
The WSL read-only snapshot exception was authorized; D-origin comparison was not required.

Private source text/token arrays, saved K/INT4 weights/large X/Z/logits are not in this repository or ZIP.
Hashes alone cannot reconstruct them. Follow the stage-specific requirements:

- [P1 original reproduction](publication/originals/pre_publication/REPRODUCE.md)
- [P1b read-only-artifact reproduction](followups/p1b_selection/REPRODUCE.md)
- [P2 own-trajectory reproduction](followups/p2_core/REPRODUCE.md)

Those original commands require their external materials and, where the runner records Git HEAD, a repository checkout.
Replaying already-observed EVAL is not a new fresh test. CPU publication checks do not invoke these commands.

## C. Static site and archive

```bash
OUT=$(mktemp -d)
"$CPU_PY" -B tools/showcase/run_checks.py --output "$OUT/portfolio"
```

The same built site is checked and deployed by the existing manual portfolio-pages workflow. No new framework is needed.
The unified download contains this Case014 tree once, with a separate content manifest; historical ZIPs are not nested.
Completed archive SHA/bytes and publication/merge/deploy revisions are separate receipts, avoiding self-hash cycles.
''')
    (c/'PORTFOLIO.md').write_text('''# 구현과 기술 / Implementation and technology

실제 입력 경로를 따라 보정하고 같은 INT4 저장 형식에 효과를 남기는 도구입니다. PG19의 전층 품질 이득과 ArXiv 전이 한계를 함께 제공합니다.
The pipeline captures actual paths, fits compensation and stores it in the same INT4 representation, with whole-model quality and domain-limit evidence.

| Project implementation | Upstream / prior work | Source |
| --- | --- | --- |
| Native capture and topological own-path orchestration | Qwen3 model, Transformers, PyTorch hooks/SDPA | [runner](followups/p2_core/source/run_p2.py) |
| Full-input FP64 ridge and shared-factor scheduling | PyTorch Cholesky/triangular solve; Guo compensation motivation | [fit](followups/p2_core/source/p2_fit.py) |
| Versioned signed INT4 storage, independent reload | Standard RTN, NumPy arrays, Python struct | [codec](scripts/codec.py), [independent decoder](followups/p2_core/source/independent_codec.py) |
| Three-term decomposition, paired scoring and audit | Linear algebra, NLL/PPL/KL, document bootstrap | [decompose](scripts/decompose.py), [auditor](followups/p2_core/source/audit.py) |
| Raw-derived bilingual result display and closed stage views | Python stdlib, GitHub Actions/Pages | [exporter](publication/scripts/derive.py), [restorer](publication/scripts/restore_stage.py) |

[Results](REPORT.md) · [한국어](REPORT.ko.md) · [CPU workflow](REPRODUCE.md) · [Original sources](NOTICE.md)

Codex assisted implementation/execution/analysis/writing. Separate CPU calculation paths are verification tools, not independent human peer review or official author approval.
''')
    (c/'NOTICE.md').write_text('''# Sources, attribution and current publication scope

Project code follows the unchanged repository/Case014 Apache-2.0 LICENSE. No third-party source was copied or relicensed.
This is an independent ridge/RTN reference study, not the author's complete QEP reproduction or a new quantizer.

- Tianhao Guo, *When Does Compensating Inherited Quantization Error Help?*, arXiv2610.16020v1:
  [paper](https://arxiv.org/html/2610.16020v1), [historical source audit](references/source_audit.json).
- [Author code identity](https://github.com/tellsgo/quantization-compensation/tree/1e5dea93dfd9466ce0569fc8dde55ef8c2fcb9e0): identified historically, not vendored/executed as official full reproduction.
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), Transformers/PyTorch native model execution and NumPy arithmetic are upstream.
- Project contributions: actual-path capture/fit orchestration, signed INT4 codec/independent decoder,
  decomposition/measurement, independent scalar auditors and traceable bilingual publication.

External model/book/article rights remain applicable. Model/INT4 weights/K/activation/logits/source text are excluded.
Historical raw reports/decisions/failed attempts remain preserved; their old nonpublication scope is historical.
Current publication approval covers PR, normal main merge and the existing manual Pages workflow. No new GPU inference
is performed in publication. Codex assisted this work; no independent human review, external GPU rerun or official author approval is claimed.
''')
    (c/'references.bib').write_text('''@article{guo2026compensating,
  title={When Does Compensating Inherited Quantization Error Help? Mean-Square Guarantees for Layer-Wise Quantization and Their Limits},
  author={Guo, Tianhao}, year={2026}, eprint={2610.16020}, archivePrefix={arXiv},
  url={https://arxiv.org/abs/2610.16020v1}
}
''')
    for name in ['STAGE_MAP.md','REPRODUCE.md','PORTFOLIO.md','NOTICE.md']:
        p=c/name;p.write_text(english_spacing(p.read_text()))
    print('Unified bilingual README/report, stage map, reproduction and contribution/source documents generated.')
if __name__=='__main__':main()
