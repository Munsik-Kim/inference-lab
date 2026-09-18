# 구현 역량과 근거

[English](../en/PORTFOLIO.md) | 한국어 · [홈](../../README.ko.md)

## 30초 소개

<!-- claims: c006-native c006-readout -->
Inference Lab은 실행 성공, 빠른 연산, 안정적인 모델 답변에 왜 서로 다른 검사가 필요한지 조사합니다. 기존 추론 구성요소 위에 비교 조건의 통제, 별도로 계산한 정답(gold)을 갖춘 과제, 실행 경로·유효성 검사, CPU에서 확인하는 보고 체계를 구현했습니다. B는 원래 BF16 실행인 비교 기준선입니다. A_PUBLIC과 V4는 같은 모델의 attention 연산 한 곳에 적용한 두 저정밀 설정의 이름이며, 모델 가중치는 BF16으로 유지합니다. 대표 Qwen 실험의 표준셋에서는 A_PUBLIC이 192개 중 5개, V4가 192개 중 8개의 선택을 바꾸었고 정답 획득과 다른 오답으로의 변경이 포함됐습니다. Readout은 모델 내부 상태를 어휘 점수로 바꾸는 마지막 출력 계산입니다. 동일 입력 후속 진단은 원래 출력 계산(native)과 별도 진단용 출력 계산(shadow)을 구분합니다. 원래 동률이 해석에 중요함을 보여주지만 본래 결과를 대체하지 않습니다. [사례 안내](CASEBOOK.md#case-006)와 [오프라인 근거 안내](GETTING_STARTED.md#offline-explorers)에서 정확한 항목까지 확인할 수 있습니다. 제한된 합성 입력의 조사이며 모델 품질을 보증하지 않습니다.

## 실제 자료로 확인하는 역량

- **실행 경로 진단:** Case001은 지원하지 않는 CUTLASS를 허용한 선택 조건과 기존 Triton 대체 경로를 구분합니다. [수정 전후 runner — Python 소스](../../cases/001-sm120-int8-fallback/run.py)와 [검증 기록 — 기술 원문(영어)](../../cases/001-sm120-int8-fallback/README.md)에서 재현 도구와 upstream 수정을 나누어 확인합니다.
- **공정한 비용 경계:** Case004는 필요한 전처리와 출력 변환을 전체 연산 호출 안에 포함하고 kernel-only 시간을 별도로 표시합니다. [방법 — 영어](../../cases/004-low-precision-attention-break-even/METHODS.md)와 [backend 어댑터 — Python 소스](../../cases/004-low-precision-attention-break-even/src/backends.py)가 그 계약을 보여줍니다.
- **기준을 바꾸지 않는 수치 비교:** Case003은 매핑과 scale 효과를 분리하고, Case005는 개발 탈락을 유지하면서 실제 절충을 보여줍니다. [수치 감사 — 영어](../../cases/003-efq-softmax-numerical-audit/ANALYSIS.md), [개발 분석 — 영어](../../cases/005-attention-precision-pareto/ANALYSIS.md), [별도 사후 검토 — 영어](../../cases/005-attention-precision-pareto/POSTHOC_THRESHOLD_REVIEW.md)는 서로 다른 근거 범위를 유지합니다.
- **모델 내부 개입 통제:** Case006은 prefill 한 층만 가로채고 원래 경로를 복구하며 일부 logits만이 아닌 전체 출력 유효성을 검사합니다. [개입 코드 — Python](../../cases/006-attention-decision-stability/src/intervention.py)와 [유효성 검사 — Python](../../cases/006-attention-decision-stability/src/validity.py)에서 경계를 확인합니다.
- **점수 해석의 수치 진단:** 보충 자료는 정확한 동률, 정답 전환, 각 readout 안의 후보 대 B 비교를 구분하며 native와 FP32는 별도 보기로 유지합니다. [방법 — 영어](../../cases/006-attention-decision-stability/supplemental/readout-ties-v1/METHODS.md)과 [스칼라 검사기 — Python](../../cases/006-attention-decision-stability/supplemental/readout-ties-v1/scripts/verify_scalar.py)는 사후 진단을 새 평가와 분리합니다.
- **검사할 수 있는 근거 전달:** 정확한 입력, 소스 식별자, 쌍별 기록과 로컬 HTML이 집계표를 개별 관측과 연결합니다. [입력 manifest JSON](../../cases/006-attention-decision-stability/configs/eval_manifest_freeze_v1.json), [독립 스칼라 계산 — Python](../../cases/006-attention-decision-stability/scripts/audit_study.py), [탐색기 이용 안내](GETTING_STARTED.md#offline-explorers)가 경로를 보여줍니다.

## 구조화 압축: 또 하나의 구현 사례

<!-- claims: c007-transfer c007-quality -->
Case007은 선택한 채널 그룹을 실제로 더 작은 dense MLP로 바꾸고, 범위를 제한한 교체와 마스킹·행렬 축소의 대응을 검사합니다. 25%에서 PAIRWISE의 미관측 입력 평균 국소 오차는 29.6288%로 INDEPENDENT의 29.8291%보다 낮았습니다. 삭제하지 않은 BF16 기준선에는 더 가까웠지만 정답 선택지 NLL까지 가장 좋지는 않았습니다. 50%에서는 같은 모듈을 만들었으므로 고정 종합 판정은 **COMPLETED_NO_CLEAR_TRANSFER**이며 모델 전체 prefill 가속은 입증하지 못했습니다. 기준선 보존을 과제 품질과 혼동하지 않고 선택·실제 모델 구조 변경·미관측 입력 평가를 연결한 구현입니다. [사례 안내](CASEBOOK.md#case-007) · [구조 변경 Python 소스](../../cases/007-interaction-aware-mlp-pruning/src/surgery.py) · [저장된 탐색기](GETTING_STARTED.md#case007-explorer).

Qwen/Transformers는 모델을 제공하며, HOPE는 별도 MoE expert pruning 맥락에서 개념적 동기를 줍니다. 그룹의 부호 있는 기여 항등식은 기본적인 계산 관계이며 새 pruning 알고리즘이나 HOPE 재현이 아닙니다. [Case007 출처와 재사용 — 영어](../../cases/007-interaction-aware-mlp-pruning/NOTICE.md). 아래의 Codex 지원 및 스칼라 검산·GPU 재현 구분도 동일하게 적용됩니다.

<a id="contribution-and-reuse"></a>
## 프로젝트 기여와 재사용

<!-- claims: attribution -->
프로젝트의 작업은 실행 도구, 실험 통제, 근거 수집, 수치·행동 분석과 전달입니다. Qwen과 Red Hat AI는 체크포인트를 제공합니다. vLLM은 서빙 엔진, PyTorch와 Transformers는 수치 연산과 모델 통합, SageAttention은 저정밀 커널을 제공합니다. Case001 guard의 작성자는 hclsys이고 EFQ-Softmax의 저자는 Han과 공동저자입니다. 이 저장소가 새로 발명한 커널이나 quantizer가 아닙니다.

[Case001 출처 — 영어](../../cases/001-sm120-int8-fallback/NOTICE.md), [Case003 출처 — 영어](../../cases/003-efq-softmax-numerical-audit/NOTICE.md), [Case006 출처 — 영어](../../cases/006-attention-decision-stability/NOTICE.md)에 원래 기여와 해당 라이선스가 있습니다. 저장소의 [Apache-2.0](../../LICENSE)이 upstream 모델 조건을 대신하지 않습니다. OpenAI Codex가 구현·로컬 실행·테스트·분석·문서 작성을 지원했습니다. 별도 CPU 계산 경로는 보존된 기록의 일관성을 확인하며 외부 제3자 GPU 재현이나 독립적인 사람의 검토가 아닙니다.

## 이 프로젝트로 이야기할 수 있는 범위

**연구·평가 엔지니어링:** 비교 조건 설정, 독립 정답 유지, 실패 기록, 불확실성 검사, 결과가 측정한 대상의 구분입니다.

**추론 진단·분석 도구 협업:** 기존 runtime 주위의 어댑터, 출처를 추적할 수 있는 비교 보고, CPU 재분석, 오프라인 근거 탐색기입니다. 기록된 구현을 근거로 협업할 수 있는 주제입니다. 결과는 조사한 모델·장치·과제 범위에 한정되며 배포 적합성은 평가하지 않았습니다.

## 근거에 연결되는 소개 문장 세 개

<!-- claims: c006-implementation c004-cost c005-stop -->
- Qwen3-0.6B layer 13의 prefill만 바꾸는 통제 비교를 구현하고, 기준선과의 불일치를 정답 획득·손실과 구분했습니다. [범위와 구현 — 기술 원문(영어)](../../cases/006-attention-decision-stability/METHODS.md).
- 저정밀 attention의 전체 호출 비용을 fused BF16과 비교하고, 빨랐던 길이에서도 국소 오차 기준 탈락을 그대로 기록했습니다. [연산 결과 — 영어](../../cases/004-low-precision-attention-break-even/README.md).
- Case005의 **STOP_DEV_SCREEN**을 보존하면서 측정된 정밀도·비용 절충과 별도 사후 해석을 공개했습니다. [개발 판정 — 영어](../../cases/005-attention-precision-pareto/README.md).

## 세 단계로 설명하기

1. 표준 결과표와 독립 시나리오 분모를 먼저 읽습니다. 속도 수치보다 먼저 어느 연산을 바꾸었는지 설명합니다.
2. 원래 탐색기에서 ID 하나를 골라 gold·기준/후보 점수·결과 유형을 확인합니다. 별도 스트레스셋으로 옮겨 선정 규칙을 설명합니다.
3. 보충 탐색기에서 같은 ID의 native 동률과 shadow readout을 비교합니다. [한계 — 기술 원문(영어)](../../cases/006-attention-decision-stability/LIMITATIONS.md)로 돌아가 낮은 코드 성능, 실패한 구조화 생성, 작은 모델 prefill 효과, 미평가 배포 상태를 함께 설명합니다.

[이용 안내](GETTING_STARTED.md)에 로컬 열기와 CPU 검증 방법이 있습니다. 이 시연에는 호스팅된 추론 서비스가 포함되지 않습니다.
