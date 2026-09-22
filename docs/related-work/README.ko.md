# 선행연구와 구현의 대응

[English](README.md) · [홈](../../README.ko.md)

기존 사례의 정식 보고서는 당시 기록으로 유지합니다. 아래 문헌은 DIOVA가 사용한 계산과 평가의 배경이며, 새 측정은 Case 009에 남깁니다.

| 선행연구 | 공통점과 차이 | 이 프로젝트의 구현 |
|---|---|---|
| He·Zhang·Sun, *Channel Pruning for Accelerating Very Deep Neural Networks*, ICCV 2017 | CNN의 LASSO 채널 선택과 최소제곱 재구성입니다. R은 이미 정한 LLM MLP 삭제 집합에서 정규화된 down projection을 맞춥니다. CNN 선택기의 직접 재현은 아닙니다. | [고정 구조 ridge 적합](../../tools/modelpack/numerics.py), [층별 구조 로더](../../tools/modelpack/artifact.py). |
| An·Zhao·Yu·Tang·Wang, *Fluctuation-based Adaptive Structured Pruning for Large Language Models*, AAAI 2024 | FLAP은 fluctuation 중요도, 전체 구조 배분, 출력 bias 보상을 함께 사용합니다. R은 작은 기존 가중치에 보정값을 넣으며 bias를 추가하지 않습니다. 향후 bias만 비교하면 FLAP-inspired 대조군입니다. | [후속 다층 설계](../phase2a-design.md). 이번 실행에는 포함하지 않습니다. |
| Frantar·Castro·Chen·Hoefler·Alistarh, *MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models*, PPoPP 2025 | 양자화 Linear 커널의 병렬 autoregressive 실행입니다. 논문의 속도비는 해당 장치·shape·dtype·기준선 조건에 속합니다. 클라이언트 동시성과 Linear의 token-row 수는 서로 다릅니다. | 기존 GPTQ 저장본과 vLLM의 Marlin 구현을 사용하고 [새 runner](../../cases/009-q-serving-quality/scripts/serving.py)로 요청 전체를 측정합니다. |
| Dutta·Krishnan·Kwatra·Ramjee, *Accuracy is Not All You Need*, NeurIPS 2024 | 압축 전후 평균 정답률과 입력별 정답 전환을 나눕니다. 논문의 flips는 오답에서 다른 오답으로 바뀐 경우를 제외합니다. | [비교 도구](../../packages/diova-compare/src/diova_compare/core.py)가 correctness flip과 전체 답변 변경을 별도 필드로 제공합니다. |

입력 i의 정답 여부를 bᵢ, cᵢ, 추출된 답을 aᵢᴮ, aᵢᶜ라 하면 Dutta의 correctness-flip 비율은 Σ1[bᵢ≠cᵢ]/N, 즉 (정답 손실+새 정답)/N입니다. 전체 답변 변경은 Σ1[aᵢᴮ≠aᵢᶜ]/N입니다. 정답이 하나라면 두 비율의 차이는 다른 오답으로의 변경이고, 허용 정답이 여럿이면 양쪽 정답의 답변도 달라질 수 있습니다. 보기 점수를 정규화한 KL은 전체 어휘 next-token KL이 아닙니다. Dutta의 KL은 답변 보기의 각 token에서 전체 어휘 분포를 비교한 집계입니다.

프로젝트 구현은 BF16 보정값 저장, artifact 검사, meta에서 구조 복원과 strict 할당, 실행기 연결과 결과 화면입니다. GPTQ·Qwen·Transformers·safetensors·vLLM과 CUDA 커널은 upstream 작업입니다. Codex가 구현·실행·검사·문서 작성을 지원했습니다. 각 사례의 고지와 라이선스는 그대로 유지합니다.

원문 제목·저자·연도·버전과 공식 링크는 [영문 참고 목록](README.md#primary-sources)에 있습니다. MARLIN은 arXiv:2408.11743v1, Dutta는 2407.09141v1, FLAP은 2312.11983v1, He는 1707.06168을 확인했습니다. CVF는 이번 확인에서 HTTP 403을 반환해 저자의 arXiv 자료를 읽었습니다. vLLM 공식 문서는 실행 개념의 참고이고, 실제 플래그와 지표 정의는 설치된 **0.29.0** 소스에서 확인합니다. lm-evaluation-harness는 `d6de81643928d653435c431bae19945d41d32520`으로 고정했습니다.
