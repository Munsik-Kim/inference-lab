# 처음 보는 분을 위한 DIOVA 안내

[English](../en/START_HERE.md) | 한국어 · [홈](../../README.ko.md)

## 1. 어떤 문제를 다루나요?

내 GPU에서 모델을 더 가볍게 실행하려면 저장 공간뿐 아니라 요청 속도와 답변도 확인해야 합니다. DIOVA는 RTX 5080 16GB에서 공개 가중치 모델을 압축해 저장하고 다시 실행하는 도구, 같은 질문의 답변과 비용을 비교하는 실행기, 도구 작업의 실제 완료를 확인하는 검사를 만듭니다. 결과를 읽는 데 AI 전공 지식이나 GPU는 필요 없습니다.

## 2. 온라인 화면부터 읽어보세요

[공개 한국어 홈페이지](https://munsik-kim.github.io/inference-lab/ko/index.html)에서 설치 없이 기록된 사례와 코드를 볼 수 있습니다. 실시간 모델 데모가 아니라 저장된 결과입니다. [16GB GPU 모델별 안내](https://munsik-kim.github.io/inference-lab/ko/llm-guide.html)에서 Qwen3-4B의 실행·품질·실패 기록을 이어서 읽으세요. [안내서 연결 문서](LLM_GUIDE.md)에는 로컬 열기 방법도 있습니다.

목적에 따라 바로 시작할 수 있습니다.

- **모델을 쓰려는 사람:** [모델별 실전 안내](LLM_GUIDE.md)에서 실행 경로와 전제조건을 확인합니다.
- **기술 역량을 보려는 사람:** [대표작 A/B/C와 코드 근거](PORTFOLIO.md)를 읽습니다. A는 [Case008 Q 저장·재로딩](https://munsik-kim.github.io/inference-lab/ko/case008.html#track-q) → [Case009 실제 요청 비교](https://munsik-kim.github.io/inference-lab/ko/case009.html#serving)로 이어집니다.
- **직접 비교해보려는 사람:** [diova-compare 합성 예제](../../packages/diova-compare/examples/compare/README.ko.md)로 두 파일을 validate하고 보고서를 만듭니다. 이후 자기 기록의 필드를 바꿔 넣습니다.
- **실험을 검산하려는 사람:** [개발자 재현 안내](GETTING_STARTED.md)에서 과거 기록의 CPU 재계산 경로를 고릅니다. 위의 자기 결과 비교와 별개입니다.

대표작 C는 [도구 이름 제한·오류 피드백과 실제 작업 완료 검사](../../research/agent_tool_contract_repair_20261014/README.ko.md)입니다. 기존 DEV 8과제를 재사용한 BF16 복원 reference 계산이며 새 입력 일반화나 packed INT4 가속 근거로 넓히지 않습니다.

## 3. 어느 안내를 읽으면 되나요?

| 알고 싶은 것 | 읽을 곳 |
|---|---|
| 이 모델에서 무엇이 실행됐고 어디서 문제가 생겼는가 | [모델별 실전 안내](LLM_GUIDE.md) — Qwen3-4B 상세와 짧은 영어 요약 |
| 메모리·속도·품질 숫자를 어떻게 구분하는가 | [측정 방법과 결과 읽는 법](https://munsik-kim.github.io/inference-lab/ko/measurements.html) |
| 결과와 도구를 내 컴퓨터에서 어떻게 재확인하는가 | [개발자 재현 안내](GETTING_STARTED.md) — 기존 사이트의 `guide.html`도 이 역할 |
| 생소한 용어가 무엇인가 | [용어집](GLOSSARY.md) |

메모리에 올라가는 것, 요청이 빨라지는 것, 내 업무를 정확히 수행하는 것은 서로 다른 판단입니다. 다른 실험의 좋은 결과를 합쳐 한 설정의 성능으로 읽지 않습니다.

## 4. 전체 사례는 어떻게 이어지나요?

각 사례는 모델·입력·조건이 다른 기록입니다. 처음부터 모두 읽을 필요는 없습니다.

- [001 실행 호환성](CASEBOOK.md#case-001)
- [002 공식 BF16·FP8 문서 추출](CASEBOOK.md#case-002)
- [003 수치 근사의 비교 기준](CASEBOOK.md#case-003)
- [004 준비 작업을 포함한 호출 비용](CASEBOOK.md#case-004)
- [005 속도와 오차의 설정 비교](CASEBOOK.md#case-005)
- [006 같은 질문의 점수와 답변 변경](CASEBOOK.md#case-006)
- [007 행렬을 실제로 줄인 구조 압축](CASEBOOK.md#case-007)
- [008 양자화 저장·재실행과 별도 구조 복원](CASEBOOK.md#case-008)
- [009 서빙 비용과 공식 품질 평가](CASEBOOK.md#case-009)
- [010 상태 저장·재시작과 기억 수명](CASEBOOK.md#case-010)
- [011 같은 상태에서 판독기 변경](CASEBOOK.md#case-011)
- [012 이미지 생성 시간 배분](CASEBOOK.md#case-012)
- [013 가중치 오차와 실제 답변·도구 사용](CASEBOOK.md#case-013)

Case008 R은 Qwen3-0.6B의 pruning 후 복원이며 Qwen3-4B 양자화 보정과 별개입니다. 전체 수치·판정은 [사례 안내](CASEBOOK.md), 직접 구현한 부분과 외부 기술의 역할은 [포트폴리오](PORTFOLIO.md)로 이어집니다.

## 5. 다운로드하거나 직접 검사하려면?

먼저 온라인에서 결과를 읽고, 필요할 때 [다운로드·로컬 열기](GETTING_STARTED.md#offline-explorers)나 [CPU 기록 검사](GETTING_STARTED.md#cpu-checks)를 선택하세요. 공개 ZIP에는 모델 가중치가 없으며, 공개 기록 검산과 GPU 재실행은 다른 절차입니다.

## 6. 한 층의 계산을 바꾼 관련 구현

<!-- claims: c007-transfer c007-quality c006-native c006-readout -->
**Case 007:** 모델 안에서 정보를 변환하는 큰 블록인 MLP 하나를 16그룹으로 나눴습니다. 그룹을 따로 평가하거나 서로의 상호작용까지 보고 제거 대상을 골랐습니다. 행렬을 실제로 잘라 작은 블록을 만든 뒤, 선택에 쓰지 않은 입력 192개로 검사했습니다. 25% 제거에서는 상호작용 선택이 원래 출력에 조금 더 가까웠고, 50%에서는 같은 구조를 골랐습니다. 블록 자체는 빨라졌지만 전체 모델의 가속은 뚜렷하지 않았습니다. 원래 출력에 가까운 정도와 정답 확률 점수도 다른 결과였습니다. [정확한 결과](CASEBOOK.md#case-007).

**Case 006:** 질문의 정보를 조합하는 attention 연산 한 곳을 바꾸고, 같은 질문의 점수와 네 가지 보기 중 고른 답을 비교했습니다. 두 설정은 표준 192개 중 각각 5개·8개의 선택을 바꿨습니다. 그 집합에서는 기준선의 정답을 잃지 않았지만, 별도 스트레스 집합에서는 잃은 경우가 있었습니다. 후속 진단은 같은 입력에서 최고점 동률과 마지막 단어 점수 계산을 살폈습니다. 원래 실험과 별도로 표시됩니다. [입력별 결과와 한계](CASEBOOK.md#case-006).

<!-- case014-entry -->
## 심화 연구: 같은 INT4 저장량의 품질 보정

[Case014](CASEBOOK.md#case-014) · [Online](https://munsik-kim.github.io/inference-lab/ko/case014.html#overview) · [CPU checks](GETTING_STARTED.md#case014-cpu). 0.6B 전층 품질을 새 문서에서 비교한 연구이며, 기존 Qwen3-4B 실전 안내와 모델·측정 범위가 다릅니다.
