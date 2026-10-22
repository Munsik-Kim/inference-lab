# 처음 보는 분을 위한 DIOVA 안내

[English](../en/START_HERE.md) | 한국어 · [홈](../../README.ko.md)

## 1. 어떤 문제를 다루나요?

모델 사용, 기술 역량 확인, 자기 결과 비교, 원자료 검산 중 목적에 맞는 입구를 고르세요. 저장된 기록을 읽고 CPU 비교 도구를 쓰는 데 GPU는 필요 없습니다.

## 2. 온라인 화면부터 읽어보세요

[공개 한국어 홈페이지](https://munsik-kim.github.io/inference-lab/ko/index.html)에서 설치 없이 기록된 사례와 코드를 볼 수 있습니다. 실시간 모델 데모가 아니라 저장된 결과입니다. [16GB GPU 모델별 안내](https://munsik-kim.github.io/inference-lab/ko/llm-guide.html)에서 Qwen3-4B의 실행·품질·실패 기록을 이어서 읽으세요. [안내서 연결 문서](LLM_GUIDE.md)에는 로컬 열기 방법도 있습니다.

목적에 따라 바로 시작할 수 있습니다.

- **로컬 모델을 실행하려는 사람:** [모델별 실전 안내](LLM_GUIDE.md)에서 무엇이 실행됐고 무엇이 필요하며 어디서 실패했는지 확인합니다.
- **채용할 역할이나 의뢰할 기술 작업과의 관련성을 보려는 사람:** [대표작 A/B/C와 코드 근거](PORTFOLIO.md)를 읽습니다. A는 [Case008 Q 저장·재로딩](https://munsik-kim.github.io/inference-lab/ko/case008.html#track-q) → [Case009 실제 요청 비교](https://munsik-kim.github.io/inference-lab/ko/case009.html#serving)로 이어집니다.
- **자기 답변 기록을 비교하려는 사람:** [diova-compare의 가상 답변 예제](../../packages/diova-compare/examples/compare/README.ko.md)로 두 파일의 형식을 검사하고 비교 보고서를 만듭니다. 이후 문서의 형식에 맞춰 자기 기록을 넣습니다. 모델 없이 쓰는 독립 CPU 도구입니다.
- **실험을 검산하려는 사람:** [개발자 재현 안내](GETTING_STARTED.md)에서 과거 기록의 CPU 재계산 경로를 고릅니다. 위의 자기 결과 비교와 별개입니다.

대표작 C는 [Case017 결과와 실제 사건](../../cases/017-agent-contract-cost-bench/REPORT.ko.md)에서 시작합니다. [모델 없이 CPU 재생](../../cases/017-agent-contract-cost-bench/REPRODUCE.ko.md)과 [앞선 Case013 후속 구현](../../research/agent_tool_contract_repair_20261014/README.ko.md)도 연결합니다.

## 3. 참고 안내

[메모리·속도·정답 숫자 읽기](https://munsik-kim.github.io/inference-lab/ko/measurements.html) · [쉬운 용어집](GLOSSARY.md). 파일 크기·실행 메모리·답변 품질은 서로 다른 판단입니다.

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

Case008 R은 더 작은 Qwen3-0.6B에서 한 계산 블록의 일부를 제거한 뒤(pruning) 남은 출력 가중치를 보정하는 작업입니다. Qwen3-4B 압축 흐름과는 별개입니다. 전체 수치·판정은 [사례 안내](CASEBOOK.md), 직접 구현한 부분과 외부 기술의 역할은 [포트폴리오](PORTFOLIO.md)로 이어집니다.

## 5. 다운로드하거나 직접 검사하려면?

먼저 온라인에서 결과를 읽고, 필요할 때 [다운로드·로컬 열기](GETTING_STARTED.md#offline-explorers)나 [CPU 기록 검사](GETTING_STARTED.md#cpu-checks)를 선택하세요. 공개 ZIP에는 모델 가중치가 없으며, 공개 기록 검산과 GPU 재실행은 다른 절차입니다.

## 6. 한 층의 계산을 바꾼 관련 구현

<!-- claims: c007-transfer c007-quality c006-native c006-readout -->
**Case 007:** 모델 안에서 정보를 변환하는 큰 블록인 MLP 하나를 16그룹으로 나눴습니다. 그룹을 따로 평가하거나 서로의 상호작용까지 보고 제거 대상을 골랐습니다. 행렬을 실제로 잘라 작은 블록을 만든 뒤, 선택에 쓰지 않은 입력 192개로 검사했습니다. 25% 제거에서는 상호작용 선택이 원래 출력에 조금 더 가까웠고, 50%에서는 같은 구조를 골랐습니다. 블록 자체는 빨라졌지만 전체 모델의 가속은 뚜렷하지 않았습니다. 원래 출력에 가까운 정도와 정답 확률 점수도 다른 결과였습니다. [정확한 결과](CASEBOOK.md#case-007).

**Case 006:** 질문의 정보를 조합하는 attention 연산 한 곳을 바꾸고, 같은 질문의 점수와 네 가지 보기 중 고른 답을 비교했습니다. 두 설정은 표준 192개 중 각각 5개·8개의 선택을 바꿨습니다. 그 집합에서는 기준선의 정답을 잃지 않았지만, 별도 스트레스 집합에서는 잃은 경우가 있었습니다. 후속 진단은 같은 입력에서 최고점 동률과 마지막 단어 점수 계산을 살폈습니다. 원래 실험과 별도로 표시됩니다. [입력별 결과와 한계](CASEBOOK.md#case-006).

<!-- case014-entry -->
## 심화 연구: 같은 INT4 저장량의 품질 보정

[Case014](CASEBOOK.md#case-014) · [온라인](https://munsik-kim.github.io/inference-lab/ko/case014.html#overview) · [모델 없는 CPU 검산](GETTING_STARTED.md#case014-cpu). Qwen3-0.6B의 압축 오차를 보정한 뒤 새 문서의 텍스트 예측 품질을 비교한 별도 연구입니다. 변환한 행렬의 저장량을 유지한 PG19 결과와 ArXiv의 한계를 나눠 읽으며, 기존 Qwen3-4B 실전 안내와 모델·측정 범위가 다릅니다.

<!-- case015-entry -->
<a id="case015-cpu"></a>
## API 이름 변경 회귀시험

도구 이름만 바뀌어도 AI가 같은 일을 끝내는지 확인합니다. 예제 작업은 **구매한 개수와 개당 가격을 읽고→총금액을 계산하고→저장한 뒤→맞게 기록됐는지 확인**하는 일입니다. 기능과 입력 형식은 유지하고 `read_order`를 `readOrder`처럼 이름 표기만 바꿉니다. 16개 문제에서 BF16은 **15/16→16/16**, RTN은 **16/16→16/16**, D16은 **12/16→12/16**이었습니다. [작업·용어 안내](../../cases/015-tool-interface-sensitivity/README.ko.md)에서 주문(구매 내역)·단가(개당 가격)와 결과표 읽는 법을 설명합니다. 기존 시험 P1과 새 응용시험 A1의 각 96회는 별도로 집계합니다.

<a id="3-어느-안내를-읽으면-되나요"></a>

## Case017 — 도구 호출 규칙과 업무 완주 비용

도구 호출 규칙이 실제 저장·검증·최종 답의 완료를 얼마나 바꾸는지 비교하는 평가기를 구현했습니다. 상품 합계 기록·창고 이동·문서 정보 변경을 소유한 가상 기록에서 시험합니다.

BF16 가중치를 고정한 NVFP4 KV의 신규 합성 24과제에서 문법 제약은 완주 7/24→15/24, 상태 정확 7/24→22/24를 기록했습니다. 회복 8개·손실 0개와 함께 추가 실행 비용과 남은 실패도 추적했습니다. 완주 수가 늘면서 전체 실행 비용도 늘었습니다. 실패를 포함한 24개 실행의 비용을 완주 수로 나누면 입력 토큰은 약 3.62%, 출력 토큰은 12.36%, 한 과제 실행 시간(episode wall)은 11.27% 증가했습니다.

확대 기준 이탈: 설정 점검용 DEV에서 BF16 KV+C0의 완주는 5/8로 원안의 6/8 기준에 미달했습니다. 당시 로컬 계획과 실행 코드가 이 성공 조건을 누락하고 완료·동결·봉인만 검사해 P2를 진행했습니다. 별도 면제 승인 기록은 확인하지 못했습니다. 이 경위를 공개하며, P2는 해당 기준을 통과한 확인시험이 아닌 신규 합성 과제의 탐색적 측정으로 제시합니다.

[Report](../../cases/017-agent-contract-cost-bench/REPORT.ko.md) · [CPU command and scope](../../cases/017-agent-contract-cost-bench/REPRODUCE.ko.md) · [Strict parser](../../cases/017-agent-contract-cost-bench/src/agent_contract_cost/contracts.py) · [Independent auditor](../../cases/017-agent-contract-cost-bench/src/agent_contract_cost/audit.py)

<!-- case018-entry -->
## Case018 — 상태 복구와 최종 보고는 함께 맞을까?

재고의 잘못된 대상·중복·부분 차감을 복구한 뒤 저장 상태와 최종 수량 보고를 함께 검사했습니다. 신규 합성24과제에서 BF16 KV는 두 정책 모두 Q24/W0, NVFP4 KV는 Q19/W5였습니다. 최신 검증 가드가 부적격 final 수용을 막아도 복구·정확 보고가 개선되지는 않았습니다. 현재 수량 재관측과 최종 보고의 관계는 별도 사후 관측 분석입니다.

[Report](../../cases/018-agent-state-recovery/REPORT.ko.md) · [CPU replay](../../cases/018-agent-state-recovery/REPRODUCE.ko.md)
