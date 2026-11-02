# 처음 보는 분을 위한 DIOVA 안내

[English](../en/START_HERE.md) | 한국어 · [홈](../../README.ko.md)

[한영 홈페이지](https://munsik-kim.github.io/inference-lab/ko/index.html)는 설치 없이 결과를 읽는 입구입니다. 다음 세 가지 중 원하는 일을 고르세요.

## 기술 역량 확인

[대표작 A/B/C](PORTFOLIO.md) → 본인이 한 판단 → 코드와 테스트 순서입니다. A는 [Case008 Q 저장·재로딩](https://munsik-kim.github.io/inference-lab/ko/case008.html#track-q) → [Case009 실제 요청](https://munsik-kim.github.io/inference-lab/ko/case009.html#serving)로 이어집니다. C는 [Case017 보고서](../../cases/017-agent-contract-cost-bench/REPORT.ko.md)와 [Case018 상태·최종 보고](CASEBOOK.md#case-018)를 나누어 읽습니다. [앞선 Case013 후속](../../research/agent_tool_contract_repair_20261014/README.ko.md)은 별도 연구입니다.

## 도구 사용

[작은 CPU 예제](https://munsik-kim.github.io/inference-lab/ko/compare.html#run)로 두 답변 JSONL을 검사하고 비교하세요. `report.md`가 요약, `pairs.csv`가 질문별 변화, `report.json`이 프로그램용 결과입니다. 자신의 자료에는 같은 질문 식별자와 정답·채점 규칙이 필요합니다. [내 자료 준비](https://munsik-kim.github.io/inference-lab/ko/compare.html#own-records).

지원 과제의 평가 도구가 필요하면 [inference-engineering-kit](https://github.com/Munsik-Kim/inference-engineering-kit), 실제 GPU 모델 실행 조건은 [모델 안내](LLM_GUIDE.md)로 이어집니다.

## 연구 검토와 검산

[공개 연구 지도](https://munsik-kim.github.io/inference-lab/ko/index.html#research)에서 질문·관측·적용 범위를 읽고 정본으로 이동하세요. 이 저장소의 숫자는 [CPU 기록 검산](GETTING_STARTED.md#cpu-checks), 화면은 [로컬 열기](GETTING_STARTED.md#offline-explorers)로 확인할 수 있습니다. 저장자료의 재계산은 새 GPU 실험이나 외부 심사가 아닙니다.

## 전체 사례 바로 찾기

설정과 입력이 다른 연구들입니다. 처음부터 모두 읽을 필요는 없습니다. [검색·주제 필터](https://munsik-kim.github.io/inference-lab/ko/index.html#cases) · [숫자 읽는 법](https://munsik-kim.github.io/inference-lab/ko/measurements.html) · [용어집](GLOSSARY.md).

<a id="1-어떤-문제를-다루나요"></a>
<a id="2-온라인-화면부터-읽어보세요"></a>
<a id="3-어느-안내를-읽으면-되나요"></a>
<a id="3-참고-안내"></a>
<a id="4-전체-사례는-어떻게-이어지나요"></a>
<a id="5-다운로드하거나-직접-검사하려면"></a>
<a id="6-한-층의-계산을-바꾼-관련-구현"></a>
<a id="api-이름-변경-회귀시험"></a>
<a id="case015-cpu"></a>
<a id="case017--도구-호출-규칙과-업무-완주-비용"></a>
<a id="case018--상태-복구와-최종-보고는-함께-맞을까"></a>
<a id="case019--손실은-그대로인데-학습은-진행될까"></a>
<a id="심화-연구-같은-int4-저장량의-품질-보정"></a>
- [001 — GPU에서 모델이 실행되지 않는다면?](CASEBOOK.md#case-001)
- [002 — 적은 비트로 저장하면 메모리와 시간이 줄까?](CASEBOOK.md#case-002)
- [003 — 간단해진 계산은 원래 값과 얼마나 비슷할까?](CASEBOOK.md#case-003)
- [004 — 핵심 연산이 빠르면 호출 전체도 빨라질까?](CASEBOOK.md#case-004)
- [005 — 속도와 오차를 함께 만족하는 설정이 있을까?](CASEBOOK.md#case-005)
- [006 — 어느 질문의 답이 바뀌는가?](CASEBOOK.md#case-006)
- [007 — 작은 계산 블록에 어느 그룹을 남길까?](CASEBOOK.md#case-007)
- [008 — 바꾼 모델을 저장하고 다시 실행할 수 있을까?](CASEBOOK.md#case-008)
- [009 — 긴 decode와 동시 요청에서 비용은 어떻게 바뀔까?](CASEBOOK.md#case-009)
- [010 — 저비트 상태를 실패 후에도 다시 실행할 수 있는가?](CASEBOOK.md#case-010)
- [011 — 같은 상태에서 판독기만 바꾸면?](CASEBOOK.md#case-011)
- [012 — 이미지 생성 시간을 어디에 더 쓸까?](CASEBOOK.md#case-012)
- [013 — 가중치 오차가 줄면 실제 답도 좋아질까?](CASEBOOK.md#case-013)
- [014 — 같은 4비트 저장량에서 품질을 개선할 수 있는가?](CASEBOOK.md#case-014)
- [015 — 도구 이름만 바뀌어도 AI가 같은 일을 끝낼까?](CASEBOOK.md#case-015)
- [016 — 실제 NVFP4 KV 저장과 동시 요청 비용은?](CASEBOOK.md#case-016)
- [017 — 도구 호출 규칙과 업무 완주 비용](CASEBOOK.md#case-017)
- [018 — 상태 복구와 최종 보고는 함께 맞을까?](CASEBOOK.md#case-018)
- [019 — 손실은 그대로인데 학습은 진행될까?](CASEBOOK.md#case-019)
- [020 — 작게 보관하면 실행도 빨라질까?](CASEBOOK.md#case-020)
- [021 — 같은 INT4인데 왜 값이 달라질까?](CASEBOOK.md#case-021)
