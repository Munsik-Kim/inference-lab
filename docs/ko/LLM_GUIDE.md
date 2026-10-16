# 16GB GPU에서 LLM 사용하기

[English summary](../en/LLM_GUIDE.md) | 한국어 · [홈](../../README.ko.md)

내 PC에서 모델을 쓰려면 GPU 메모리에 올라가는지, 답변에 얼마나 걸리는지, 내 업무를 정확히 수행하는지를 확인해야 합니다. RTX 5080에서 저장한 Qwen 측정 기록으로 이 질문들을 살펴보는 안내서입니다. 공개 가중치 모델은 학습한 숫자 배열인 가중치를 공개한 모델이며, 이용 조건은 별도로 적용됩니다.

채용·기술 협업 관점에서 연결된 코드는 모델 파일 변환·재시작 검사, 요청 비용 측정, 답변·실제 도구 작업 평가를 보여줍니다. 로컬 AI 실행·모델 최적화·품질 회귀 검사와 관련된 작업이며, [역량과 코드 근거](PORTFOLIO.md)에서 논의 가능한 범위를 볼 수 있습니다.

현재 상세 모델은 **Qwen3-4B-Instruct-2507** 하나입니다. [온라인 안내서](https://munsik-kim.github.io/inference-lab/ko/llm-guide.html)에서 다음 페이지로 이어집니다. 같은 정적 빌드를 로컬에서도 열 수 있습니다.

| 빌드 안의 경로 | 역할 |
|---|---|
| [안내서 첫 화면](https://munsik-kim.github.io/inference-lab/ko/llm-guide.html) | 내 문제에서 시작하는 안내서 첫 화면 |
| [Qwen3-4B 상세](https://munsik-kim.github.io/inference-lab/ko/qwen3-4b.html) | 모델·변환해 저장한 모델 파일(artifact)·실행 경로·품질·실패·완화 시도·코드와 원자료 |
| [측정 방법](https://munsik-kim.github.io/inference-lab/ko/measurements.html) | 시간·메모리·정답·실행 상태와 표본 수를 읽는 공통 기준 |
| `en/llm-guide.html`, `en/qwen3-4b.html`, `en/measurements.html` | 영어 소개와 핵심 요약. 완전한 영어 상세 번역은 제공하지 않음 |

[로컬 빌드와 열기](../../presentation/README.md#practical-guides)에서 실행 명령을 확인하세요. GPU나 모델 다운로드 없이 읽을 수 있습니다. 기존 Case URL·anchor·한영 이동은 유지합니다.

판단에 앞서 경로를 구분합니다. [Case008 Q](../../cases/008-build-reconstruct-reload/REPORT.ko.md)와 [Case009](../../cases/009-q-serving-quality/REPORT.ko.md)는 기존 압축 방법인 GPTQ로 만든 파일과 vLLM의 실제 저비트 실행을 확인했습니다. [Case013](../../cases/013-dualquant-reference-quality/REPORT.ko.md)과 [후속](../../research/agent_tool_contract_repair_20261014/README.ko.md)은 압축 값을 16비트 숫자 형식인 BF16으로 복원해 품질을 보는 참조 계산(reference)입니다. D16은 INT4 가중치·BF16 입력, D4는 같은 가중치·INT4 입력을 사용합니다. 압축 상태로 GPU에서 직접 계산하는 속도를 측정한 결과가 아닙니다. 압축 연산 전체 모델의 수치 검사 실패, 품질 프로그램의 비정상 종료와 설정 조정에 쓰던 기존 문제(DEV)의 남은 실패도 모델 페이지에 표시합니다.

Case008 R은 더 작은 Qwen3-0.6B에서 한 블록의 일부를 제거하고(pruning) 남은 가중치를 보정하는 구현입니다. Qwen3-4B 양자화 보정의 한 단계로 합치지 않습니다. 일반적인 지표 설명은 [기존 용어집](GLOSSARY.md), CPU 재확인은 [실행 안내](GETTING_STARTED.md), 이전 사례는 [사례 안내](CASEBOOK.md)로 이어집니다.

본문은 `presentation/guides/`에서 관리하고, 주요 수치는 원자료를 읽는 빌더에서 넣습니다. 같은 수치를 이 Markdown에 따로 복사하지 않습니다. 출처 commit·파일별 SHA256·필드·정확한 값은 빌드의 `data/practical-guide.json`과 `build_manifest.json`에서 확인할 수 있습니다.

2026-10-15: 기존 공개 기록을 연결한 안내서 추가. 새 GPU 실험·모델 다운로드·양자화·벤치마크 재실행 없음. 실제 독자 평가는 미실행입니다.

<!-- case014-entry -->
## 별도0.6B 보정 연구

[Case014](../../cases/014-quantization-compensation/README.ko.md) · [Pages](https://munsik-kim.github.io/inference-lab/ko/case014.html#overview). Qwen3-0.6B의 압축 오차를 줄이도록 가중치를 조정하고 문장 예측 품질을 확인한 별도 연구입니다. PG19의 이득과 ArXiv의 한계는 위 Qwen3-4B 성능표에 합산하지 않습니다.
