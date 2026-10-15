# 16GB GPU에서 LLM 사용하기

[English summary](../en/LLM_GUIDE.md) | 한국어 · [홈](../../README.ko.md)

RTX 5080에서 공개 가중치 모델을 실행하려는 개발자와 모델 교체·업무 자동화를 검토하는 기술 담당자를 위한 안내서입니다. 모델을 메모리에 올리는 일, 요청을 빠르게 처리하는 일, 업무를 정확하게 수행하는 일은 따로 확인합니다.

현재 상세 모델은 **Qwen3-4B-Instruct-2507** 하나입니다. [온라인 안내서](https://munsik-kim.github.io/inference-lab/ko/llm-guide.html)에서 다음 페이지로 이어집니다. 같은 정적 빌드를 로컬에서도 열 수 있습니다.

| 빌드 안의 경로 | 역할 |
|---|---|
| [안내서 첫 화면](https://munsik-kim.github.io/inference-lab/ko/llm-guide.html) | 내 문제에서 시작하는 안내서 첫 화면 |
| [Qwen3-4B 상세](https://munsik-kim.github.io/inference-lab/ko/qwen3-4b.html) | 모델·artifact·실행 경로·품질·실패·완화·코드와 원자료 |
| [측정 방법](https://munsik-kim.github.io/inference-lab/ko/measurements.html) | 시간·메모리·정답·실행 상태와 표본 수를 읽는 공통 기준 |
| `en/llm-guide.html`, `en/qwen3-4b.html`, `en/measurements.html` | 영어 소개와 핵심 요약. 완전한 영어 상세 번역은 제공하지 않음 |

[로컬 빌드와 열기](../../presentation/README.md#practical-guides)에서 실행 명령을 확인하세요. GPU나 모델 다운로드 없이 읽을 수 있습니다. 기존 Case URL·anchor·한영 이동은 유지합니다.

판단에 앞서 경로를 구분합니다. [Case008 Q](../../cases/008-build-reconstruct-reload/REPORT.ko.md)와 [Case009](../../cases/009-q-serving-quality/REPORT.ko.md)는 GPTQ 저장본의 실제 저비트 실행을 확인했습니다. [Case013](../../cases/013-dualquant-reference-quality/REPORT.ko.md)과 [후속](../../research/agent_tool_contract_repair_20261014/README.ko.md)은 BF16으로 복원한 reference 품질 경로입니다. Packed 전체 모델 수치 검사 실패, 일부 품질 프로세스의 종료 실패와 남은 DEV 실패를 모델 페이지에 함께 표시합니다.

Case008 R은 다른 모델인 Qwen3-0.6B의 pruning 후 복원 구현입니다. Qwen3-4B 양자화 보정의 한 단계로 합치지 않습니다. 일반적인 지표 설명은 [기존 용어집](GLOSSARY.md), CPU 재확인은 [실행 안내](GETTING_STARTED.md), 이전 사례는 [사례 안내](CASEBOOK.md)로 이어집니다.

본문은 `presentation/guides/`에서 관리하고, 주요 수치는 원자료를 읽는 빌더에서 넣습니다. 같은 수치를 이 Markdown에 따로 복사하지 않습니다. 출처 commit·파일별 SHA256·필드·정확한 값은 빌드의 `data/practical-guide.json`과 `build_manifest.json`에서 확인할 수 있습니다.

2026-10-15: 기존 공개 기록을 연결한 안내서 추가. 새 GPU 실험·모델 다운로드·양자화·벤치마크 재실행 없음. 실제 독자 평가는 미실행입니다.

<!-- case014-entry -->
## 별도0.6B 보정 연구

[Case014](../../cases/014-quantization-compensation/README.ko.md) · [Pages](https://munsik-kim.github.io/inference-lab/ko/case014.html#overview). Qwen3-0.6B의 별도 품질 연구이며 위 Qwen3-4B 추천·성능표에 합산하지 않습니다.
