# Case 008 — 양자화 모델 제작·재실행과 작은 MLP의 계산 복구

[English](README.md) | [한국어](README.ko.md)

RTX 5080에서 Qwen 모델을 바꾸어 저장하고 다시 실행하는 두 경로를 만들었습니다. **Q**는 4B 모델을 GPTQ W4A16으로 변환하고 packed 저장본을 검사해 새 vLLM 프로세스로 실행합니다. **R**은 0.6B 모델의 이미 축소한 MLP에서 출력 가중치만 보정하고 층별 크기를 복원하는 로더로 실행합니다. 두 트랙의 모델과 실행기는 서로 다릅니다.

**[구조화 보고서부터 읽기](REPORT.ko.md):** 배경 → 가설 → 이론 → 방법 → 실험 → 실험 결과 → 결과 분석 → 결론 → 레퍼런스. 설정 이름과 지표를 먼저 설명하고, 데이터에서 무엇을 확인할 수 있는지 정리했습니다.

| 트랙 | 실제 관측 | 해석 |
|---|---|---|
| Q: 가중치 표현과 재실행 | Safetensors 8,044,982,000 → 2,651,839,568바이트; 두 설정 모두 137/192 정답 | 파일과 기록된 상주 가중치가 감소. 정답 손실 1개·새 정답 1개이며 전체/선택지 NLL은 반대 방향으로 변화. |
| R: 고정된 작은 MLP와 보정 | 세 구조에서 삭제에 따른 제곱 출력 오차의 94.1–95.4% 회복 | 보정 전후 shape는 같음. 짧은 합성 192입력에서 관측했고 정답 점수 변화는 혼재. |

요청 속도 구간은 가속을 확정하지 못했습니다. R의 회복률은 국소 제곱오차 감소율이며 정답률이 아닙니다. 코드 과제에서는 두 트랙 모두 상수 선택이 관측됐습니다. [결과 분석](REPORT.ko.md#analysis)에서 보기 확률질량·과제별 결과·생성 규칙의 단서를 확인할 수 있습니다. 연구 실행은 `COMPLETED`, 배포는 `NOT_ASSESSED`입니다.

## 직접 확인하기

- [한국어 보고서](REPORT.ko.md) · [원래 방법론(영어)](METHODS.md) · [분석 안내(영어)](ANALYSIS.md).
- [CPU 검산·GPU 재현의 실행 범위(영어)](REPRODUCTION.md) · [Modelpack 소스](../../tools/modelpack/) · [구현 기여와 코드 투어](PORTFOLIO.md).
- [저장 결과 탐색기](demo/index.html): 내려받아 로컬 브라우저로 여세요. 기존 UI는 영어이며 모델을 실행하지 않습니다. GitHub의 HTML 소스 미리보기는 실행 화면이 아닙니다.
- [코드·recipe·측정 자료 ZIP](../../downloads/case008_build_reconstruct_reload_reviewed_publication_v2.zip) · [ZIP 메타데이터·체크섬](../../downloads/case008_build_reconstruct_reload_reviewed_publication_v2.json). 큰 모델 가중치와 private 활성값·전체 logits는 포함하지 않습니다.
- [Q 점수 원자료](results/raw/Q/model_records.json) · [R 점수 원자료](results/raw/R/model_records.json) · [원래 요약](results/derived/summary.json) · [같은 기록의 사후 진단](publication/posthoc/metrics.json).

[공개본 검사·원본 복원 안내(영어)](publication/README.md)는 수정된 설명과 원래 실험을 나누어 검증합니다. 입력·source 동결, 과학 코드, 원래 그림·HTML·판정은 보존했습니다. 원래 `RUN_STATE.json`은 당시 검토 snapshot의 기록이며 [현재 패키지 상태](publication/status.json)와 다릅니다.

## 기여 및 출처

Qwen/Transformers는 모델, GPTQ/LLM Compressor는 양자화, compressed-tensors/vLLM/Marlin은 저장·실행 구현을 제공합니다. 이 프로젝트는 저장본 검사, 층별 구조 복원, 고정 집합 ridge 보정과 paired 결과 도구를 연결했습니다. Codex가 작업을 지원했습니다. [NOTICE](NOTICE.md) · [라이선스](LICENSE).
