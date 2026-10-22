# Case018 — 재고 복구와 완료 보고를 따로 검사했습니다

잘못 옮긴 재고, 두 번 처리한 이동, 일부만 차감한 기록을 AI가 되돌리게 했습니다. 재고가 올바르게 복구됐는지와 마지막 보고까지 맞는지를 따로 검사했습니다.

재고·처리 원장 환경과 검사 도구를 구현해 저장된 값, 무관 기록의 보호, 수정 뒤 실제 받은 검증 결과와 최종 답을 추적했습니다. 복구가 끝나면 평가기가 두 가지 정해진 후속 절차를 실행하고 정상 checkpoint의 결과와 비교합니다.

**상태·후속 검사 통과(Q)**는 저장된 업무 값이 복구 목표와 같고, 복구 과정의 보호 계약과 두 후속 절차를 모두 만족한 경우입니다. **검증·완료 보고까지 충족(W)**은 여기에 최신 성공 검증의 실제 관측과 시스템이 수용한 정확한 완료 응답을 더합니다.

| KV 저장 | 완료 응답 정책 | 상태·후속 검사 통과 (Q) | 검증·완료 보고까지 충족 (W) |
| --- | --- | --- | --- |
| BF16 | 기본 수용 (R0) | 24/24 | 0/24 |
| BF16 | 최신 검증 가드 (R1) | 24/24 | 0/24 |
| NVFP4 | 기본 수용 (R0) | 19/24 | 5/24 |
| NVFP4 | 최신 검증 가드 (R1) | 19/24 | 5/24 |

각 행은 같은 24개 합성 과제이며 총 96개 episode입니다. 양쪽 Qwen 가중치는 BF16으로 고정했습니다. KV는 앞선 토큰의 attention 계산에 쓰는 key/value 저장 공간입니다.

BF16 KV는 두 정책 모두 24/24에서 상태·후속 검사를 통과했지만, 검증·정확한 완료 보고까지 충족한 수는 0/24였습니다. NVFP4 KV는 각각 19/24, 5/24였습니다. 재고를 복구한 뒤에도 처음 읽은 수량을 마지막 답에 적은 사례가 반복됐습니다.

최신 검증 가드는 완료 응답 수용을 차단했지만, 복구·정확한 완료 건수는 늘리지 못했습니다. NVFP4의 출력 토큰은 18.69% 늘었습니다.

[실제 사건 보기](REPORT.ko.md#원문-사건) · [결과와 방법](REPORT.ko.md) · [모델 없이 검산하기](REPRODUCE.ko.md)

## 직접 검산

```bash
python3 -B cases/018-agent-state-recovery/publication/run_checks.py --output /tmp/case018-public-check-new --tests
```

Python 3.12+ 표준 라이브러리로 새 output 경로에 CPU 기록을 재생합니다. 모델 다운로드가 필요하지 않습니다.

[CPU 기록 재생 · main](https://github.com/Munsik-Kim/inference-lab/actions/workflows/case018-checks.yml?query=branch%3Amain)

직접 구현한 부분은 재고·처리 원장, 보상·처리 해제 API, 엄격한 parser, 완료 가드, 생성 전 관측 추적, 별도 상태·후속 절차 감사, 사후 수량 분석과 공개 도구입니다. Qwen 모델·vLLM·FlashInfer KV 커널·xgrammar는 upstream을 사용했습니다.

[사후 분석 정의](publication/analysis/DEFINITION.json) · [120행 원장](publication/summary/episode_ledger.csv) · [출처·라이선스](NOTICE.md)
