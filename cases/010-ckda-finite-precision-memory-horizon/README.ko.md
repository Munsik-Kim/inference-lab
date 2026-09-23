# Case 010 — 저정밀 recurrent state의 저장·재시작과 기억 수명

[English](README.md) · [DIOVA 홈](../../README.ko.md) · [통합 보고서](REPORT.ko.md) · [CPU 검산](REPRODUCTION.ko.md)

반복 갱신되는 모델 상태(recurrent state)를 실제 저비트로 저장하고, 수치 실패와 난수를 보존해 새 프로세스에서 이어 실행하는 도구를 구현했습니다. 같은 저장 한도에서 올바른 기호 상태를 연속해서 읽는 길이를 비교하고, 연산 정밀도·상태 변화·CPU 계산 비용을 함께 분석합니다.

## 구현한 기능

| 기능 | 구현과 검사 |
| --- | --- |
| 상태 코드·scale·난수를 실제 바이트로 압축 | [PackedCodec](versions/v2/source/v1_reference/codec/packed.py) · [packing 테스트](versions/v1/tests/) |
| 자신의 저장 잔차를 다음 전이로 전달하고 혼합정밀도와 비교 | [OnlineAdapter](versions/v2/source/online_v2.py) · [CAL 기반 설정 구성](versions/v2/source/arms.py) |
| cursor·terminal 원인·최초 수치 실패 위치를 저장하고 마지막 정상 body·RNG 유지 | [실패 보존 adapter](versions/v2/source/online_v2.py) · [계약 테스트](versions/v2/tests/test_online_v2.py) |
| 새 프로세스에서 같은 나머지 입력을 이어 실행 | [평가 checkpoint](versions/v2/source/checkpoint.py) · [별도 프로세스 검사](versions/v2/tests/restart_online_v2_child.py) |
| 최초 판독 실패·평균 연속 정답 길이·위험 상한 재계산 | [지표 계산](versions/v2/source/metrics.py) · [독립 스칼라 검산기](versions/v2/analysis/audit_v2.py) |
| 고정 계수 FP32/FP64와 시간별 상태 변화 분석 | [정밀도 경로](versions/v2/source/precision.py) · [저장된 진단 스칼라](versions/v2/results/diagnostic-summary/) |

**기술:** Python, NumPy, CPU PyTorch, 실제 packed byte buffer, checksum을 가진 기록, Matplotlib 그림. 학습 실험은 작은 S3 Complex KDA 한 층과 12개 head를 사용하며, 두 단계 모두 같은 세 checkpoint를 비교합니다.

## 확인한 대표 관측

- **stream별 직렬화 state 크기 12,305 → 3,137 bytes:** native FP32 대비 INT8에서 **74.5% 감소**했습니다. 세 checkpoint 각각 평균 연속 정답 길이의 점추정 차이는 1 token 미만이었습니다. 저장 상태의 크기와 관측 수명을 비교한 값이며, 전체 RAM·VRAM 감소나 품질 동등성 검정은 아닙니다. [저장량·native 비교표](REPORT.ko.md#native-int8).
- **실패 이후에도 같은 상태로 재시작합니다.** v2는 함수 호출과 프로세스가 바뀌어도 terminal·cursor·RNG를 유지합니다. 기존 검증 기록에는 19개 대조 셀과 실제 수치 실패 직후의 네 재시작 사례가 있습니다. [재시작 계약과 근거](REPORT.ko.md#restart).
- **같은 예산의 결과는 checkpoint에 따라 달랐습니다.** Rank2−Mixed5/6의 평균 연속 정답 길이 변화는 +9.27 / −20.76 / +4.91 token입니다. N=128의 총 저장량은 각각 292,633 / 292,469 bytes입니다. [다섯 설정·구간·CPU 비용](REPORT.ko.md#fresh).

## 하나의 프로젝트, 두 연구 단계

**단계 A / v1 — 저장 방식 탐색과 비교군 강화.** 실제 packing, 잔차 전달, 균일·혼합정밀도 기준선과 최초 실패 분석을 checkpoint당 512개 입력에서 평가했습니다. 같은 TEST를 다시 쓰는 보충 예산 감사에서 가능한 혼합정밀도 배치 네 개를 추가했습니다.

**단계 B / v2 — 실패 상태 보존과 fresh 입력 검증.** 실패를 잃던 재시작 경계를 직렬화 계약으로 수정했습니다. 이어 같은 세 checkpoint와 고정된 다섯 설정을 각각 1,024개 새 입력에서 비교했습니다. 재시작 구현의 수정과 기억 길이의 개선 여부는 별도 결과입니다.

[v2 현재 실행 코드](versions/v2/source/online_v2.py) · [단계 A 원래 보고서](versions/v1/REPORT.ko.md) · [단계 B 원래 보고서](versions/v2/REPORT.ko.md) · [버전 대응표](VERSION_MAP.md)

## CPU로 확인하기

이 case 디렉터리에서 NumPy가 있는 Python 환경을 사용합니다.

```bash
python -B scripts/verify_unified.py
python -B scripts/check_cpu.py --output /PATH/TO/NEW_CASE010_CHECK
```

두 번째 명령은 case 밖에 원래 디렉터리 구조를 복원한 뒤 저장된 결과를 검산하고 합성 재시작 계약을 검사합니다. 학습 checkpoint를 읽거나 모델 추론을 실행하지 않습니다. [요구 환경·명령·검산 범위](REPRODUCTION.ko.md).

[통합 보고서](REPORT.ko.md) · [파생 요약](summary/project.json) · [원본 inventory](provenance/snapshot_manifest.json) · [포트폴리오](PORTFOLIO.md) · [기여·라이선스](NOTICE.md)

Complex KDA는 모델·과제·원래 출력 계산을 제공합니다. DIOVA는 저장 adapter, 재시작 계약, 비교 실험과 검산 도구를 구현했으며 Codex가 구현·문서화를 지원했습니다. 각 snapshot의 원래 고지는 함께 보존합니다.
