# Case015 P2 — 같은 정보를 받은 지점에서 JSON 복구를 비교할 준비

Case015 A1은 **합성 구매 기록을 읽고 → 수량과 개당 가격을 곱하고 → 파일에 합계를 저장하고 → 저장 결과를 확인하는** 실험입니다. 실제 주문이나 결제 시스템을 사용하지 않습니다. 이번에는 그중 ‘조회와 계산 결과를 이미 받은 뒤, 저장할 도구 호출을 작성하는 순간’으로 돌아가 오류를 확인했습니다. 이 순간의 대화·관측·파일 상태를 함께 저장한 것이 **체크포인트**입니다.

기존 A1의 D16 실패 8건은 도구에 전달하는 JSON의 따옴표·금액 키가 깨져 저장 도구를 실행하지 못했습니다. 출력 길이는 43–48토큰이고 모두 EOS로 끝났습니다. 192토큰 상한이나 모델·도구 호출 예산을 다 쓴 실패는 아니었습니다. 같은 8지점에서 BF16/RTN/D16이 받은 입력과 관측은 같았습니다.

**이번에 끝낸 것은 CPU 준비와 사전 등록입니다. 새 모델 응답이나 복구 성공 결과는 없습니다.**

| 구분 | 이번 상태 |
|---|---|
| 기존 A1 96개 기록의 응답 재파싱·도구 실행·최종 판정 재생 | PASS |
| 4개 선택된 fixture × API 2종의 공통 체크포인트 | 8개, 방법별 입력 24개 대응 확인 |
| 로컬 토크나이저로 입력 재계산 | 8개 고유 입력 / 저장된 24개 방법별 해시 일치 |
| 저장된 출력 토큰 복호화 | 463개 원문·파서 입력 일치; 새 생성은 아님 |
| 문법 제약 | 실제 토크나이저의 CPU 마스크 검사 통과; 모델 연결 미실행 |
| 새 복구 C0 / G / N / E | 모두 NOT_RUN |

등록한 대조는 다음과 같습니다. **C0**는 원래 입력·정책, **G**는 첫 응답에만 도구 계약에서 만든 문법 마스크 적용, **N**은 같은 새 C0 파싱 실패 뒤 중립 문장으로 한 번 재시도, **E**는 그 문장에 파싱 오류 설명만 더한 한 번의 재시도입니다. G도 틀린 ID·금액을 문법상 허용합니다. N/E는 도구가 실행되지 않은 같은 새 실패에서만 갈라지며 기존 D16 응답을 다른 모델의 새 응답으로 쓰지 않습니다.

모델 없이 확인하려면 저장소 루트에서:

```bash
P2=cases/015-tool-interface-sensitivity/followups/json-recovery-p2-v1
export PYTHONPATH="$P2/src" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=''
python3 -B -m json_recovery_p2.cli validate
python3 -B -m json_recovery_p2.cli --help
```

이후 [CPU 재현 명령](REPRODUCE.ko.md)으로 새 외부 출력 폴더에 원자료를 다시 재생할 수 있습니다.

- [한국어 보고서와 실패 표](REPORT.ko.md)
- [고정 protocol](specs/protocol.json), [체크포인트](data/checkpoints.json), [모델 입력 allowlist](data/model_request_snapshots.json)
- [새 소스·테스트 hash](provenance/code_freeze.json), [재사용한 원 소스](provenance/source_manifest.json)
- [실행·검사 기록](results/cpu_preparation/command_receipts.json), [남은 gate](results/cpu_preparation/gates.json)
- [복구 결과: 전 칸 NOT_RUN](results/cpu_preparation/recovery_results.json)

공개 설명·원 P1/A1·사이트는 바꾸지 않았습니다. Git stage/commit/push/PR/merge/배포도 수행하지 않았습니다. 이 후속 폴더를 추가했으므로 과거의 닫힌 파일 목록 검사는 원래 457개 파일만 바이트 그대로 복원한 view에서 실행합니다. 원래 manifest를 바꾸어 통과시키지 않습니다.
