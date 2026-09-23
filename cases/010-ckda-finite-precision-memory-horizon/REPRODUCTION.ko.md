# Case 010 근거 재현 안내

[English](REPRODUCTION.md) · [Case 홈](README.ko.md) · [버전 대응표](VERSION_MAP.md)

확인하려는 대상에 따라 경로를 선택합니다. 통합본에는 저장된 compact prediction과 원래 두 연구 subtree가 있습니다. 모델 checkpoint, upstream checkout, 전체 private state이력은 별도 자료입니다.

## 1. 구현과 결과 읽기

[통합 보고서](REPORT.ko.md), [파생 요약](summary/project.json), [실패 보존 OnlineAdapter](versions/v2/source/online_v2.py), [새 프로세스 계약 테스트](versions/v2/tests/test_online_v2.py)부터 볼 수 있습니다. 현재 실행 계약은 v2이며 단계 A의 코드는 역사적 reference로 남습니다.

[Snapshot manifest](provenance/snapshot_manifest.json)는 원래 경로와 보존 경로를 bytes·SHA256과 함께 대응합니다. v1의 1,075 파일과 v2의 387 파일을 포함하며 각 tree의 과거 manifest는 수정하지 않습니다. 원래 `LOCAL_REVIEW`·미게시 상태는 당시 snapshot 기록이고, 통합본의 branch·PR 상태와 별개입니다.

## 2. 모델 없이 검산하기

NumPy가 있는 기존 Python 환경을 사용합니다. Wrapper는 Torch·CUDA·모델 다운로드·학습 checkpoint를 요구하지 않습니다. Bytecode 쓰기를 끄고 모든 출력은 보존 case 밖에 둡니다. 저장소 clone에서:

```bash
cd cases/010-ckda-finite-precision-memory-horizon
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=
python -B scripts/verify_unified.py
python -B -m unittest discover -s tests -v
python -B scripts/check_cpu.py --output /PATH/TO/NEW_CASE010_CHECK
```

마지막 경로는 **아직 존재하지 않는 외부 디렉터리**로 바꿉니다. CPU wrapper는 원본 identity를 검사하고 원래 workspace 구조를 복원한 뒤 v1 publication·scalar·supplement 감사와 v2의 독립 fresh 결과 감사를 실행합니다. 이어 원래 합성 failure-aware byte-codec 테스트를 실행합니다. 이 테스트에는 별도 child process가 포함되지만 학습 모델은 필요하지 않습니다. 합성 fixture를 학습 CKDA의 기억 성능 예제로 해석하지 않습니다.

표시 요약을 다시 만들어도 원본 snapshot 변경은 verifier가 거절합니다. 스칼라 감사는 저장 prediction과 정수 gold를 읽어 파일 identity·pairing·최초 실패·RMST·horizon을 확인합니다. 원래 forward 나 비공개 최종 recurrent tensor를 다시 계산한 것은 아닙니다.

단계를 나누려면 통합 case 디렉터리에서 다음을 실행합니다.

```bash
python -B scripts/derive_summary.py --output /PATH/TO/NEW_PROJECT.json
python -B scripts/restore_workspace.py --output /PATH/TO/NEW_RESTORED_WORKSPACE
```

첫 명령은 원자료에서 작은 표시용 데이터를 다시 만듭니다. 선택 인자 `--figure /PATH/TO/NEW_FIGURE.png`에는 Matplotlib도 필요합니다. 두 번째 명령은 원형을 수정하지 않고 다음 과거 경로를 복원합니다.

```text
NEW_RESTORED_WORKSPACE/
  cases/010-ckda-finite-precision-memory-horizon/   # v1
  research/case010-failure-aware-v2/              # v2
```

원래 source와 상대경로 검사는 이 복원 구조에서 수행합니다. 예를 들어 `RESTORED`를 새 workspace 경로로, `AUDIT`을 별도의 새 출력 경로로 지정한 뒤:

```bash
python -B "$RESTORED/cases/010-ckda-finite-precision-memory-horizon/scripts/verify_publication.py"   --root "$RESTORED/cases/010-ckda-finite-precision-memory-horizon"
python -B "$RESTORED/research/case010-failure-aware-v2/analysis/audit_v2.py"   --results "$RESTORED/research/case010-failure-aware-v2/results/fresh"   --output "$AUDIT"
```

바깥 통합 tree에 과거 exact-inventory verifier를 그대로 적용하거나 새 위치에 맞춰 과학 checksum을 재작성하지 않습니다. [v1 안내](versions/v1/REPRODUCTION.md)와 [v2 안내](versions/v2/REPRODUCTION.md)는 당시 상대경로를 유지하므로 먼저 복원해야 합니다.

### 작은 새 프로세스 예제

```bash
python -B scripts/restart_demo.py
```

NumPy만 사용하는 예제입니다. 두 합성 stochastic stream 중 하나에 수치 오류를 넣고 실제 bytes를 저장한 뒤 새 Python 프로세스에서 나머지 입력을 이어갑니다. Active 1개와 terminal 1개, suffix 일치, 최종 bytes 일치를 출력하며 child가 Torch를 읽지 않았는지도 확인합니다. 학습 CKDA 평가가 아닌 byte-codec 계약 시연입니다.

## 3. 실제 연구 재실행은 별도 경로

원래 안내에는 모델 의존 replay, 학습 기록, 고정 checkpoint 정밀도 진단, fresh 평가와 시간 측정이 있습니다. 고정 upstream `ef9d108d1692387cae37f5b2d539a71826a127c1`, 호환 환경, [v2 protocol](versions/v2/protocol_v2.json)의 hash와 일치하는 세 private checkpoint가 필요합니다. 통합본은 가중치를 배포하지 않습니다. 모델 추론·학습·calibration·TEST 재평가·GPU 실행은 이번 통합 범위 밖입니다.

과거 전체 테스트에는 모델·upstream의존 검사가 있습니다. 모델 없는 wrapper가 통과했다고 전체 검사가 실행됐다고 해석하지 않고 실제 count·skip·error를 확인합니다. 과거 receipt의 v1 141개와 v2 76개는 당시 실행 기록이며 이번 통합의 테스트 수가 아닙니다.

## 근거의 범위

- v1 스칼라 재계산은 기록된 toy125개, 원래 learned78개, 보충 12개를 다룹니다. 독립 모델 수나 독립 시퀀스 수가 아니라 record 수입니다.
- v2 fresh는 15개 셀이며 각 셀의 입력 1024 개·길이 2048입니다. 설정들은 같은 입력 ID와 원래 세 checkpoint를 공유합니다. Historical·diagnostic 입력은 fresh N에 더하지 않습니다.
- 공개 payload 길이와 hash는 identity·장부를 검증합니다. Private 최종 hidden-state body를 읽거나 checkpoint 기반 재시작을 다시 실행하려면 별도 자료가 필요합니다.
- 브라우저 에뮬레이션·CPU 검사·원격 CI는 각 범위의 결과입니다. 새 GPU 실험이나 실제 iPad 검사를 대신하지 않습니다.

## 이번 통합에서 실제 재실행한 검사

[통합 검사 receipt](provenance/integration_checks.json)에 현재 실행을 기록합니다. 통합 계약 테스트 18개와 원래 v2의 codec 전용 합성 테스트 17개가 skip 없이 통과했습니다. 새 프로세스 예제는 active 1개와 terminal 1개를 유지하며 suffix·최종 bytes가 일치했습니다. v1 publication 검사는 스칼라 감사 전후에 통과했고 toy 125개·learned 78개·보충 12개 record를 검산했습니다. v2 독립 감사는 fresh 15개 셀을 확인했습니다. 저장 자료와 합성 CPU 검사의 결과입니다. 과거 전체 141/76개 suite, 학습 checkpoint replay, fresh 추론, 시간 재측정은 이번에 다시 실행하지 않았습니다.
