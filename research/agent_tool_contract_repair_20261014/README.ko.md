# Case 013 후속 — 도구 호출 오류를 막고 실제 작업을 확인하기

모델이 실제 등록명과 다른 함수 이름을 생성하는 문제를 대상으로, 오류 안내와 이름 제한을 독립적으로 구현했습니다. 기존 양자화 가중치는 그대로 읽고, 같은 8개 DEV 과제를 기본·안내 개선·이름 제한·두 수정 결합 조건에서 비교합니다. 기존 Case 013 결과는 보존합니다.

같은 기존 DEV에서 **D16은 6/8→8/8**, activation INT4를 추가한 **D4는 0/8→2/8**로 바뀌었습니다. 이름 제한 조건의 미등록 호출은 0회였습니다. D4의 나머지 6개 과제 실패는 남았습니다.

[English](README.md) · [구현 코드](guard.py) · [모든 조건](records/summary.json) · [정식 보고서](records/REPORT_KO.md) · [출처](NOTICE.md)

## 실제 결과

| 실행 조건 | D16 성공 | D4 성공 |
|---|---:|---:|
| 원래 실행 | 6/8 | 0/8 |
| 오류 안내만 개선 | 8/8 | 0/8 |
| 이름 제한만 적용 | 8/8 | 2/8 |
| 두 수정 결합 | **8/8** | **2/8** |

BF16/RTN 대조는 각각 8/8입니다. 오류 안내는 D16의 실패를 회복했지만 처음의 미등록 호출 4회는 남았습니다. 이름 제한과 결합 조건은 미등록 호출이 0회였습니다. 여기서는 이름 제한에 안내를 더한 추가 이득은 관측하지 못했습니다.

회복된 두 과제는 올바른 이름으로 patch와 validation을 실제 실행했습니다. **D16의 최종 JSON은 수정 전후 같았지만 실제 작업 성공 여부는 달랐습니다.** D4에는 도구를 읽지 않거나 잘못 답하는 다른 실패가 남았습니다. [원시 실패와 회복 사례](records/failure_examples.json) · [확인된 D16 설정](records/recommended_preset.json)

Qwen3-4B-Instruct-2507, RTX 5080, 복원된 BF16 reference 경로입니다. 80개 episode는 같은 8개 task의 10개 조건이며 독립 과제 80개가 아닙니다. 새로운 일반화·비열등성·packed serving 속도나 최대 문맥 이득을 판정하지 않습니다. [앞선 원인 진단](../agent_stability_20261013/README.ko.md) · [원형 Case013](../../cases/013-dualquant-reference-quality/README.ko.md)

## 저장 기록만 재계산

저장소 루트에서 모델 없이 실행합니다. 출력 디렉터리는 새 경로여야 합니다.

```bash
STUDY=research/agent_tool_contract_repair_20261014
python3 -B "$STUDY/verify_public.py" --study "$STUDY" --output /tmp/tool-repair-public-new.json
python3 -B "$STUDY/analyze.py" --result "$STUDY/records" --inputs "$STUDY/records/inputs" --output /tmp/tool-repair-audit-new --skip-external-preservation
diova-compare validate --input research/agent_tool_contract_repair_20261014/records/compare_inputs/D16-BASELINE.jsonl
diova-compare compare --baseline research/agent_tool_contract_repair_20261014/records/compare_inputs/D16-BASELINE.jsonl --candidate research/agent_tool_contract_repair_20261014/records/compare_inputs/D16-BOTH.jsonl --output /tmp/tool-repair-compare-new
```

외부 원본 보존 skip은 공개본에 없는 모델/case를 이번 재계산에서 검증하지 않았다는 뜻입니다. 포함 파일은 별도 공개 manifest로 검사하며 원 실행의 로컬 보존은 [검증 receipt](records/verification.json)에 남겼습니다. CLI 구간은 같은 DEV 기록의 기술적 계산입니다. [실행한 비교](records/compare_verification.json) · [검토 ZIP](../../downloads/case013_tool_contract_repair_review_v1.zip)

## 무엇을 바꿨나요?

- `FEEDBACK`: 이름 오류와 인자 오류를 구분하고, 이름 오류에는 실제 등록명을 알려줍니다. 정답 인자는 제공하지 않습니다.
- `ENUM`: 열린 `<tool_call>` JSON의 최상위 `name` 값만 등록명 중에서 생성하도록 제한합니다. 호출 여부·인자·최종 답은 모델이 결정합니다. 생성된 답을 사후에 교정하거나 이름 alias로 바꾸지 않습니다.
- `BOTH`: 두 수정을 함께 적용합니다.

이번 경로는 실행 계약을 보강한 시스템 비교입니다. DualQuant의 스케일이나 양자화 알고리즘을 바꾼 실험이 아닙니다. 이미 관측한 DEV 8개를 사용하며 새로운 일반화·비열등성·serving 속도 증거로 해석하지 않습니다.

## 실행

저장소 루트에서 모델 없는 작은 기능 검사:

```bash
python3 -B -m unittest discover -s research/agent_tool_contract_repair_20261014 -p 'test_*.py' -v
```

GPU 재실행에는 기존 연구 root, `.venv-reference`, 같은 Qwen3-4B revision, 원 r2 RTN/D16 artifact 각 252개와 입력이 필요합니다. 기존 폴더에 쓰지 않고 새 결과 경로를 지정합니다. 실제 실행한 진입점은 다음과 같습니다.

```bash
STUDY=research/agent_tool_contract_repair_20261014
python3 -B "$STUDY/prepare.py" --research "$RESEARCH_ROOT" --result "$NEW_RESULT"
"$RESEARCH_ROOT/.venv-reference/bin/python" -B "$STUDY/launch.py" --result "$NEW_RESULT" --method BF16
"$RESEARCH_ROOT/.venv-reference/bin/python" -B "$STUDY/launch.py" --result "$NEW_RESULT" --method RTN
"$RESEARCH_ROOT/.venv-reference/bin/python" -B "$STUDY/launch.py" --result "$NEW_RESULT" --method D16
"$RESEARCH_ROOT/.venv-reference/bin/python" -B "$STUDY/launch.py" --result "$NEW_RESULT" --method D4
python3 -B "$STUDY/analyze.py" --result "$NEW_RESULT"
```

준비 단계가 결과 경로 중복을 거절하고 첫 GPU 실행 전에 설정·입력·코드와 보호 원본 목록을 동결합니다. 완료 worker나 실패 attempt를 덮어쓰지 않습니다. 전체 worker wall 예산은 3,600초이며 모델 로딩·artifact 읽기·CPU decoder·직렬화를 포함합니다. GPU에서 한 모델만 실행하며 backbone 학습·solver 재실행·다운로드·패키지 업그레이드는 하지 않습니다.

## 검토 ZIP의 CPU 재계산

ZIP을 새 폴더에 풀고 해당 폴더에서 다음 명령을 실행합니다. 공개 준비용 `records/`에는 경로를 치환한 원시 이벤트와 실험 소유 synthetic fixture가 있습니다. 외부 모델·기존 case 전체 파일은 포함하지 않습니다.

```bash
python3 -B code/analyze.py --result records --inputs records/inputs --output /tmp/tool-repair-audit-new --skip-external-preservation
```

출력 디렉터리는 새 경로여야 합니다. 이 명령은 도구 호출/인자·최종 fixture 상태·정답 조건·pairing·분모를 재검산합니다. `--skip-external-preservation`은 ZIP에 없는 기존 원본을 이번 재계산에서 검증하지 않았다는 뜻이며, 원 실행의 원본 보존 receipt와 구분됩니다. logits·weight 수치를 다시 측정하지 않습니다.

## 지원 범위

이름 제한은 ASCII 등록명을 사용합니다. 이름과 최상위 JSON key의 escape 표기는 지원하지 않으며 거절합니다. 문자열 인자와 일반 최종 답은 제한하지 않습니다. 실제 토크나이저로 vocab 문자열을 확인하고, tool marker와 name 필드가 한 토큰으로 합쳐진 미지원 구성은 명시적으로 거절합니다. 배치 크기는 1입니다. 이름 제한 CPU 비용이 포함된 생성 시간은 serving TTFT/TPOT 또는 packed INT4 속도가 아닙니다.

원형 모델·INT4 reference 및 fixture harness는 기존 연구의 구현을 재사용합니다. 최종 보고서는 새 결과의 `REPORT_KO.md`와 `REPORT.md`, 실행·원본 보존·CPU 검산은 `gpu_commands.json`, `verification.json`, `cpu_verification.json`에 기록합니다. GitHub·PR·main·Pages 작업은 이번 실행에 포함하지 않습니다.
