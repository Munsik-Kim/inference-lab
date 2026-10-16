# 내 결과로 시작하는 diova-compare

[English](README.md) · [패키지 안내](../../README.ko.md)

두 모델의 **저장된 답변**을 같은 질문·정답 규칙으로 비교합니다. 모델 실행과 채점은 먼저 끝내고, 이 CLI로 결과를 검사·집계합니다. 아래 네 질문과 답변은 손으로 정한 공개 합성 자료(`synthetic_test`)이며 GPU 실측이 아닙니다.

## 1. 합성 예제 실행

저장소 루트에서 Python 3.10 이상으로 실행합니다. 전용 venv와 출력에 매번 새 경로를 만듭니다. 아래는 Linux/macOS 명령이며 Windows에서도 별도 Python 환경의 같은 CLI 인자를 사용합니다.

```bash
EXAMPLE_ROOT=$(mktemp -d)
python3 -m venv "$EXAMPLE_ROOT/venv"
"$EXAMPLE_ROOT/venv/bin/python" -m pip install --no-deps ./packages/diova-compare
"$EXAMPLE_ROOT/venv/bin/diova-compare" validate --input packages/diova-compare/examples/compare/baseline.jsonl
"$EXAMPLE_ROOT/venv/bin/diova-compare" validate --input packages/diova-compare/examples/compare/candidate.jsonl
"$EXAMPLE_ROOT/venv/bin/diova-compare" compare --baseline packages/diova-compare/examples/compare/baseline.jsonl --candidate packages/diova-compare/examples/compare/candidate.jsonl --output "$EXAMPLE_ROOT/report"
```

각 validate는 `{"status": "PASS", "records": 4}`, compare는 `status: PASS`와 `n_pairs: 4`를 반환하며 정상 종료 코드가 0입니다. 새 디렉터리에 `report.md`, `report.json`, `pairs.csv`가 만들어집니다. 설치 시 패키지 빌드 도구를 가져올 수 있지만 모델·Torch·CUDA·NumPy는 필요하지 않습니다. 데이터는 로컬 CLI에서 읽습니다.

[기준선 파일](baseline.jsonl) · [후보 파일](candidate.jsonl)

## 2. 보고서에서 알 수 있는 것

정답은 기준선의 답을 베끼지 않고 **두 정수의 합**으로 정했습니다. `answer`는 추출·공백 제거를 끝낸 십진 정수 문자열입니다. 아래 문장 자체의 UTF-8 SHA256을 `prompt_hash`로 사용했습니다. 끝에 줄바꿈은 없습니다.

| ID | 정확한 입력 문장 | gold | 기준선 → 후보 | 의미 |
|---|---|---|---|---|
| sum-01 | What is 2 + 2? Reply with an integer. | 4 | 4 → 5 | 정답 손실 |
| sum-02 | What is 3 + 5? Reply with an integer. | 8 | 7 → 8 | 새 정답 |
| sum-03 | What is 6 + 1? Reply with an integer. | 7 | 6 → 9 | 다른 오답 |
| sum-04 | What is 4 + 5? Reply with an integer. | 9 | 9 → 9 | 둘 다 정답 |

`report.md`는 과제 요약, `report.json`은 설정·집계와 모든 입력 쌍, `pairs.csv`는 입력별 확인에 사용합니다. `tasks["integer-addition@1"].counts`에서 정답은 양쪽 **2/4**지만 `correct_to_wrong=1`, `wrong_to_correct=1`, `wrong_to_wrong=1`, `all_answer_disagreement=3`입니다. 평균 정답률만 보면 사라지는 손실과 회복을 구분할 수 있습니다. 합성 네 쌍은 모델 우열이나 실제 업무 성능의 근거가 아닙니다.

## 3. 자신의 기록을 넣기

두 실행의 결과를 각각 JSONL로 바꾸되, 양쪽에 같은 질문과 같은 정답 정의를 사용합니다. 위 두 파일을 형식의 출발점으로 삼으세요.

| 내 기록 | JSONL 필드 | 맞춰야 할 계약 |
|---|---|---|
| 평가셋·질문 ID | `task`, `task_version`, `sample_id` | 양쪽에서 같은 입력 쌍을 식별 |
| 실제 모델에 준 입력 | `prompt_hash` | 동일한 입력 표현의 UTF-8 SHA256; 템플릿·시스템 지시도 포함하는 규칙을 양쪽에 적용 |
| 독립적으로 정한 허용 답·채점 규칙 | `gold`, `gold_definition` | 허용 문자열 목록과 추출·정규화 규칙을 동일하게 유지 |
| 모델·실행 가중치 식별 | `model_id`, `artifact_id` | 각 파일 안에서 하나의 모델/저장본 조합; 서로 다른 저장본끼리 비교 가능 |
| 추출 답·그 규칙의 판정 | `output_type`, `answer`, `correct` | 자유 생성은 `extracted_answer`; boolean 정오를 사용자가 제공 |
| 자료 종류·형식 | `evidence_kind`, `schema_version` | 실제 측정은 `measurement`, 과거 기록은 `historical`; 현재 버전은 정수 1 |
| 선택: 원 실행 종료 코드 | `process_exit_code` | 양쪽 모든 행에 정수로 제공하거나 모두 생략; 비정상 종료도 유지 |

입력 hash 계산은 `hashlib.sha256(prompt.encode("utf-8")).hexdigest()`입니다. CLI는 hash 문자열이 같은지 검사하며 원문을 다시 읽거나 실제 입력이었다는 사실을 인증하지 않습니다. `artifact_id` 역시 사용자가 제공한 식별자이며 artifact 파일 내용의 진위를 검증하지 않습니다. 자유 생성의 `correct`는 **입력받는 판정**입니다. 모든 답을 자동으로 채점하지 않습니다.

답변 변경·정답 전환만 보려면 `distribution`과 `scores`를 생략해도 됩니다. 선택지 확률을 제공할 때만 해당 NLL·Brier를, 호환되는 분포를 제공할 때만 KL을 계산합니다. 전체 어휘 확률이나 logits는 기본 비교에 필요하지 않습니다. [정확한 schema와 지표 계약](../../README.ko.md#입력과-지표).

자기 파일에도 validate를 각각 실행한 뒤 같은 compare 인자에 경로를 넣으세요. 공개할 수 있는 입력만 예제·보고서에 포함하세요. 브라우저 업로드 단계는 없습니다.

## 4. 잘못된 입력 쌍은 거절하기

다음은 후보 파일의 첫 입력 hash만 바꾼 **의도적인 불일치**입니다. 같은 `EXAMPLE_ROOT`에서 실행합니다.

```bash
"$EXAMPLE_ROOT/venv/bin/python" - "$EXAMPLE_ROOT/mismatch.jsonl" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(line) for line in Path("packages/diova-compare/examples/compare/candidate.jsonl").read_text().splitlines()]
rows[0]["prompt_hash"] = "0" * 64
Path(sys.argv[1]).write_text("".join(json.dumps(row) + "\n" for row in rows))
PY
"$EXAMPLE_ROOT/venv/bin/diova-compare" compare --baseline packages/diova-compare/examples/compare/baseline.jsonl --candidate "$EXAMPLE_ROOT/mismatch.jsonl" --output "$EXAMPLE_ROOT/rejected-report"
```

예상하는 오류는 `diova-compare: pair mismatch: prompt_hash`이며 종료 코드는 **2**입니다. 보고서 디렉터리도 만들지 않습니다. 정답 정의만 다르면 `pair mismatch: gold_definition`으로 거절합니다. 질문이나 채점 기준이 달라진 두 값을 모델 변경의 효과로 집계하지 않기 위한 검사입니다. 단독 validate가 통과해도 **두 파일의 pairing**은 실패할 수 있습니다. `--intersection`은 입력 계약 불일치를 해결하는 옵션이 아닙니다.

## 기술적 판단 — 현재 구현의 해석

- **선택:** 같은 입력 쌍에서 손실·회복·불일치를 나누고 원 실행 종료 코드를 보존합니다.
- **대안:** 양쪽 평균 정답률만 볼 수 있지만, 이 예제처럼 같은 평균 아래의 회귀가 사라집니다.
- **근거:** [집계 코드](../../src/diova_compare/core.py)와 [정답 전환·실패 보존 테스트](../../tests/test_compare.py)가 이 계약을 구현합니다.
- **한계·다음 판단:** free-form 정오와 입력·artifact 식별자는 호출자가 제공합니다. 새 과제에서는 채점 규칙의 적합성을 먼저 확인해야 합니다. 이는 현재 설계의 장단점 설명이며 당시 대안 실험을 추가로 주장하지 않습니다.

## 다른 경로: 과거 실험 재분석

자신의 JSONL을 비교하는 위 경로와 [Case013 export](../../README.ko.md#case013-export)는 별개입니다. 과거 export는 공개 기록의 고정된 채점·도구 실행 계약을 적용하고 원 실행의 실패를 보존합니다. 합성 예제 실행은 외부 사용자 테스트나 GPU 재현이 아닙니다.
