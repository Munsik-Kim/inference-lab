# 모델 없이 API 회귀표 재계산하기

Python3.10+ 표준 라이브러리만 사용합니다. GPU·모델·새 설치 없이 저장된 응답을 실행하고 원형 파일·점수·표를 확인합니다. 저장소 root에서 실행하며 출력은 새 외부 디렉터리입니다.

```bash
OUT=$(mktemp -d)
python3 -B cases/015-tool-interface-sensitivity/publication/scripts/run_checks.py --output "$OUT/case015" --tests
```

검산기는 P1의228개 원형 파일만 새 view에 복원해 옛 closed-set 검산기를 그대로 실행합니다. A1은96개 응답을 새 주문 fixture에 재실행하고 별도 endpoint와 회귀표를 재계산합니다. 27개 P1 회귀와18개 A1 고유 검사를 구분합니다.

## A1: API definitions → equivalence → raw replay → regression table

```bash
A1=cases/015-tool-interface-sensitivity/followups/order-api-regression-v1
OUT=$(mktemp -d)
python3 -B -m json.tool "$A1/data/schema.json"
PYTHONPATH="$A1/src" python3 -B -m order_regression.cli validate
PYTHONPATH="$A1/src" python3 -B -m order_regression.cli oracle-check --output "$OUT/oracle.json"
PYTHONPATH="$A1/src" python3 -B -m order_regression.cli audit --output "$OUT/audit.json"
PYTHONPATH="$A1/src" python3 -B -m order_regression.cli analyze --output "$OUT/regression.json"
PYTHONPATH="$A1/src" python3 -B -m order_regression.cli render --output "$OUT/explorer"
```

`oracle.json`의34개는17개 fixture×2API의 합성 계약 검사입니다. 실측96개와 합치지 않습니다. `explorer/ko.html`, `en.html`은 file://에서도 열립니다. JSON·모델 출력은 HTML text로 표시합니다.

## GPU 실험의 별도 재현 경계

이번 게시 과정은 위 GPU 시험을 반복하지 않습니다. 새 empty private run root와 원 snapshot·RTN/D16 각252개 파일·원 r2 메타데이터·호환 GPU 환경이 필요합니다. private/runtime.json은 실제 경로가 포함된 로컬 기록에서 복원해야 합니다. 공개 hash만으로 파일이나 입력을 복원할 수 없습니다.

```bash
# GPU_PYTHON and A1_PRIVATE refer to existing validated local inputs.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$A1/src" "$GPU_PYTHON" -B -m order_regression.cli run --private "$A1_PRIVATE"
python3 -B "$A1/postprocess/collect.py" --private "$A1_PRIVATE"
```

같은 완료 run 디렉터리는 거절합니다. source/spec/input/model/artifact identity를 먼저 대조하고 새 결과를 옛 결과 위에 쓰지 않습니다. 모델 loader는 누락 artifact를 생성하지 않습니다. 기존 P1 전체 run_all·solver·encode·ENUM·feedback은 guard로 막습니다.

[Report](REPORT.ko.md) · [Stage map](STAGE_MAP.md) · [Original P1 commands](publication/originals/p1_review/REPRODUCE.ko.md) · [Model identity](provenance/model_identity.json)
