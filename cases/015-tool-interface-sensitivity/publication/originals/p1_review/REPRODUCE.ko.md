# CPU 검산과 실제 모델 재실행

[English](REPRODUCE.md). 압축을 푼 Case015 폴더에서 실행합니다. CPU 경로는 Python 3.10 이상 표준 라이브러리만 사용하며 설치가 필요 없습니다. 출력은 모두 새 경로로 지정합니다.

```bash
export PYTHONPATH="$PWD/src"
python3 -B -m interface_sensitivity.cli validate
python3 -B -m unittest discover -s tests -v
python3 -B -m interface_sensitivity.cli oracle-check --output /tmp/case015-oracle-new.json
python3 -B -m interface_sensitivity.cli analyze --records results/p1_r1/episodes.jsonl --output /tmp/case015-analysis-new.json
python3 -B -m interface_sensitivity.cli audit --records results/p1_r1/episodes.jsonl --output /tmp/case015-audit-new.json
python3 -B -m interface_sensitivity.cli render --records results/p1_r1/episodes.jsonl --output /tmp/case015-explorer-new
```

기존 결과 → 입력·mapping 검사 → 실제 fixture oracle 실행 → raw 응답·도구 사건 재적용 → task별 paired 분석 → 한영 탐색기 순서입니다. `validate`는 모델을 읽지 않습니다. `oracle-check`는 32개 CPU episode의 48단계 상태·관측을 확인합니다. `audit`는 저장 응답에서 호출을 다시 파싱하고 새 fixture에서 실행해 두 점수를 다시 계산합니다. 원자료에 없는 logits나 모델 추론을 재현하지 않습니다.

같은 인터페이스 안의 모델 비교는 다음처럼 export합니다. 다른 인터페이스의 입력은 서로 다르므로 동일 입력으로 위장하지 않습니다.

```bash
python3 -B scripts/export_diova.py --records results/p1_r1/episodes.jsonl --output /tmp/case015-diova-new
```

생성된 `I0_ORIGINAL/BF16.jsonl`과 `I0_ORIGINAL/D16.jsonl` 등을 기존 `diova-compare`의 validate·compare로 읽을 수 있습니다. 이 패키지는 ZIP에 포함하지 않으며 저장소의 기존 패키지 경로를 사용합니다. 인터페이스 사이의 변화는 이 case의 `analyze`가 canonical task와 동결 mapping으로 비교합니다.

GPU 재실행에는 기존 Qwen3-4B revision, RTN·D16 각 252개 payload, 동일한 환경이 별도로 필요합니다. [영문 안내](REPRODUCE.md)의 실제 `run --private` 진입점과 원 실행 receipt를 참고합니다. 고정한 private resolver hash는 입력 경로의 무단 변경을 막으므로 다른 PC에서는 명시적인 경로 재등록과 amendment가 필요합니다. ZIP만으로 가중치를 복원하거나 solver를 자동 실행하지 않습니다.

새 calibration·양자화·설치·학습·D4·P2/P3·원격 게시는 이번 실행 범위 밖입니다. 모델·환경·기존 결과는 읽기 전용으로 보존합니다.
