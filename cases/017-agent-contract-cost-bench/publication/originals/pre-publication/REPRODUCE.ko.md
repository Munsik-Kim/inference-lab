# Case017 재현

모델 없이 하는 확인은 저장된 원응답·SSE를 다시 파싱하고, 새 fixture에서 실제
도구 사건·상태·관측을 재구성한 뒤 별도 endpoint 계산으로 표를 다시 만드는
작업입니다. 모델 추론이나 새 재토큰화가 아닙니다. Python3.10 이상 표준
라이브러리로 실행하며, 가중치·환경·KV tensor는 필요하지 않습니다.

이 case 디렉터리에서 다음 명령을 실행합니다. 출력은 기존 자료 밖 새 경로입니다.

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
CASE017_CHECK_ROOT=$(mktemp -d)
python3 -B -m agent_contract_cost.verify --output "$CASE017_CHECK_ROOT/replay"
python3 -B -m unittest discover -s tests -v
python3 -B -m unittest discover -s postprocess/tests -v
python3 -B postprocess/verify_recorded_inputs.py --stage P2 --output "$CASE017_CHECK_ROOT/inputs"
python3 -B postprocess/inspect_contract_boundaries.py --stage P2 --output "$CASE017_CHECK_ROOT/annotations"
```

`replay/verification.json`에 범위가 기록됩니다. `P1-audit`/`P2-audit`는 새 fixture의
재생 행이며 `P1-analysis`/`P2-analysis`는 바로 그 새 행에서 다시 계산한 표입니다.
저장된 success나 summary를 정답으로 삼지 않습니다. 누락을 0점이나 성공으로
채우지 않습니다. ZIP에는 source hash 확인에 필요한 Case016 import 소스 5개도
포함합니다.

짧은 CPU 사용 순서는 다음과 같습니다.

```bash
python3 -B -m agent_contract_cost validate --output "$CASE017_CHECK_ROOT/validate"
python3 -B -m agent_contract_cost oracle-check --output "$CASE017_CHECK_ROOT/oracle"
python3 -B -m agent_contract_cost audit --stage P1 --output "$CASE017_CHECK_ROOT/p1-audit"
python3 -B -m agent_contract_cost.reporting --output "$CASE017_CHECK_ROOT/report"
python3 -B -m agent_contract_cost.review --output "$CASE017_CHECK_ROOT/explorer"
```

`oracle-check`는 사람이 정한 합성 정답 행동열의 계약 검사이며 모델 성공률이
아닙니다. `reporting`은 감사된 실측에서 한영 Markdown·SVG를 만듭니다.
`review`는 한영 HTML·JSON·receipt를 만들며, 같은 generation 번호라도 다른
조건의 관측 이력이 같다고 표시하지 않습니다. 실제 확인한 [로컬 탐색기](demo/ko.html)는
`file://`로 열 수 있고 JavaScript가 꺼져도 핵심 표를 읽을 수 있습니다.
표시 사본은 메시지를 hash로 중복 제거하지만 복원한 값·UTF8 bytes가 원문과
같은지 검사합니다. 전체 SSE·입력 IDs는 연결된 원 cell에 그대로 있습니다.

신규 CPU 단위 검사와 기존 Case015/016 재실행 수는 분리합니다.
`tests/tokenizer_probe.py`는 선택적 CPU 통합 검사입니다. 실제 로컬 tokenizer와
xgrammar가 필요하며, 기록된58개 token-mask 경계를58개 단위 검사나 새 추론으로
합산하지 않습니다. 표준 CLI/help/재생 경로는 Torch·Transformers를 import하지 않습니다.

GPU 재현에는 기존 snapshot·vLLM 환경·비공개 runtime JSON이 필요합니다.
[영문 상세 안내](REPRODUCE.md#gpu-reproduction-boundary)의 실제 명령과 필드를
확인합니다. 완료된 결과 경로는 덮어쓰기를 거절합니다. 다른 PC·경로의 GPU 반복은
새 격리 작업공간과 별도 등록이 필요하며 원 freeze를 고쳐 통과시키면 안 됩니다.
base Git SHA, 실행 source SHA256, 모델 revision은 서로 다른 identity입니다.
이번 범위에서는 Git stage/commit/push/PR/merge·Pages를 수행하지 않습니다.

새 검토 ZIP 생성·검사는 다음 명령입니다.

```bash
python3 -B -m agent_contract_cost.package --output "$CASE017_CHECK_ROOT/package"
```

CRC·안전 경로·중복·symlink·bytes/hash 및 추출본의 CPU 재계산을 실제 실행합니다.
ZIP SHA256·추출 후 receipt는 자기참조를 피하기 위해 ZIP 옆에 둡니다.
원 token·SSE를 모두 유지하면서 크기를 줄이기 위해 ZIP_LZMA 무손실 압축을
사용합니다. Python zipfile 또는7-Zip으로 풀 수 있으며 기본 ZIP 뷰어는 이 방식을
지원하지 않을 수 있습니다.

원 scorer의 보조 라벨 `terminal_budget_or_infrastructure`는 의미적 실패 종료에도
붙는 포괄 라벨입니다. 이를 예산 소진 근거로 쓰지 않고 실제 `terminal_reason`과
카운터를 확인합니다. [별도 검산](results/terminal_independence_r1/receipt.json)은
저장된 종료 자격을 믿지 않고256개 endpoint를 계산했으며 점수 차이는0개입니다.
원 라벨·점수는 유지했습니다. 다음 명령으로 재확인할 수 있습니다.

```bash
python3 -B postprocess/check_terminal_independence.py --output "$CASE017_CHECK_ROOT/terminal"
```
