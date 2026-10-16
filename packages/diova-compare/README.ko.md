# diova-compare

[English](README.md) · [프로젝트](../../README.ko.md)

제가 만든 **저장된 AI 답변 비교 도구**입니다. 모델을 바꾸거나 압축하기 전후의 답변 파일 두 개를 넣으면, 이전에 맞혔다가 틀린 문제와 새롭게 맞힌 문제를 보고서로 볼 수 있습니다. 평균 정답률이 같아도 놓칠 수 있는 변화를 확인하는 데 씁니다.

사용자가 모델 실행과 답변 채점을 먼저 마치고, 같은 질문·정답 규칙으로 정리한 파일을 준비합니다. 이 도구는 CPU에서 파일을 검사하고 결과를 집계하는 명령줄 프로그램(CLI)입니다. 모델 실행, 자유 서술 답변의 범용 자동 채점, 입력·모델 파일의 진위 인증은 제공하지 않습니다. Python 3.10 이상이며 모델 다운로드나 Torch·CUDA가 필요하지 않습니다.

<a id="compare-your-results"></a>
## 내 결과로 시작하기

[합성 네 쌍 예제 실행](examples/compare/README.ko.md): 저장된 답변 준비 → 같은 질문·채점 규칙으로 필드 채우기 → 형식 검사(validate) → 비교(compare) → report.md / report.json / pairs.csv 읽기. 최소 입력, 자유 생성 정오를 제공하는 방식, 입력 불일치 거절 명령까지 따라갈 수 있습니다.

[과거 실험 재분석](#case013-export)은 별도 사용 경로입니다.

## 설치와 실행

이 디렉터리에서 별도 환경을 사용합니다.

```bash
python -m pip install .
diova-compare validate --input /path/to/B.jsonl
diova-compare compare --baseline /path/to/B.jsonl --candidate /path/to/Q.jsonl --output /path/to/new-report
```

출력 디렉터리는 새 경로여야 합니다. 먼저 `report.md`에서 과제별 요약을 읽고, `pairs.csv`에서 질문별 변화를 확인하세요. `report.json`에는 설정·집계·모든 비교 행이 담깁니다. `PASS`는 입력 검사와 비교가 완료됐다는 뜻이며, 모델의 모든 답이 맞았다는 뜻은 아닙니다. 설치한 CLI는 저장소 밖에서도 실행합니다. `python -m pip wheel --no-deps . --wheel-dir /path/to/wheels`로 로컬 wheel을 만들 수 있으며 업로드는 별도입니다.

## 입력과 지표

JSONL은 한 줄에 한 질문의 결과를 넣는 텍스트 형식입니다. 각 행에 다음 값을 담습니다. 실제 필드명은 그대로 사용하세요.

| 준비할 내용 | 필드와 의미 |
|---|---|
| 형식 버전과 질문 식별 | `schema_version=1`, `sample_id`, `task`, `task_version`: 어떤 평가의 몇 번째 질문인지 |
| 같은 입력인지 확인할 문자열 | `prompt_hash`: 시스템 지시·질문 틀까지 같은 규칙으로 담은 입력의 SHA256 해시 |
| 독립적으로 정한 정답과 채점 규칙 | `gold`: 허용 정답 문자열 목록; `gold_definition`: 답 추출·공백 처리 등 정답 판정 규칙 |
| 비교할 모델과 저장 파일 식별 | `model_id`, `artifact_id`: 모델 이름과 사용한 모델 파일의 식별자 |
| 답과 판정 | `output_type`: 선택형 `choice` 또는 자유 서술에서 뽑은 답 `extracted_answer`; `answer`: 답; `correct`: 맞음/틀림인 boolean 값 |
| 자료의 출처 종류 | `evidence_kind`: 실제 측정 `measurement`, 과거 기록 `historical`, 인공 예제 `synthetic_test` |

필수 값이 빠지거나 질문 ID가 중복되면 거절합니다. 두 파일에서 질문·정답 규칙이 다른 경우에도 비교하지 않습니다. 공통 ID만 비교하려면 `--intersection`을 명시하며 제외한 ID도 결과에 남습니다. 식별자는 사용자가 제공하므로, 일치 검사가 실제 입력이나 모델 파일의 진위를 보증하지는 않습니다. 자유 서술형의 `correct`도 사용자가 정한 채점 결과를 받습니다.

답변 변경만 비교할 때 확률 자료는 필요하지 않습니다. 추가 진단을 원하면 선택지 또는 전체 단어 후보의 확률 분포(`distribution`)에 순서가 있는 후보 이름(`labels`), 각 확률(`probabilities`), 문장 분할기 이름(`tokenizer_id`), 입력 해시(`prefix_hash`), 숫자 형식(`dtype`), 확률 합 규칙(`normalization`)을 제공합니다. 유한하고 음수가 아닌 값의 합이 1인지 1e-8 범위에서 검사합니다. 이는 확률을 파일로 표현할 때의 허용 오차이며, 같은 확률의 순위를 바꾸기 위해 더하는 작은 수(epsilon)가 아닙니다. 전체 단어 후보 분포는 `complete=true`와 실제 후보 수인 `vocabulary_size`가 필요합니다. 빠진 단어 후보의 원점수(logits)를 복구하거나 제공된 자료의 진위를 보증할 수는 없습니다.

정답에 높은 확률을 줬는지 보는 손실(NLL)과 전체 선택지의 확률 예측 오차(Brier)는 정답이 하나인 선택형 분포에서 계산합니다. 둘 다 낮을수록 좋습니다. NLL 차이는 후보−기준선이며 음수가 더 좋습니다. 원래 모델과 확률이 얼마나 달라졌는지 보는 값(KL)은 기준선→후보 방향입니다. 낮을수록 원래 분포와 가깝지만 정답률이 높다는 뜻은 아닙니다. 후보의 확률 0으로 KL이 무한대가 되면 null과 별도 infinite 표시를 남깁니다. 자유 생성의 추출 답변에는 선택형 Brier를 붙이지 않습니다. 원래 평가의 답 추출·표기 정리 규칙이 정오(correctness)를 결정합니다.

`correctness_flip`은 맞음→틀림과 틀림→맞음의 합입니다. `all_answer_disagreement`는 모든 답변 변경이고, `wrong_to_wrong`은 다른 오답으로 바뀐 경우입니다. `both_wrong`에는 같은 오답도 포함됩니다. [정의와 선행연구](../../docs/related-work/README.ko.md)를 함께 볼 수 있습니다.

기본 통계 구간은 같은 과제(task)의 질문 쌍을 2,000회 다시 뽑는 방식(bootstrap)으로 계산합니다. 서로 다른 평가 과제를 하나의 점수로 평균하지 않습니다. 변화가 한 번도 없었거나 구간 폭이 0이어도, 다른 입력에서도 두 모델이 동등하다는 증명은 아닙니다. MMLU의 과목·그룹 집계와 WikiText의 단어·바이트 기준은 원래 평가 프로그램(harness)의 결과를 사용합니다.

## 구현

기존 실험의 [지표 계산식](https://github.com/Munsik-Kim/inference-lab/blob/9f1e7cbca19334a168561e05be8884bb16815d69/tools/modelpack/numerics.py)를 새 코드로 구현했습니다. 원래 소스는 그대로 두고, 형식 버전을 둔 입력, 다양한 선택지 수, 자유 서술에서 뽑은 답, 같은 질문끼리의 짝 맞추기(pairing) 오류, 독립 실행 CLI를 추가했습니다. 보존된 GPU 실험 도구를 불러와 실행하지 않습니다.

[코어](src/diova_compare/core.py) · [CLI](src/diova_compare/cli.py) · [평가 로그 변환 코드(harness adapter)](src/diova_compare/adapters.py) · [실패 조건 테스트](tests/test_compare.py).

기존 평가 로그를 이 입력 형식으로 바꾸는 연결 코드(adapter)는 로그의 `acc`(정답률)와 `exact-match`(추출 답의 정확한 일치)를 지원합니다. `acc_norm`(길이를 보정한 선택 점수)과 `perplexity`(다음 글 예측 점수)를 일반 정답률로 바꾸지 않습니다. 과거 기록에는 historical 표시를 유지합니다. [Apache-2.0](LICENSE)이며 Codex가 구현과 검사를 지원했습니다.

선택 필드 `process_exit_code`는 양쪽 모든 기록에 정수로 제공하거나 모두 생략합니다. 입력별 결과와 과제 요약에 그대로 남기므로, 저장된 수치의 검산이 비정상 종료를 정상 실행으로 바꾸지는 않습니다.

<a id="case013-export"></a>

## Case 013: 과거 결과를 변환하고 검사해 보고서 읽기

0.1.2의 `export-case013`은 동결된 [Case 013](../../cases/013-dualquant-reference-quality/README.ko.md)의 긴 문서 답변·도구 사용 기록을 비교용 파일로 변환합니다(export). Torch·NumPy·모델은 필요하지 않습니다. 저장소 루트에서 설치한 CLI로 실행합니다. 아래 세 출력 경로는 모두 새 경로여야 합니다.

여기서 D16과 D4는 같은 INT4 가중치 파일을 쓰는 조건 이름입니다. D16은 계산 입력값을 BF16으로 쓰고, D4는 입력값도 INT4로 줄였다가 BF16으로 복원합니다. 두 조건 모두 품질 확인을 위해 BF16으로 복원해 계산하며, 모델 크기가 다른 두 모델이나 저비트 실행 가속을 뜻하지 않습니다. 아래 6/8은 후속 이름 제한을 적용하기 전의 원래 기록입니다.

```bash
diova-compare export-case013 --case cases/013-dualquant-reference-quality --output /tmp/case013-export
diova-compare validate --input /tmp/case013-export/long/8192/RTN.jsonl
diova-compare compare --baseline /tmp/case013-export/long/8192/RTN.jsonl --candidate /tmp/case013-export/long/8192/D16.jsonl --output /tmp/case013-long-report
diova-compare compare --baseline /tmp/case013-export/agent/base/FREE/RTN.jsonl --candidate /tmp/case013-export/agent/base/FREE/D16.jsonl --output /tmp/case013-agent-report
```

각 비교는 `report.json`, `pairs.csv`, `report.md`를 만듭니다. 장문 8K의 RTN→D16은 둘 다 정답9/기준선만2/후보만1/둘 다 오답0이고, 기본 FREE는8개 중 RTN8개·D16 6개 정답입니다. 최종 `valid:true`가 같아도 실제 파일 수정과 성공한 검사(validation)가 없으면 오답입니다. 원문·실행 조건·마지막 설정(config)은 짝을 이룬 부가 기록 `.sidecar.jsonl`에 남습니다.

`long/16384`, BF16만의 `long/32768`, `agent/base/{REPLAY,FREE}`, `agent/extension16k/{REPLAY,FREE}`를 별도 비교합니다. 16K 확장은 기존8개 중4개 task의 길이 변경본입니다. 후보32K는 메모리 정책상 미실행이므로 비교 행·0점으로 만들지 않습니다. 실행 방식(mode)·길이·질문·정답 규칙이 서로 다르면 기본 질문 짝 검사(pairing)에서 거절합니다. `--intersection`으로 누락을 숨기지 마세요.

`adapter_manifest.json`은 사용한 공개 파일·변환 코드의 해시와 제외 사유를 담은 목록입니다. `continuous_summary.json`은 원래 확률 손실(NLL), 다음 글 예측 점수(PPL), 확률 분포 차이(KL)의 집계를 보존합니다. 새로운 모델 원점수(logits)나 GPU 답변을 만들지 않으며, 품질이 일정 범위 이상 나빠지지 않았다는 통계 판정(비열등성)도 하지 않습니다. 재사용 흐름은 **모델 압축 전후의 업무 답변 기록 → 같은 질문·채점 규칙 검사 → 새롭게 틀린 답·맞힌 답·도구 사용 규칙 실패 보고서**입니다. 고객 데이터나 외부 도구 연결은 이 adapter의 범위 밖입니다.

## 0.1.1 검토 수정판

[CPU wheel 다운로드](../../downloads/diova_compare-0.1.1-py3-none-any.whl) 후 별도 Python 환경에 설치합니다:

```bash
python -m pip install --no-index --no-deps /path/to/diova_compare-0.1.1-py3-none-any.whl
```

PyPI에는 업로드하지 않았습니다. 같은 과제·버전(task/version) 안에서는 모든 질문에 적용하는 NLL/Brier·KL 지표가 일치해야 하며, 다른 과제는 별도 지표 규칙을 사용할 수 있습니다. 아주 작은 양수 확률은 유지합니다. 기준선이 양의 확률을 준 후보에 비교 모델이 정확히 0을 주면, 무한대가 되는 KL을 `null`과 `infinity` 표시로 구분합니다. [수정 기록](CHANGELOG.md).
