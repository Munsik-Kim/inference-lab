# diova-compare

기존 실험의 [스칼라 정의](https://github.com/Munsik-Kim/inference-lab/blob/9f1e7cbca19334a168561e05be8884bb16815d69/tools/modelpack/numerics.py)를 새 코드로 구현했습니다. 원래 소스는 그대로 두고, 버전이 있는 입력 형식·가변 선택지 수·추출 답안·명시적인 pairing 오류·독립 CLI를 추가했습니다. 동결된 GPU 도구를 import하지 않습니다.

[English](README.md) · [프로젝트](../../README.ko.md)

두 모델 저장본의 입력별 결과를 비교하는 CPU CLI입니다. 입력과 정답 정의가 같은지 확인한 뒤 정답 손실·새 정답·답변 변경을 집계합니다. 호환되는 확률 분포를 제공하면 NLL·Brier·KL도 계산합니다. Python 3.10 이상이며 모델 다운로드나 Torch·CUDA가 필요하지 않습니다.

## 설치와 실행

이 디렉터리에서 별도 환경을 사용합니다.

```bash
python -m pip install .
diova-compare validate --input /path/to/B.jsonl
diova-compare compare --baseline /path/to/B.jsonl --candidate /path/to/Q.jsonl --output /path/to/new-report
```

출력 디렉터리는 새 경로여야 합니다. `report.json`, `pairs.csv`, `report.md`를 만듭니다. 설치한 CLI는 저장소 밖에서도 실행합니다. `python -m pip wheel --no-deps . --wheel-dir /path/to/wheels`로 로컬 wheel을 만들 수 있으며 업로드는 별도입니다.

## 입력과 지표

JSONL 각 행에는 schema_version=1, sample_id, task/version, SHA256 prompt_hash, 허용 정답 문자열 목록 gold, gold_definition, model_id/artifact_id, output_type, answer, bool correct, evidence_kind를 둡니다. evidence_kind는 measurement/historical/synthetic_test입니다. 누락·중복 ID, 서로 다른 입력·정답 정의는 거절합니다. 공통 ID만 비교하려면 `--intersection`을 명시하며 제외한 ID도 결과에 남습니다.

선택지나 전체 어휘 분포는 ordered labels, probabilities, tokenizer_id, prefix_hash, dtype, normalization을 포함합니다. 유한하고 음수가 아닌 값의 합이 1인지 1e-8 범위에서 검사합니다. 이는 입력 표현 검사이며 동률을 바꾸는 epsilon이 아닙니다. 전체 어휘 분포는 complete=true와 실제 vocabulary_size가 필요합니다. 생략한 전체 logits를 이 도구로 복구하거나 진위를 보증할 수는 없습니다.

NLL과 Brier는 정답이 하나인 선택형 분포에 계산합니다. NLL 차이는 후보−기준선이며 음수가 더 좋습니다. KL 방향은 기준선→후보입니다. 후보의 확률 0으로 KL이 무한대가 되면 null과 별도 infinite 표시를 남깁니다. 자유 생성의 추출 답변에는 선택형 Brier를 붙이지 않습니다. 공식 추출·정규화 기준이 correctness를 결정합니다.

correctness_flip은 정답 손실과 새 정답의 합입니다. all_answer_disagreement는 모든 답변 변경이고, wrong_to_wrong은 다른 오답으로 바뀐 경우입니다. both_wrong에는 같은 오답도 포함됩니다. [정의와 선행연구](../../docs/related-work/README.ko.md)를 함께 볼 수 있습니다.

기본 bootstrap은 task 안의 입력 쌍 2,000회입니다. 서로 다른 benchmark를 하나의 점수로 평균하지 않습니다. 관측 0회와 폭 0인 기술적 구간은 동등성의 증명이 아닙니다. MMLU의 공식 subject/group 집계와 WikiText의 단어·바이트 분모는 harness 결과로 유지합니다.

## 구현

[코어](src/diova_compare/core.py) · [CLI](src/diova_compare/cli.py) · [harness adapter](src/diova_compare/adapters.py) · [실패 조건 테스트](tests/test_compare.py).

adapter는 로그의 acc와 exact-match를 지원합니다. acc_norm과 perplexity를 일반 정답률로 바꾸지 않습니다. 과거 기록에는 historical 표시를 유지합니다. [Apache-2.0](LICENSE)이며 Codex가 구현과 검사를 지원했습니다.

선택 필드 `process_exit_code`는 양쪽 모든 기록에 정수로 제공하거나 모두 생략합니다. 입력별 결과와 과제 요약에 그대로 남기므로, 저장된 수치의 검산이 비정상 종료를 정상 실행으로 바꾸지는 않습니다.

## 0.1.1 검토 수정판

[CPU wheel 다운로드](../../downloads/diova_compare-0.1.1-py3-none-any.whl) 후 별도 Python 환경에 설치합니다:

```bash
python -m pip install --no-index --no-deps /path/to/diova_compare-0.1.1-py3-none-any.whl
```

PyPI에는 업로드하지 않았습니다. 같은 task/version 안에서는 모든 입력의 유효 NLL/Brier·KL 지표가 일치해야 하며 다른 task는 별도 계약을 사용할 수 있습니다. 극소 양수 확률은 유지하고 정확한 0의 누락 질량은 null과 infinity 표시로 구분합니다. [수정 기록](CHANGELOG.md).
