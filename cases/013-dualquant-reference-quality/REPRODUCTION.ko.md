# 재현 안내와 지원 범위

[English](REPRODUCTION.md) | 한국어

저장소 root에서 실행하며 모든 결과·receipt는 새 출력 경로를 사용합니다.

## 1. 모델 없이 기록 재계산

Python3.12와 NumPy2.3.5를 사용합니다. [CPU 의존성](publication/requirements-cpu.txt)만 별도 검증 환경에 설치할 수 있습니다.

```bash
VERIFY_ENV=$(mktemp -d)
python3 -m venv "$VERIFY_ENV/venv"
"$VERIFY_ENV/venv/bin/python" -m pip install -r cases/013-dualquant-reference-quality/publication/requirements-cpu.txt
"$VERIFY_ENV/venv/bin/python" -B cases/013-dualquant-reference-quality/analysis/audit.py --output "$VERIFY_ENV/scalars.json"
"$VERIFY_ENV/venv/bin/python" -B -m unittest discover -s cases/013-dualquant-reference-quality/tests -v
"$VERIFY_ENV/venv/bin/python" -B cases/013-dualquant-reference-quality/publication/verify.py --output "$VERIFY_ENV/publication.json"
```

Token별 loss로 NLL/PPL, 예측 ID로 flip, 장문 exact 점수, agent의 최종 답·실제 실행 조건·입력 대응·paired 획득/손실을 다시 계산합니다. Layer별 저장 SSE/energy와 byte 장부도 합산합니다. 이는 metadata 집계의 대조이며 원 weight 오차나 비공개 payload hash의 새 검증이 아닙니다. 고정 위치 KL은 저장 reduction의 집계이고, full logits가 없어 새로운 KL 계산은 아닙니다.

## 2. CPU 합성 INT4 검사

호환되는 기존 Torch 환경이 있으면 다음을 실행합니다.

```bash
CUDA_VISIBLE_DEVICES='' python3 -B -m unittest discover -s cases/013-dualquant-reference-quality/synthetic -v
```

작은 CPU 행렬로 packing·rounding·scale transport·전체 row reduction·실제 bytes·실패 처리를 확인합니다. 학습 모델 품질 검사가 아닙니다. 과거 고유39개 검사 receipt는 [records](records/verification.json)에 보존되며 새 NumPy job의 실행 수와 구분합니다. 원래 source-equation AST 검사는 재배포하지 않는 pinned vendor 파일도 필요합니다.

## 3. 실제 모델 재실행의 별도 요건

이번 공개 정리에서 모델은 실행하지 않았습니다. 원형 [model runner](source/paper_model_runner.py)는 pinned Qwen 로컬 snapshot과 원래 WikiText token arrays, 호환 Torch/Transformers/CUDA가 필요합니다. Weight artifact와 원문 데이터는 이 Case의 다운로드에 없습니다. INT4 variant는 비공개 block-format 라이브러리가 필요하지 않습니다.

허용된 로컬 파일을 준비한 뒤 CLI를 확인하세요.

```bash
python3 -B cases/013-dualquant-reference-quality/publication/prepare_rerun.py --help
python3 -B cases/013-dualquant-reference-quality/source/paper_model_runner.py --help
```

`prepare_rerun.py`의 인자는 `--model06b`, `--model4b`, `--wiki06b`, `--wiki4b`, `--output`입니다. Config/tokenizer/입력 hash를 확인하고 소유한 frozen fixture를 복사해 새 실행 manifest를 만듭니다. 모델 loading·추론은 없습니다. Config가 일치한다는 것만으로 대형 weight 동일성까지 인증되지는 않습니다.

실제 GPU 명령 형태는 `paper_model_runner.py --run NEW_RUN --model 06b --method BF16`입니다. 각 모델에서 BF16을 먼저 실행해 reference arrays를 만들고 RTN/D16을 실행합니다. D4는 저장된 D16 artifact를 재사용합니다. `--model 4b`는 고정 장문/agent DEV도 실행합니다. 이 경로는 외부 환경·모델을 준비한 사람을 위한 설명이며 새 환경에서 성공을 미리 보증하는 예제가 아닙니다. 공개된 완료 결과 디렉터리를 재실행 출력으로 사용하지 않습니다.

Packed probe receipt는 제공하지만 수치 검증을 통과한 serving 구현은 제공하지 않습니다. 공식 W4A4 실행이나 stock GPTQ 비교와 구분합니다.

## 4. 한영 로컬 화면 생성

```bash
PREVIEW=$(mktemp -d)
python3 -B tools/showcase/build.py --output "$PREVIEW/site" --base-path /inference-lab/
python3 -B tools/showcase/check.py --site "$PREVIEW/site" --output "$PREVIEW/site-check.json"
```

생성 폴더의 `en/case013.html` 또는 `ko/case013.html`을 엽니다. 저장 결과와 source 사본을 보여주는 화면이며 모델을 실행하지 않습니다. 실제 배포는 이번 PR 게시와 별도입니다.

## 5. CPU 결과 그림 재생성

Matplotlib와 NumPy가 있는 별도 기존 CPU 환경에서 실행합니다.

```bash
FIGURES=$(mktemp -d)
python3 -B cases/013-dualquant-reference-quality/publication/plot_results.py --output "$FIGURES/plots"
```

[Figure metadata](figures/figure_manifest.json)에 관측된 plot 환경·source를 적었습니다. 추론 명령이 아니며 원형 CSV/JSON을 바꾸지 않습니다. [그림 캡션](figures/README.md)은 분모와 오차 정규화를 설명합니다.
