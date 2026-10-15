# Case014 P1b 재현 / Reproduction

검토 ZIP만으로 가능한 것은 저장 기록의 CPU 검산입니다. 실제 GPU 재실행에는
기존 WSL 모델, P1 K/prefix/target 파일, 새 P1b private token 입력이 필요합니다.
해시에서 원문·K·weight를 복원할 수 없습니다. 모든 명령은 Case014 디렉터리가 cwd입니다.

## A. GPU·모델 없이 저장 기록 검산

사용한 환경은 기존 CPU NumPy 환경과 기존 Torch 환경입니다. 설치 명령은 포함하지
않습니다. `CPU_PYTHON`은 NumPy가 있는 Python, `REFERENCE_PYTHON`은 기존 Torch/NumPy
Python을 가리킵니다. 전체 관측 버전은 [runtime manifest](../../results/p1b_r2/runtime_manifest.json)에 있습니다.

```bash
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
"$REFERENCE_PYTHON" -m unittest discover -s tests -v
"$REFERENCE_PYTHON" -m unittest discover -s followups/p1b_selection/tests -v
"$CPU_PYTHON" followups/p1b_selection/source/audit_saved.py \
  --run results/p1b_r2 --output /tmp/case014-p1b-audit-new.json
```

첫 검사는 기존 고유15개 재실행, 둘째는 새 고유10개입니다. 감사 script는 별도의
검산 실행이며 테스트 수에 더하지 않습니다. output은 존재하지 않는 새 파일이어야 합니다.
P0 Monte Carlo를 실행하는 명령은 없습니다.

원행만 복사해 표와 10,000회 bootstrap 구간을 다시 만듭니다. `NEW_ANALYSIS`는 새 경로입니다.

```bash
python3 followups/p1b_selection/source/copy_for_reanalysis.py \
  --run results/p1b_r2 --output "$NEW_ANALYSIS"
"$CPU_PYTHON" followups/p1b_selection/source/analyze_saved.py \
  --case . --run "$NEW_ANALYSIS"
"$CPU_PYTHON" followups/p1b_selection/source/audit_saved.py \
  --run "$NEW_ANALYSIS" --output "$NEW_ANALYSIS/independent_audit.json"
```

복사하는 것은 raw cells와 동결 metadata뿐이며 `summary.json`·정책표는 복사하지
않습니다. `old_CAL_reference.json`은 C 유형 진단에 필요한 기존 CAL 스칼라 합계입니다.
selector는 SELECT만 받으며 이 참조나 old EVAL을 읽지 않습니다.

그림 재생성에는 기존 Matplotlib 환경이 추가로 필요합니다. 새 출력 폴더를 사용합니다.

```bash
MPLCONFIGDIR=/tmp/case014-p1b-mpl "$CPU_PYTHON" \
  followups/p1b_selection/source/plot_saved.py \
  --run results/p1b_r2 --output "$NEW_FIGURES"
```

전체 raw3,360행,90 target/alpha/split 파일,216요약,12선택,4,704 policy aliases,
462개의 저장 구간을 검산합니다. 기록된 NLL/KL/argmax의 GPU 계산 자체는 이 경로에서
다시 수행하지 않습니다. ZIP에 없는 모델·배열 hash는 external artifact identity입니다.

## B. 원래 private 자산으로 실제 GPU 재실행

이번에 실제 실행한 명령의 치환 가능한 형태입니다. 원 명령·절대경로는 외부 private
receipt에, 경로 label을 적용한 공개 receipt는 [여기](../../results/p1b_r2/worker_receipt.json)에 있습니다.
**아래 GPU 명령을 CPU 감사 과정에서 자동 실행하지 않습니다.**

필수 자산:

- Qwen/Qwen3-0.6B @ `c1899de289a04d12100db370d81485cdf75e47ca`, 기존 WSL snapshot.
- 기존 P1 private root의 `PRIVATE_MANIFEST.json`, `snapshot.private.json`, `artifacts/`:
  K6개,140 prefix INT4,30 target INT4. 누락 시 자동 fit/encode/download하지 않습니다.
- 모델 전체 local weight SHA256:
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- P1b private `inputs/documents.private.json`113개(새112+재사용 SMOKE1).
  토큰 파일 SHA256: `3e6874e9e7c31eaabc6b5c0e9689781454efaa2854daa174414e4b807b3d1454`.
- 기존 환경: Python3.12.14, Torch2.13.0+cu130, Transformers5.17.0, NumPy2.3.5;
  RTX5080 SM120/driver610.88. 이 목록은 관측 환경이며 새 설치용 requirements가 아닙니다.

`NEW_PRIVATE`와 `NEW_RESULTS`는 반드시 새 디렉터리여야 합니다. private 입력은 기존
P1b 것을 복사·읽기 전용 연결하고, 그에 대응하는 input manifest를 새 결과에 보존합니다.
기존 P1b 결과에 재실행하지 않습니다. 모델 경로는 기존 snapshot을 그대로 사용합니다.

```bash
python3 followups/p1b_selection/source/validate_assets.py \
  --case . --old-private "$P1_PRIVATE" --private "$NEW_PRIVATE" \
  --output "$NEW_RESULTS"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
python3 followups/p1b_selection/source/launch_readonly.py \
  --case . --private "$NEW_PRIVATE" --output "$NEW_RESULTS"
```

`validate_assets.py`가 기존 P1 snapshot receipt에서 실제 모델·Python 경로를 읽습니다.
`launch_readonly.py`는 그 Python을 사용합니다. 런타임 spec 위치는 고정된
`followups/p1b_selection/specs/p1b.json`입니다. `--previous-receipt`는 실패 시간이 있는
동일 승인 회차의 누적 예산 회계용이며, 이번 r2 실제 실행에서는7.042초를 이월했습니다.
누적5400초를 넘겨 계속하지 않습니다.

새 문서의 원본 입력 준비를 재현하려면 기존 cache와 tokenizer 및 기존 pyarrow 환경이
필요합니다. 아래는 실제 사용한 CLI이며 원문을 내려받지 않습니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
"$DATA_PYTHON" followups/p1b_selection/source/prepare_inputs.py \
  --model "$MODEL_SNAPSHOT" --pg19 "$PG19_TRAIN_SHARD_00000" \
  --pile "$PILE10K_ARROW" --old-inputs "$P0_PRIVATE_INPUTS" \
  --private "$NEW_PRIVATE" --output "$NEW_RESULTS"
```

이는 고정된 local cache 첫 train shard pool의 재현입니다. source 파일 bytes/hash,
revision/split/row/offset와 라이선스는 [input manifest](../../results/p1b_r2/input_manifest.json)에
있습니다. 공개 hash만으로 원문을 복원하지 못합니다. 이미 있는 private 입력을 재사용할
때 이 명령을 중복 실행하지 않습니다.

## 실제 수행·실패·비용

첫 worker는 과거 SMOKE prefix 배열이 저장됐다고 가정한 비교에서 exit1/3 forward로
중단했습니다. 모든 새 SELECT/EVAL 출력 전입니다. source와 실패 기록을 보존하고,
해당 비교의 가용성만 고친 r2에서 기존 teacher 정확 일치·prefix hash·현재 반복·공통 Z를
검사했습니다. spec/입력/K/quantizer를 바꾸지 않았습니다.

r2는 exit0,90/90 파일,3,360/3,360 후보 행,3,813 전체 모델 forward를 완료했습니다.
전체 attempt는3,816 forward/519.991초 worker wall입니다. 단일 조건 성능 예측이나
GPU busy 시간이 아닙니다. 실제 Torch peak allocated1,613,842,432 B/reserved1,845,493,760 B,
장치 사용은 forward 뒤 샘플입니다. 관측 시 다른 GPU 작업은 종료하지 않았습니다.

최종 ZIP은 새 폴더에 풀어 CRC·경로·중복·symlink·파일 SHA와 위 CPU 두 검사 및 raw→table
재계산을 수행합니다. 최종 ZIP 크기/SHA·검증 receipt는 ZIP과 같은 외부 검토 디렉터리에
있습니다. 모델·K·payload·전체 X/Z/logits·원문은 제외합니다. P2/P3·원격 게시는 미실행입니다.
