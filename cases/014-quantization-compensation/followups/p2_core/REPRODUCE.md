# Case014 P2 재현 / Reproduction

cwd는 Case014입니다. `REFERENCE_PYTHON`은 기존 Torch/NumPy Python,
`CPU_PYTHON`은 기존 NumPy/Matplotlib Python입니다. 새 설치 명령은 없습니다.
원 실행의 실제 Python 경로는 로컬 private identity receipt에 보존하고 공유본에서는
환경 label로 매핑합니다. 관측 버전은 [runtime manifest](../../results/p2_r1/runtime_manifest.json)에 있습니다.

## A. 모델·GPU 없이 저장 결과 검산

```bash
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
"$REFERENCE_PYTHON" -m unittest discover -s tests -v
"$REFERENCE_PYTHON" -m unittest discover -s followups/p1b_selection/tests -v
"$REFERENCE_PYTHON" -m unittest discover -s followups/p2_core/tests -v
"$CPU_PYTHON" followups/p2_core/source/audit.py \
  --run results/p2_r1 --output /tmp/case014-p2-audit-new.json
```

기존15개와 P1b10개는 회귀 재실행입니다. 새 고유 P2 검사는9개입니다. P0 MC나
기존 P1/P1b GPU sweep은 이 명령에 포함되지 않습니다. 감사 script는 테스트 수에
추가하지 않습니다. 출력 파일·폴더는 존재하지 않는 새 경로를 사용합니다.

원자료만 복사해 표·구간·decision을 재생성합니다.

```bash
python3 followups/p2_core/source/copy_for_reanalysis.py \
  --run results/p2_r1 --output "$NEW_ANALYSIS"
"$CPU_PYTHON" followups/p2_core/source/analyze.py --run "$NEW_ANALYSIS"
"$CPU_PYTHON" followups/p2_core/source/audit.py \
  --run "$NEW_ANALYSIS" --output "$NEW_ANALYSIS/independent_audit.json"
MPLCONFIGDIR=/tmp/case014-p2-mpl "$CPU_PYTHON" followups/p2_core/source/plot.py \
  --run "$NEW_ANALYSIS" --output "$NEW_FIGURES"
```

기존 summary·decision·집계 CSV를 정답으로 읽지 않습니다. 문서별 측정320행,
artifact metadata588건, 자체 경로 구축224그룹, CAL 분해392건, block checkpoint56건을
검사합니다. paired 비교10개의 NLL95% 구간과 PPL 비율로 변환한 구간을
10,000회/seed6101523으로 재계산합니다. 공유본의 CPU 검산은 private weights나
GPU logits 자체의 재계산과 구분합니다.

## B. 외부 모델·private 입력이 필요한 GPU 재실행

같은 기록 입력의 replay 경로입니다. 이미 본 P2 EVAL을 새 fresh 평가로 부르지 않습니다.

필요한 외부 재료:

- 既存WSL Qwen/Qwen3-0.6B snapshot @ `c1899de289a04d12100db370d81485cdf75e47ca`。
- model.safetensors local SHA256:
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。
- P2 private의 `identity.private.json`, `inputs/documents.private.json`.
  CAL32는 원 P1, SMOKE1도 재사용합니다. 평가는 새 PG19 64책과 ArXiv16 content group입니다.
- 既存環境: Python3.12.14/Torch2.13.0+cu130/Transformers5.17.0/NumPy2.3.5、
  RTX5080 SM120/driver610.88. 관측 환경이며 새 설치용 requirements를 추정한 목록이 아닙니다.

새 private/output 경로에 동결한 입력 token 파일과 input manifest를 복사합니다.
identity에는 원 snapshot/Python 참조와 새 private root를 기록합니다. 기존 P1/P1b의 K나
INT4 payload를 복사하지 않습니다. P2는 새 출력에 자체 K/artifact를 구축합니다.

실제 사용한 CLI:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
python3 followups/p2_core/source/launch.py \
  --case . --private "$NEW_PRIVATE" --output "$NEW_RESULTS"
```

launch는 identity의 기존 GPU Python으로 `run_p2.py`, 종료 뒤 `reload_worker.py`를
순차 실행합니다. GPU에는 모델1개만 상주시킵니다. native forward·CPU FP64 full-input
Cholesky·packing·I/O·실패·fresh process reload를10800초 이내에서 회계합니다.
runtime spec은 `followups/p2_core/specs/p2.json`이며 CAL-only guard가 EVAL fitting을 거절합니다.

등록한 보조 SMOKE 행렬 검사는 별도 프로세스에서 같은 예산에 가산합니다.

```bash
"$REFERENCE_PYTHON" followups/p2_core/source/smoke_matrix_probe.py \
  --case . --private "$NEW_PRIVATE" --output "$NEW_RESULTS"
```

새 출력에는 실행 전 고정 module·입력·허용오차를 등록합니다.
이번 [보조 등록](../../results/p2_r1/supplemental_smoke_registration.json)은 fresh forward0의
시점에 작성했습니다. 보조 검증도 남은 worker 예산 안에서 수행합니다.

데이터 준비에 실제 사용한 CLI입니다. 기존 r1에 재실행하면 덮어쓰기 guard가 거절합니다.
local cache만 읽었고 새 전송은0 bytes입니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
"$DATA_PYTHON" followups/p2_core/source/prepare_inputs.py \
  --case . --private "$P2_PRIVATE" \
  --pg19 "$PG19_TRAIN_SHARD_00000" --pile "$PILE10K_ARROW"
```

신규 입력은 별도 등록과 새 출력 설정이 필요합니다. 이 명령을 고정 r1에 대한
새 selector 탐색이나 문서 보충에 사용하지 않습니다. cached source의 hash/revision/split/row/
offset은 [input manifest](../../results/p2_r1/input_manifest.json) 에 있습니다. hash만으로
원문이나 token을 복원할 수는 없습니다.

## C. 저장·실행의 경계

method별196 projection은 header36bytes·group64·FP32 scale·low-nibble-first INT4
파일입니다. 전체588파일을 독립 decoder로 대조하고, 원 checkpoint의 공통 부분과 함께
새 프로세스 full-model SMOKE를 검사했습니다. runtime에 K/residual GEMM을 추가하지 않습니다.
projection core는 embedding/head/norm을 포함한 standalone checkpoint가 아닙니다.

측정은 packed 저장본을 BF16으로 복원한 dense 품질 reference입니다. 파일 bytes와
GPU memory, offline 구축시간과 serving latency를 구분합니다. TTFT/TPOT/packed kernel
speedup·최대 context는 이번 측정 대상이 아닙니다.

## D. 검토 ZIP과 원형 보호

검토 ZIP에는 코드·작은 fixture·모든 scalar 원행·spec·manifest·보고서를 담습니다.
model/K/INT4 payload/전체 X/Z/logits/원문/private 절대경로는 제외합니다.
새 폴더에 풀어 CRC·경로·중복·symlink·모든 file hash, CPU 회귀 검사와
원행→표·decision 재계산을 수행합니다. ZIP bytes/SHA256과 검증 receipt는
ZIP 옆의 외부 review 디렉터리에 저장합니다.

P0/P1/P1b 원형·WSL snapshot·기존 환경은 보호 대상입니다. 과거 not-run 판정을
소급 변경하지 않습니다. P3·4B·agent·selector 탐색·Git stage/commit/push/PR/merge·
Pages/Drive는 실행하지 않았습니다.
