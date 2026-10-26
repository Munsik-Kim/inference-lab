# Case020 재현

## 모델·GPU 없이 원자료 재검산

NumPy 2.2.6 또는2.3.5가 있는 Python을 사용한다. 그림에는 Matplotlib 3.10.8도 필요하다. verify/summarize/figures와 --help는 Torch를 import하거나 CUDA context를 만들지 않는다. 실제 그림 runtime은 Python 3.10.19이며 공식 operator 예제는 Python≥3.12가 필요하다.

```bash
# From this case directory. These commands need no model, Torch, or CUDA.
python src/audit_v2.py verify
python src/audit_v2.py summarize --out /tmp/optic-summary
MPLCONFIGDIR=/tmp/optic-mpl python src/audit_v2.py figures --out /tmp/optic-figures
python tests/test_audit_v2.py -v
```

경로: `results/20261026-1235/*.json`+`inputs/`+보존 BF16 raw → `src/audit_v2.py` → `results/summary/{correctness,memory,latency,paired}.csv`·`decision.json` → `src/figures.py` → `figures/01_memory,02_latency,03_unique_cost.(png,svg)`. 저장된 PASS/summary를 정답으로 쓰지 않고 raw timing·storage 식·수치 gate를 재계산한다. 다른 block 표본은 합치지 않는다.

B block0의10개 NPZ는 dense table이 아닌 요청 출력의 BF16 uint16 words다. `(words.astype(uint32)<<16).view(float32)`로 해석한다. P1 raw에는 모든 method/seed/control의257개 출력이 있다. NPZ는 allow_pickle=False로 읽는다.

## 공식 CPU 예제와 P1

테스트한 operator runtime 또는 NumPy/Torch가 있는 Python≥3.12에서 실행한다. generation/자동 다운로드 entry point를 실행하지 않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=vendor/optic/src python vendor/optic/examples/decode_optic.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B src/cpu_checks.py --out /tmp/optic-p1
```

P1은 deterministic inputs/p1_*.npz와 fixture_manifest.json을 다시 쓰므로 전달 후보의 bytes를 보존하려면 별도 사본에서 실행한다. vendored 최소 source/라이선스/hash를 사용하며 editable install·vLLM·Transformers가 필요하지 않다.

## GPU를 새로 측정할 때

현재 run은 완료됐으며 덮어쓰거나 다른 반복으로 덧붙이지 않는다. 새 worktree/output 사본에 새 run ID/장부를 지정하고 경로를 명시적으로 바꿔 freeze한 뒤 prepare→freeze→probe→preflight→smoke→sweep 순서로 수행한다. 제공된 orchestration은 원 run ID와1800초 장부에 고정되어 있다. 원자료를 reset하는 단축 명령을 제공하지 않는다. 개별 worker만 띄우면 상위 장부를 우회하므로 권장 계측 경로가 아니다.

GPU compute가 비어 있어야 하며 Python≥3.12·sm_120 지원 Torch·실제 FP16 mm(out_dtype=FP32) probe 통과가 필요하다. 추가 allocated2GiB·누적 worker1800초를 유지한다. dtype/CPU fallback으로 공식 경로 PASS를 만들지 않는다. serving 환경 설치/업그레이드나 모델·dataset 다운로드는 필요하지 않다.

로컬 관측 operator executable은 `/home/mskim/inference-lab/.venv-baseline/bin/python`, 그림 executable은 `/home/mskim/miniforge3/envs/ai_lab/bin/python`이다. 둘 다 읽기 전용 재사용이며 clean install/다른 머신/원격 CI 검증을 뜻하지 않는다.

## 파일 확인

PUBLIC_MANIFEST.json은 자기 자신을 제외한 Case 파일의 상대경로·bytes·SHA256을 담고 Git base/upstream/미커밋 benchmark bundle을 구분한다. provenance/reproduction_receipt.json에 같은 환경에서 summary/그림 재생성 및 시각 QA를 남겼다. wheel/cache/model/dense table 사본은 공개 산출물에 넣지 않았다.
