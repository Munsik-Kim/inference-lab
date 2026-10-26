# Case020 — 모델·GPU 없이 저장 기록 검산하기

[English](REPRODUCE.md) · [보고서](REPORT.ko.md)

Linux/WSL에서 Python3.12.14의 새 venv로 실행했다. 필요한 것은 NumPy2.3.5와 Matplotlib3.10.8 및 고정한 작은 종속 package뿐이다. Torch·CUDA·모델 다운로드는 필요 없다. 기존 GPU 환경을 업그레이드하지 않았다. native Windows 검증은 하지 않았다.

```bash
git clone https://github.com/Munsik-Kim/inference-lab.git
cd inference-lab
python3.12 -m venv /tmp/case020-cpu-env
/tmp/case020-cpu-env/bin/python -I -B -m pip install \
  -r cases/020-optic-on-demand-decode/publication/requirements-cpu.txt
/tmp/case020-cpu-env/bin/python -I -B -m pip check
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/case020-cpu-env/bin/python -I -B \
  cases/020-optic-on-demand-decode/publication/run_checks.py \
  --mode quick --tests --output /tmp/case020-quick-new
```

venv와 output 경로가 이미 있으면 새 이름을 쓴다. output은 저장소 밖의 존재하지 않는 폴더여야 한다. script는 일반 경로에서 직접 실행하며 editable 설치나 개인 절대경로가 필요 없다.

## 빠른 검사와 전체 기록 검산

quick은 공개 파일·원형 mapping·실행 source/input/raw bytes/hash, 모든 원 P2/P4 timing의 mean/median/p95, storage·실제 session/장부·문서 facts·작은 독립 FP64 참조와 unit 검사를 수행한다. 큰 GPU 출력 전체의 독립 참조 검산은 full에서 한다. 파일 전체 hash를 읽었다는 사실을 전체 수치 검산으로 부르지 않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/case020-mpl-new \
  /tmp/case020-cpu-env/bin/python -I -B \
  cases/020-optic-on-demand-decode/publication/run_checks.py \
  --mode full --tests --output /tmp/case020-full-new
```

full은 quick에 더해 NPZ CRC/member/magic, 수정 없는 원 P2/P4 감사, 원 P1의 전체 token/layer/seed/control과 보존된 P2 BF16 출력·P4 preflight 수치, 별도 NumPy 통계 경로의 대응, 그림4종 PNG/SVG 재생성을 실행한다. 과거 original manifest의 coverage는 원747파일만 임시 복사해 검사하고 복사본은 제거한다. 원 verifier나 허용 오차를 바꾸지 않는다.

P4 portable mode가 원 머신의 외부 경로를 찾지 못하면 그 사실을 보고한다. 그 외부 원본의 공개 사본19개는 preservation_map으로 별도 bytes/hash 확인한다. 기록된 절대경로는 원형으로 보존하고, wrapper가 현재 checkout의 보관본을 선택한다. GPU/모델을 새로 실행한 검증이 아니다.

## 입력과 출력

- Original Case: `publication/originals/pre-publication/` (1020 exact files including P4).
- External analysis/handoffs: `publication/originals/external/` (19 exact files).
- Path/bytes/hash mapping: `publication/preservation_map.json`.
- New raw-derived tables: `summary/`; each run regenerates them under `OUTPUT/summary/`.
- Full audit outputs: `OUTPUT/historical_P2_summary/`, `OUTPUT/historical_P4_summary/`, `OUTPUT/figures/`.
- Success/failure: `OUTPUT/verification.json` or `failure.json`, `tests.log` / `failure.log`.

모든 자료가 일반 Git 파일에 있으므로 Release asset이나 자동 evidence 다운로드는 없다. 공개 파일 하나25MiB 이하, Case 전체128MiB 이하를 검사한다. full은 임시 원본 복사 때문에 약81MiB의 추가 공간을 잠시 사용한다. 그 외 결과도 새 외부 경로에 쓴다.

실제 수행한 새 환경 검증은 publication/receipts/local_cpu.json에 있다. 원격 CI는 Case020 전용 CPU quick 범위이며 과거 GPU 실험을 다시 측정하지 않는다. 공개 후 checkout의 자체 검증은 최종 인계에서 PR/main SHA와 구분해 보고한다. 과거 receipt의 NOT_RUN은 수정하지 않았다.

기존 prepare/worker/run은 과거 실행 source를 보존한 것이므로 이번 검산에 실행하지 않는다. 새 GPU 실험은 별도 승인·run ID·freeze·예산·preflight가 필요한 다른 작업이다.
