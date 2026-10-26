# Case020 P4 — CPU에서 raw → 표 → 그림 재현

아래 경로는 이 머신에서 확인했다. 모델·GPU·Torch import 없이 재집계한다. 기존 runtime을 읽기 전용으로 사용했으며 새 clean environment 검증은 아니다.

```bash
cd /home/mskim/Codex_main/inference-lab-case020-optic-decode-20261026-1215/cases/020-optic-on-demand-decode/followup/p4_speed_cost
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1   /home/mskim/inference-lab/.venv-baseline/bin/python src/audit.py verify
```

새 외부 출력으로 재생성한다. 출력 경로가 이미 있으면 다른 새 이름을 사용한다.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1   /home/mskim/inference-lab/.venv-baseline/bin/python src/audit.py summarize   --out /tmp/case020-p4-summary-new
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/case020-p4-mpl-new   OPENBLAS_NUM_THREADS=1 /home/mskim/miniforge3/envs/ai_lab/bin/python   src/audit.py figures --out /tmp/case020-p4-figures-new
PYTHONDONTWRITEBYTECODE=1 /home/mskim/inference-lab/.venv-baseline/bin/python   -m unittest discover -s tests -v
```

현재 Case 전체를 보존한 다른 위치에서는 `--portable`을 명시한다. Case 안의 기록된 절대 경로는 현재 Case로 연결하고 Case 밖 원본이 없으면 확인했다고 주장하지 않고 경로를 보고한다. 원 artifact/ID/old raw를 빼고 P4만 복사하면 재현할 수 없다.

감사는 저장된 PASS를 정답으로 쓰지 않고 hash·shape·dtype·ID M/U·16 layer 수치 합·BF16 sample의 FP64 식·순서·K·단위·H/D/E 통계·수명주기 ns 구간·실제 N·전체 GPU 장부를 재계산한다. 단위 오류·event/K 오류·가짜 PASS·manifest 내용/coverage 변조를 거절하는 fixture가 있다.

## 과거 manifest 범위의 원 verifier

기존 verifier는 신규 P4까지 포함한 coverage를 기대하지 않는다. 원 verifier를 수정하지 않고 원747개 파일만 임시 외부 복원본으로 만들어 검사한다. 이 작업은 약81MiB를 일시 사용하고 종료 때 복원본을 제거한다.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1   /home/mskim/inference-lab/.venv-baseline/bin/python src/verify_historical.py   --tmp-parent /tmp/case020-p4-historical-new   --receipt /tmp/case020-p4-historical-receipt-new.json
```

실제 수행한 receipt는 `provenance/original_verifier_restored_scope.json`이다. 원 manifest SHA는 f4aa25642fdaa16a7d2e943da8ae903d057a0ebe52d01c337dcfa68764e801a2로 유지됐다.

## 파일 흐름과 실행 이력

- Original P2: `../../results/20261026-1235/` → `summary/legacy_*.csv`.
- P4 immutable raw: `results/p4-20261026-1451/` + append-only `worker_ledger.jsonl`.
- Design/IDs/arrays: `PROTOCOL.json`, `provenance/plans.json`, `inputs.json`, `artifacts.json`, `freeze.json`, `main_freeze.json`, `preflight_index.json`.
- Independent CPU: `src/audit.py` → `summary/` → `src/figures.py` → four PNG/SVG pairs.
- Local regeneration/test/preservation/cleanup evidence: `provenance/local_validation.json`, `preservation_after.json`, `resource_closeout.json`.
- New file identity: `PUBLIC_MANIFEST_P4.json` excludes itself; the original manifest stays historical.

GPU는 이번 장부에서 이미 완료됐으므로 `prepare.py`나 `run.py`를 재실행해 기존 파일/장부를 초기화하지 않는다. 실행 source와 order는 freeze와 raw에 보존했다. 새로운 GPU 재현은 새 run ID·source/config freeze·잔여 예산·독점 계산·전용 preflight를 다시 설계한 별도 범위가 필요하다. 원격 CI와 다른 머신 검증은 NOT_RUN이다.

최초 P4A 분석의 decision/paired/memory-stage snapshot3개는 `results/p4a_initial/`에 bytes 그대로 보존했다. CPU 외부 재생성 점검에서 이 snapshot과 재생성 대상 summary의 범위 차이를 발견해 정리했으며 수치 불일치나 GPU 재실행은 없었다. `provenance/p4a_snapshot_preservation.json`에 최초 경로·오류·보존 hash가 있다.
