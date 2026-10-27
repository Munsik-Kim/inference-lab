# CPU 재현 안내

[English](REPRODUCE.md) · [보고서](REPORT.ko.md)

공개 파일의 검산은 저장 집계와 작은 합성 산술을 확인합니다. 실제 가중치나 큰 logits를 다시 계산하는 GPU 재현과 구분합니다. Python3.11.15 / NumPy1.26.4 / Matplotlib3.10.9의 Linux/WSL CPU 환경을 사용합니다. native Windows 검증은 하지 않았습니다. 모든 output은 존재하지 않는 새 경로여야 합니다.

```bash
git clone https://github.com/Munsik-Kim/inference-lab.git
cd inference-lab
python3.11 -m venv /tmp/case021-cpu
/tmp/case021-cpu/bin/python -m pip install -r cases/021-rounding-execution-diagnostics/publication/requirements-cpu.txt
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 /tmp/case021-cpu/bin/python -I -B cases/021-rounding-execution-diagnostics/publication/run_checks.py --mode quick --tests --output /tmp/case021-quick-new
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 /tmp/case021-cpu/bin/python -I -B cases/021-rounding-execution-diagnostics/publication/run_checks.py --mode full --output /tmp/case021-full-new
```

새 venv 설치는 공개 소비자를 위한 CPU 환경 설치입니다. 원 연구 환경을 업그레이드하지 않습니다. 이번 출판에서는 호환 기존 CPU 환경을 읽기 전용으로 사용했고, 실제 명령·결과는 [검증 기록](publication/VALIDATION.md)에 남깁니다.

| Scope | Input | Entry point | GPU | This publication |
|---|---|---|---|---|
| A: public aggregates → statistics/bootstrap | Git checkout CSV/JSON | `run_checks.py --mode quick` | No | Local + clean public copy |
| A: tables + figures | Same public files | `run_checks.py --mode full` | No | Local + clean public copy |
| B: exact synthetic BF16/ULP/energy | Small public fixture and tests | `--tests`; historical public CPU child | No | Local |
| C: complete KL from private logits | Original private BF16 logits, manifests | Historical explicit-path audit source; see below | No | Prior receipts retained; new complete private audit NOT_RUN |
| D: GPU artifact replay | Private D16 artifacts, local checkpoint/runtime | Historical worker CLI | Yes | NOT_RUN |

quick은 공개 manifest·원형 대응·경로·집계 분모·단위·문서 facts, 원 주 bootstrap와 P4 교차 bootstrap, 합성 BF16 nearest-even·ULP·에너지 식을 확인합니다. full은 같은 검산 뒤 P5의 세 그림과 새 대표 두 그림을 외부 output에 다시 만듭니다. 픽셀 hash 일치는 같은 환경에서 별도로 대조할 수 있으며 환경 간 보장이나 과학적 합격선이 아닙니다.

[Public audit source](publication/run_checks.py) · [Derivation](publication/derive.py) · [Historical P5 public checks](history/p5/source/public_cpu.py) · [Original private-logit audit receipt](evidence/p0p3/audit_eval_v1.json) · [P4 full-logit audit receipt](evidence/p4/p4a_verification.json) · [Private replay boundary](HISTORY.md)

큰 raw와 실제 code/scale/Z·단일항 출력은 배포하지 않습니다. 따라서 공개 summary의 KL 값을 재집계하는 것은 원 logits에서 KL을 다시 계산한 검사가 아닙니다. 원 실행 source는 읽을 수 있지만 private 경로는 명시적 CLI로 공급해야 하며 모델/환경 자동 다운로드는 없습니다. 과거 GPU 소스를 CPU 검사 과정에서 import하지 않습니다. 새 receipt는 외부 output에만 씁니다. 원 protocol과 허용 오차는 변경하지 않습니다.
