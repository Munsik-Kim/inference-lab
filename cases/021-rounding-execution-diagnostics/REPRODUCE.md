# CPU reproduction

[한국어](REPRODUCE.ko.md) · [Report](REPORT.md)

Public checks reanalyse saved aggregates and small synthetic arithmetic. They do not recompute private logits or GPU inference. The tested CPU environment is Linux/WSL, Python3.11.15, NumPy1.26.4 and Matplotlib3.10.9. Native Windows is untested. Every output must be a new, nonexistent path.

```bash
git clone https://github.com/Munsik-Kim/inference-lab.git
cd inference-lab
python3.11 -m venv /tmp/case021-cpu
/tmp/case021-cpu/bin/python -m pip install -r cases/021-rounding-execution-diagnostics/publication/requirements-cpu.txt
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 /tmp/case021-cpu/bin/python -I -B cases/021-rounding-execution-diagnostics/publication/run_checks.py --mode quick --tests --output /tmp/case021-quick-new
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 /tmp/case021-cpu/bin/python -I -B cases/021-rounding-execution-diagnostics/publication/run_checks.py --mode full --output /tmp/case021-full-new
```

The venv installation is for a publication consumer, not an upgrade to the research environment. This publication used an existing compatible CPU environment read-only. Actual commands/results are in [validation](publication/VALIDATION.md).

| Scope | Input | Entry point | GPU | This publication |
|---|---|---|---|---|
| A: public aggregates → statistics/bootstrap | Git checkout CSV/JSON | `run_checks.py --mode quick` | No | Local + clean public copy |
| A: tables + figures | Same public files | `run_checks.py --mode full` | No | Local + clean public copy |
| B: exact synthetic BF16/ULP/energy | Small public fixture and tests | `--tests`; historical public CPU child | No | Local |
| C: complete KL from private logits | Original private BF16 logits, manifests | Historical explicit-path audit source; see below | No | Prior receipts retained; new complete private audit NOT_RUN |
| D: GPU artifact replay | Private D16 artifacts, local checkpoint/runtime | Historical worker CLI | Yes | NOT_RUN |

Quick checks the closed manifest, preservation mapping, links, denominators/units/document facts, original and P4 bootstrap intervals, and synthetic BF16 nearest-even/ULP/energy identities. Full adds regeneration of three historical P5 figures and two new overview figures in the external output. Pixel hashes can be compared within the same environment; they are not a cross-environment guarantee or a scientific acceptance criterion.

[Public audit source](publication/run_checks.py) · [Derivation](publication/derive.py) · [Historical P5 public checks](history/p5/source/public_cpu.py) · [Original private-logit audit receipt](evidence/p0p3/audit_eval_v1.json) · [P4 full-logit audit receipt](evidence/p4/p4a_verification.json) · [Private replay boundary](HISTORY.md)

Large raw logits, actual codes/scales/Z and weight-revealing single-term outputs are excluded. Aggregating public KL rows is not recomputing KL from logits. Historical source is inspectable but needs explicit private CLI inputs; there is no automatic model/environment download. CPU checks never import GPU runners and write new receipts only to the external output. Original protocols and tolerances remain unchanged.
