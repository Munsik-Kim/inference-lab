# 구현과 기술 / Implementation and technology

실제 입력 경로를 따라 보정하고 같은 INT4 저장 형식에 효과를 남기는 도구입니다. PG19의 전층 품질 이득과 ArXiv 전이 한계를 함께 제공합니다.
The pipeline captures actual paths, fits compensation and stores it in the same INT4 representation, with whole-model quality and domain-limit evidence.

| Project implementation | Upstream / prior work | Source |
| --- | --- | --- |
| Native capture and topological own-path orchestration | Qwen3 model, Transformers, PyTorch hooks/SDPA | [runner](followups/p2_core/source/run_p2.py) |
| Full-input FP64 ridge and shared-factor scheduling | PyTorch Cholesky/triangular solve; Guo compensation motivation | [fit](followups/p2_core/source/p2_fit.py) |
| Versioned signed INT4 storage, independent reload | Standard RTN, NumPy arrays, Python struct | [codec](scripts/codec.py), [independent decoder](followups/p2_core/source/independent_codec.py) |
| Three-term decomposition, paired scoring and audit | Linear algebra, NLL/PPL/KL, document bootstrap | [decompose](scripts/decompose.py), [auditor](followups/p2_core/source/audit.py) |
| Raw-derived bilingual result display and closed stage views | Python stdlib, GitHub Actions/Pages | [exporter](publication/scripts/derive.py), [restorer](publication/scripts/restore_stage.py) |

[Results](REPORT.md) · [한국어](REPORT.ko.md) · [CPU workflow](REPRODUCE.md) · [Original sources](NOTICE.md)

Codex assisted implementation/execution/analysis/writing. Separate CPU calculation paths are verification tools, not independent human peer review or official author approval.
