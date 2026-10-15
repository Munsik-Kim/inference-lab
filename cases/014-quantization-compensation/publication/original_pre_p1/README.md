# Case014 — Quantization Compensation: Cancellation, Re-quantization, and Generalization

[한국어](README.ko.md)

A diagnostic tool for checking what survives after quantization compensation is saved again. It separates inherited error, existing cancellation and re-quantization on common inputs, with a controlled path toward held-out document quality evaluation.

Implemented: correction fit → INT4 save → reload → error decomposition → common-input full-model runner. The final GPU path has not yet been execution-validated.

**Completed:** exact rational examples, INT4 save/fresh-process reload, independent NumPy/Torch FP64 checks and 2,000-seed dither controls per cell. Fifteen unique CPU tests pass.

**P1 not run:** the six-target/five-strength runner and document splits are prepared. The required D-original location is unconfirmed. There is no measured LLM improvement, regression or transfer result yet.

For fixed errors with ε=0.01, absolute error changes from 0.0001 to 0.0101 after compensation: removing one error disrupts existing cancellation. With the other sign, error improves from 0.0201 to 0.0101.

The independent four-code example gives teacher 3.016, BASE 3 and COMP 4: error 0.016→0.984. Final-layer inputs differ (3 versus2). This is a two-bit scalar illustration, not a frequency estimate for INT4 language models.

[CPU 검산 / CPU checks](REPRODUCE.md) · [결과 보기 / Results](demo/en/index.html) · [코드 / Code](scripts/run_p1.py) · [상세 보고서 / Report](REPORT.md)
