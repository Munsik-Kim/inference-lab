# Case 014 — Improving Model Quality at the Same INT4 Storage Budget

[한국어](README.ko.md) · [Pages](https://munsik-kim.github.io/inference-lab/en/case014.html#overview)

Implemented a path-aware compensation pipeline that refits projection weights and stores them in the same INT4 format. On Qwen3-0.6B, fixed quarter-strength compensation reduced perplexity by 12.92% versus RTN on 256-token excerpts from 64 fresh PG19 books, with unchanged projection storage.

**The gain did not transfer to the ArXiv evaluation. Quality was measured after dense BF16 reload; serving speed and VRAM savings were not measured.**

| Measured comparison | RTN PPL | QUARTER PPL | Change |
| --- | ---: | ---: | --- |
| PG19 ·64 book excerpts | 57.600 | 50.156 | -12.92%; NLL64/64 lower |
| ArXiv ·16 content groups | 12.452 | 12.687 | +1.89% point estimate; NLL CI crosses 0 |

**Identical INT4 projection storage:247,733,136 bytes/method.**
Original BF16 PG19 PPL is 45.405.

[Compare results](REPORT.md#results) ·
[Implementation and reproduction](REPRODUCE.md) ·
[P0→P2 research journey](STAGE_MAP.md)

## Implemented pipeline

Capture actual inputs → fit weak compensation → store INT4 → reload in a new process → compare next-token scores on unused documents.

- [Native capture / own-trajectory runner](followups/p2_core/source/run_p2.py)
- [CPU FP64 full-input fitting](followups/p2_core/source/p2_fit.py)
- [INT4 codec](scripts/codec.py) · [Independent decoder](followups/p2_core/source/independent_codec.py)
- [Three-term decomposition](scripts/decompose.py) · [Raw-to-display exporter](publication/scripts/derive.py)
- [CPU tests](followups/p2_core/tests/test_p2.py) · [Public checks](publication/scripts/run_checks.py)

## Scope and sources

One case connects P0 algebra/codec, P1 independent target interventions, P1b independent strength selection and P2 whole-model own-trajectory compensation. Negative results, original files, choices and decisions remain preserved.

This ridge/RTN reference builds on Guo’s prior work; it is not complete official QEP reproduction or a new quantizer. See [PORTFOLIO](PORTFOLIO.md), [NOTICE](NOTICE.md), [source audit](references/source_audit.json).
