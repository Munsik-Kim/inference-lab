# Case017 — Agent Contract–Cost Bench

We tested whether an agent both changes a synthetic record correctly and returns a verified final answer. Tasks cover integer order totals, atomic warehouse transfers, and document metadata updates. BF16 model weights stay fixed while four output-contract policies and two native KV-cache precisions vary.

| KV | C0 FREE | G GRAMMAR | N NEUTRAL | E ERROR |
| --- | --- | --- | --- | --- |
| K0 | 16/24 | 16/24 | 16/24 | 16/24 |
| K1 | 7/24 | 15/24 | 9/24 | 13/24 |

Denominator: 24 new synthetic HOLDOUT tasks. On NVFP4, grammar recovered 8 free-generation failures with 0 losses. All BF16 policies completed 16/24. Grammar did not eliminate wrong semantic values or stale-validation completion.

A calculates and stores discounted totals from item counts and prices. B reads two stock records and transfers units atomically. C reads a document revision and updates status, tags, and owner. All data are owned synthetic JSON records.

Implemented here: strict JSON contracts, four-policy runner, safe fixtures and observation tracking, independent endpoint audit, comparison tools, and bilingual explorer. Engine and NVFP4 kernels come from vLLM/FlashInfer upstream.

[Report](REPORT.ko.md) · [English report](REPORT.md) · [한국어 탐색기](demo/ko.html) · [English explorer](demo/en.html) · [CPU/GPU reproduction](REPRODUCE.md)

```bash
PYTHONPATH=src python3 -B -m agent_contract_cost.verify --output /tmp/case017-review-new-output
```

Use a new, nonexistent output path. This command replays saved responses without calling a model.

### DEV / P1

| KV | C0 FREE | G GRAMMAR | N NEUTRAL | E ERROR |
| --- | --- | --- | --- | --- |
| K0 | 5/8 | 5/8 | 5/8 | 5/8 |
| K1 | 2/8 | 5/8 | 2/8 | 4/8 |

P0 22 requests; P1 233; P2 775. Total 1030 API generation attempts, 50585 output tokens; worker wall 1035.148s.

K0 is BF16 KV; K1 is native NVFP4 KV. **Weights are BF16 in both.** G constrains generated tokens with a public JSON schema; N/E allow one retry at the first contract rejection. The3.556× token-slot ratio in an approximately4GiB pool is neither total VRAM reduction nor task speedup. P3, weight INT4, training, concurrent serving, remote publication, and Pages are not run.

[Protocol](PROTOCOL.md) · [Reuse](REUSE_MAP.md) · [Notice](NOTICE.md) · [Roadmap](ROADMAP.md) · [Source freeze](provenance/execution_freeze_p2.json)
