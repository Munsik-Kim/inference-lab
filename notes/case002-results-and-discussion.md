# Qwen3-4B on RTX 5080: BF16 and FP8 document extraction

[Full case and reproduction commands](../cases/002-bf16-fp8-document-extraction/README.md) · [Download experiment ZIP](../downloads/case002_bf16_fp8_document_extraction.zip)

I compared the official BF16 and FP8 releases of Qwen3-4B-Instruct-2507 on Korean document extraction. FP8 used less GPU memory and returned answers sooner, but accuracy was low with both configurations.

The task was to extract four fields for a specified request ID: owner, task name, confirmed due date, and amount in KRW. The documents included other requests, missing values, and confirmed or withdrawn changes. I used 20 development documents and 100 evaluation documents, split equally between short and long inputs. These were synthetic records. Gold answers were calculated from structured facts and confirmed changes before inference; BF16 outputs were not used as answers for scoring FP8.

A document counted as correct only when all four fields matched the gold answer.

| Evaluation result | BF16 | FP8 |
|---|---:|---:|
| All documents | 31/100 | 32/100 |
| Short inputs | 18/50 | 17/50 |
| Long inputs | 13/50 | 15/50 |
| Valid JSON and schema | 100/100 | 100/100 |

Both models got 30 documents right. BF16 alone got one right, FP8 alone got two right, and both got 67 wrong. The FP8 minus BF16 difference was +1 percentage point, with a paired bootstrap 95% interval of -2 to +5 points. That is not enough to claim that FP8 improved accuracy or preserved quality.

I ran three paired rounds, alternating which model ran first. The first round supplies the quality results; the later rounds measure timing variation and output repeatability on the same documents. Each model returned exactly the same text across all three rounds.

The following are medians across the three warmed rounds, measured by a sequential localhost streaming client. Short inputs were 498–559 tokens and long inputs were 4,009–4,069 tokens, including the instructions and chat template.

| Measure | BF16 | FP8 |
|---|---:|---:|
| Short time to first text | 51.5 ms | 26.2 ms |
| Short request completion | 658.4 ms | 399.0 ms |
| Short output tokens | 54 | 49 |
| Long time to first text | 355.5 ms | 173.4 ms |
| Long request completion | 974.4 ms | 562.3 ms |
| Long output tokens | 52 | 48 |
| Whole-GPU sampled peak, range across rounds | 11.623–11.624 GiB | 8.211 GiB |

Both models had the same 1 GiB BF16 KV cache budget. GPU memory was sampled about every 100 ms and includes Windows and background allocations. The difference between median run peaks was 3.412 GiB. Output lengths differed, so the request-time reduction is not a measurement of decode speed alone. FP8 was also slower on one document where an incorrect task field made the response longer.

The runs used an RTX 5080 16GB, WSL2 Ubuntu 24.04.4, driver 610.47, vLLM 0.29.0, PyTorch 2.13.0+cu130, and Triton 3.7.1. Both used one GPU, one request at a time, temperature 0, and the same tokenizer, template, and runtime settings. This was the V1 engine with CUDA graphs, not eager mode. The checkpoints were pinned to fixed revisions; this compares the official releases, rather than isolating weight precision as the only difference.

For this task, FP8 offers useful memory and latency savings. Neither setup is accurate enough to leave extraction unchecked. These results cover a small synthetic dataset under the recorded settings, not Korean business documents in general.

## Reply on extraction quality

The errors give me a reason to look beyond quantization. Both models failed on the same 67 documents, and every first-round response was valid JSON. The main problem was choosing the right content, not producing the required format.

Some responses copied values from another request. Others kept an earlier date despite a confirmed change, or returned the literal “not recorded” marker instead of null. When the task name was missing, neither model returned the correct null value on any of those ten documents. These observations do not prove that quantization has no effect, but they point to problems shared by both setups.

The first follow-up I would run is a development-set comparison between the full document and only the records belonging to the target request. That would help separate distraction from errors in applying the update rules. After that, I would test extracting the events first and applying confirmed changes, date conversion, and amount conversion in code. Those are proposed checks, not improvements measured in this case.

The document format here is regular enough that a rules-only extractor also belongs in the comparison. If a parser handles this format reliably, there is little reason to make the model calculate the final state itself. That would be a finding about this format, not a solution for arbitrary documents.

I would keep these results unchanged and use a fresh evaluation set after fixing the next procedure. The existing errors have already informed the follow-up ideas, so the current 100 documents should not be presented as an untouched test of those changes.

---

OpenAI Codex assisted with synthetic data preparation, code, analysis, and writing. This work does not claim human annotation review or independent reproduction.

Evidence: [methods](../cases/002-bf16-fp8-document-extraction/METHODS.md), [fixed experiment spec](../cases/002-bf16-fp8-document-extraction/configs/experiment_spec.json), [aggregate results](../cases/002-bf16-fp8-document-extraction/results/aggregate.json), [error examples](../cases/002-bf16-fp8-document-extraction/results/error_examples.md), and [full failure list](../cases/002-bf16-fp8-document-extraction/results/failures.jsonl). [Model revisions](../cases/002-bf16-fp8-document-extraction/provenance/models.json) and [environment details](../cases/002-bf16-fp8-document-extraction/provenance/INSTALL.md) are recorded alongside the execution evidence.
