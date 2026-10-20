# Native NVFP4 KV technical follow-up

[Case overview](../../README.md) · [Report](../../REPORT.md) · [CPU / GPU reproduction](../../REPRODUCE.md)

[Frozen protocol](specs/protocol.json) · [Execution source hash](provenance/execution_freeze.json) · [Kernel-path annotation](provenance/kernel_path_annotation.json) · [Raw/CSV](results/analysis/)

A separate technical follow-up designed after the agent P0/P1 results. Weights stay BF16 while native NVFP4 KV storage changes. Completed36 fixed blocks/288 timed requests,plus54 separate P0/warmup/sanity requests. F4 prefill is FA2; decode is XQA receiving packed data/scales. Original agent verdicts are unchanged.
