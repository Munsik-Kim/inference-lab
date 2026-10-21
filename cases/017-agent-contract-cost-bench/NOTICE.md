# Sources, rights and contribution

Case017 — Agent Contract–Cost Bench combines a shared action contract, owned synthetic business records, isolated callable execution, observation-aware scoring, raw-response capture and CPU replay. New repository code follows the root [Apache-2.0 license](../../LICENSE). This notice does not change the rights of upstream software, model weights or other third-party material.

## Upstream execution

Qwen provides `Qwen/Qwen3-4B-Instruct-2507` and its tokenizer. This case pins revision `cdbee75f17c01a7cc42f958dc650907174af0554` and reuses the existing local snapshot read-only. Its [file identity receipt](provenance/model_identity.json) records local hashes. The snapshot, model weights, tokenizer files and environments are excluded from the review package.

The reused installed stack includes vLLM 0.32.0, FlashInfer 0.7.0.post1, xgrammar 0.2.8, Torch 2.13.0, Transformers 5.19.0 and HTTPX 0.28.1. Their notices and licenses continue to apply. Exact installed source and package metadata hashes are recorded in [environment_identity.json](provenance/environment_identity.json). The prior Case016 [source audit](../016-kv-agent-bench/provenance/source_audit.json) and [NOTICE](../016-kv-agent-bench/NOTICE.md) retain their own verification scope.

vLLM/FlashInfer provide the native NVFP4 cache format, writer and attention kernels. xgrammar/vLLM provide structured decoding. This project does not claim to have invented those algorithms or implemented the upstream kernels. No upstream kernel, driver, model weight or installed runtime source is patched by Case017.

## Repository reuse and new work

The actual read-only Case016 import chain and the adapted client/server code are listed in [REUSE_MAP](REUSE_MAP.md). Case015 contributes design precedent for exact-name dispatch, pre-generation observations and failure replay. Case017 creates a new strict single-object action protocol, three new synthetic workflow families, four policy implementations, explicit attempt accounting, stage registration and independent endpoint/cost analysis.

Case015's selected D16 recovery observations and Case016's cache capacity/serving measurements motivate this experiment. They are not Case017 results or samples. All Case017 weight conditions use BF16 weights; no D16/RTN weight artifacts, training procedure or CIPR code are used.

All purchase, warehouse and document records in this case are generated synthetic fixtures. No real customer, payment, inventory, account or organizational document is connected. The tools run only registered callables against an owned isolated fixture root and do not execute generated code, arbitrary paths or remote services.

## Evidence and release boundary

`SYNTHETIC_CONTRACT_TEST` labels scripted CPU checks. `STORED_TRACE_REPLAY` labels audits of saved text, actions and states. Neither label means new model inference. Model measurements and failed infrastructure attempts are recorded separately. The CPU auditor cannot reproduce GPU generation or independently re-tokenize inputs without the external pinned tokenizer.

Codex assisted the source, tests and reports. A separately written CPU calculation path is not human peer review, official model/engine endorsement or external GPU replication. The current authorization ends with local implementation, registered P0/P1/P2 execution when gates permit, and a review package. Git stage/commit/push/PR/merge, GitHub Pages and Drive publication remain outside this run.

## 한국어 요약

직접 구현한 부분은 같은 출력 계약으로 정책을 비교하는 실행기, 합성 업무·도구, 관측 순서를 확인하는 채점기, 비용 장부와 저장 기록 감사입니다. 모델·캐시 커널·문법 엔진은 upstream이 제공합니다. 이전 연구의 성공 수치를 새 성과로 합치지 않으며, CPU 검사와 실제 모델 실행을 구분합니다. 가중치·KV tensor·환경·인증정보·개인 경로는 검토 패키지에 포함하지 않습니다.
