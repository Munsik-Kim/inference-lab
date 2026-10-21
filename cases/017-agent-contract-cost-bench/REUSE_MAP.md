# Reuse map / 재사용과 직접 구현

Case017은 같은 업무를 끝내는 네 가지 호출 정책과 두 KV 캐시 형식을 비교합니다. 이전 연구의 모델·환경과 작은 계측 코드를 재사용하고, 이번 업무·출력 계약·예산·채점·분석은 새 모듈로 분리했습니다. 이전 성공 수치는 이번 실험의 측정값이나 표본으로 합치지 않습니다.

The reused engine and inspection code provide the execution substrate. The new contribution is the shared action contract, isolated business fixtures, policy/budget orchestration, observation-aware endpoint, and replayable cost/quality comparison.

## Source baseline and identities

- Repository base: `740a26f013c4a516b8322c0c07e077a7667bd8f1`.
- Model: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Existing installed environment: vLLM 0.32.0, xgrammar 0.2.8, FlashInfer 0.7.0.post1, Torch 2.13.0, Transformers 5.19.0, HTTPX 0.28.1. Read-only installed-file and package metadata hashes are in [environment identity](provenance/environment_identity.json).
- Local model file hashes are in [model identity](provenance/model_identity.json). They identify the actual reused snapshot; they are separate from the Git base and execution source hash.
- [Execution freezes](provenance/execution_freeze_p0_r2.json) enumerate the Case017 execution modules, five original import files, inputs and identity receipts. Later stage freezes, if created, identify their own registered input bundle.

## Direct runtime imports: five original files

These files are imported from Case016 through explicit read-only `PYTHONPATH` roots. Their source bytes are not copied into an editable Case017 replacement or changed for this run.

| Original file | Imported role | Case017 use |
|---|---|---|
| [nvfp4_technical/__init__.py](../016-kv-agent-bench/followups/nvfp4-technical-r1/src/nvfp4_technical/__init__.py) | Package initialization | Resolve the existing inspection package. |
| [nvfp4_technical/worker.py](../016-kv-agent-bench/followups/nvfp4-technical-r1/src/nvfp4_technical/worker.py), `TechnicalInspector` | Tensor/config inspection, parameter samples, finite attention hooks | `p0_inspector.ContractInspector` extends the observer with Case017 backend/context/scheduler fields. |
| [kv_agent_bench/__init__.py](../016-kv-agent-bench/src/kv_agent_bench/__init__.py) | Package initialization | Resolve the original observer dependency. |
| [profiler_extension.py](../016-kv-agent-bench/src/kv_agent_bench/profiler_extension.py), `ProfileInspector` | Short P0-only CUDA kernel-name collection | Observe actual prefill/decode paths outside quality episodes. |
| [worker_inspection.py](../016-kv-agent-bench/src/kv_agent_bench/worker_inspection.py), `CacheInspector` | Actual cache tensor/storage/layout and runtime flags | Verify BF16 or packed NVFP4 storage with FP8 block-scale views. |

`parameter_sample_sha256` covers the first/last eight values of each named parameter. It is a runtime change detector, not a full GPU weight hash. Full snapshot-file hashes are checked separately. P0 finite hooks observe actual attention outputs; they do not constitute an independent full-vocabulary logit or language-quality test.

## Adapted code and design

| Prior source or observation | New Case017 implementation | Reuse boundary and change |
|---|---|---|
| Case016 `StreamAssembly`, `chat`, `request_cap` in [vllm_client.py](../016-kv-agent-bench/src/kv_agent_bench/vllm_client.py) | [client.py](src/agent_contract_cost/client.py) | Adapted content/SSE/request accounting. All policies use content-only JSON envelopes; native tool extraction and Hermes dispatch are rejected. Raw token IDs and content remain separate. |
| Case016 technical `OwnedServer` in [server.py](../016-kv-agent-bench/followups/nvfp4-technical-r1/src/nvfp4_technical/server.py) | [server_manager.py](src/agent_contract_cost/server_manager.py) | Adapted owned loopback process management. Separate port/cache/log roots, explicit xgrammar backend, no native tool parser, no attachment to or termination of other services. |
| Case015 exact registry and A1 fixture/observation contracts in [registry.py](../015-tool-interface-sensitivity/src/interface_sensitivity/registry.py), [A1 runner](../015-tool-interface-sensitivity/followups/order-api-regression-v1/src/order_regression/runner.py) | [contracts.py](src/agent_contract_cost/contracts.py), [tools.py](src/agent_contract_cost/tools.py), [runner.py](src/agent_contract_cost/runner.py) | Conceptual reuse of exact registered names, pristine state and immutable pre-generation observations. No Case015 model runner, parser, fixture or scorer is imported. Case017 rejects the whole response unless it is one complete object. |
| Case015 P2 [requests.py](../015-tool-interface-sensitivity/followups/json-recovery-p2-v1/src/json_recovery_p2/requests.py), `retry` | [policies.py](src/agent_contract_cost/policies.py) | Conceptual separation of failed assistant raw, host user feedback and genuine tool observations. Case017 N/E are independent free policies and can retry their first contract rejection once; they do not branch from another policy's measured output. |
| Case016 baseline audit F02/F13 in the [preserved audit archive](../016-kv-agent-bench/followups/baseline-audit-r1/Case016_baseline_audit_v1.zip) | [policies.py](src/agent_contract_cost/policies.py), [runner.py](src/agent_contract_cost/runner.py) | New explicit final-as-attempt budget; request attempts are debited before HTTP and separated from received responses and partial output costs. Historical Case016 scores/counters are unchanged. |
| Existing observation-aware endpoint and replay principles | [scoring.py](src/agent_contract_cost/scoring.py), [audit.py](src/agent_contract_cost/audit.py) | New family-specific endpoint and replay implementation. The auditor separately computes desired states, Decimal half-up amounts, protected-field preservation and dependency chains; it does not trust stored success flags. |

## New implementation

[fixtures.py](src/agent_contract_cost/fixtures.py) and [tools.py](src/agent_contract_cost/tools.py) define owned synthetic purchase, stock-transfer and document-metadata tasks. [schemas.py](src/agent_contract_cost/schemas.py) derives the G server schema solely from public action contracts. [prepare.py](src/agent_contract_cost/prepare.py) and [experiment.py](src/agent_contract_cost/experiment.py) register inputs, freeze dependencies, check runtime identity and enforce stage gates. [analysis.py](src/agent_contract_cost/analysis.py) builds paired tables and exploratory intervals from audited episode rows.

The host checks unique array items in every policy. The installed grammar backend did not enforce `uniqueItems` in CPU probes, so that keyword is explicitly omitted from the server grammar. This is the registered `TOOL_BRANCH_SCHEMA_HOST_ONLY_UNIQUE_ITEMS` scope, not a claim that G enforces every host constraint.

## Rights and assistance

The repository's own prior code follows the root [Apache-2.0 license](../../LICENSE). vLLM, FlashInfer, xgrammar, Qwen and other dependencies retain their upstream notices and licenses. No model weights, native library binaries, compiled kernels or third-party source trees are bundled here. See [NOTICE](NOTICE.md).

Codex assisted implementation, testing and documentation. The independent CPU calculation is a separate code path; it is not independent human review or an independent rerun of model inference.
