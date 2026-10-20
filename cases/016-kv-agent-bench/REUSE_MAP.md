# Reuse and new implementation

| Source | Reused idea / boundary | Case016 change |
|---|---|---|
| `cases/009-q-serving-quality/scripts/serving.py` at Git base `0c86afa` | Own-process server lifecycle; readiness polling; raw HTTP/async request pattern | New vLLM0.32 CLI, FlashInfer KV arms, explicit bytes/context and loopback ownership. The old runner-version override, FLASH_ATTN, ignore_eos, 2048 context, fixed output length and old timing definitions are not used. |
| Case015 `followups/order-api-regression-v1/src/order_regression/{fixture,runner,scoring,audit}.py` | Isolated synthetic fixtures, immutable observations at generation start, replay and strict bool/int checks | New four-family fixtures, eight/32 tools, raw SSE and token records. Public validation checks types/invariants, not the hidden gold amount. |
| Case015 A1 terminal history and P2 report | Separate actual budget use from malformed JSON | Case016 starts terminal as UNKNOWN and assigns the actual exit branch. PARSE_STOP with remaining budget is not budget exhaustion. Original Case015 files are preserved. |
| vLLM `5df63a7`, `tests/kernels/attention/test_flashinfer.py` | NVFP4 page writer and dequantization test pattern (Apache-2.0) | Small GPU check uses Qwen's 32 query/8 KV heads, head_dim128; adds a separate nibble/scale decoder of the same compressed bytes. Full upstream pytest suite is not claimed. |

All engine, attention kernels, model weights and tokenizer remain upstream implementations. Case016 implements fixture generation, evidence placement, the closed tool registry, observation/scoring orchestration, raw client records and CPU audit. No shared package or earlier case is edited.
