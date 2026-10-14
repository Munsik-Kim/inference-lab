# Tool-name contract repair for the Case 013 follow-up

**D16 recovered from 6/8 to 8/8; D4 from 0/8 to 2/8** on the same existing DEV tasks. Enum conditions emitted zero unregistered-name calls. Six other D4 task failures remain. Quantized weights are unchanged.

[한국어](README.ko.md) · [Implementation](guard.py) · [All outcomes](records/summary.json) · [Full report](records/REPORT.md) · [Sources](NOTICE.md)

| Condition | D16 success | D4 success |
|---|---:|---:|
| Original | 6/8 | 0/8 |
| Feedback only | 8/8 | 0/8 |
| Name enum only | 8/8 | 2/8 |
| Both | **8/8** | **2/8** |

BF16/RTN controls each passed 8/8. Feedback recovered D16 after errors but retained four initial unknown-name calls. Enum/both had zero. An additional benefit from feedback on top of enum decoding was not observed. **D16's extracted final answers are identical before and after; actual patch/validation execution determines success.** [Raw examples](records/failure_examples.json) · [Verified D16 configuration](records/recommended_preset.json)

Qwen3-4B-Instruct-2507 on RTX 5080, dequantized BF16 reference. The 80 episodes reuse eight task IDs across ten conditions. No generalization, noninferiority, packed speed or capacity benefit is established. The [earlier diagnostic](../agent_stability_20261013/README.md) and [original Case013](../../cases/013-dualquant-reference-quality/README.md) remain unchanged.

From the repository root, without a model and using fresh output paths:

```bash
STUDY=research/agent_tool_contract_repair_20261014
python3 -B "$STUDY/verify_public.py" --study "$STUDY" --output /tmp/tool-repair-public-new.json
python3 -B "$STUDY/analyze.py" --result "$STUDY/records" --inputs "$STUDY/records/inputs" --output /tmp/tool-repair-audit-new --skip-external-preservation
diova-compare validate --input research/agent_tool_contract_repair_20261014/records/compare_inputs/D16-BASELINE.jsonl
diova-compare compare --baseline research/agent_tool_contract_repair_20261014/records/compare_inputs/D16-BASELINE.jsonl --candidate research/agent_tool_contract_repair_20261014/records/compare_inputs/D16-BOTH.jsonl --output /tmp/tool-repair-compare-new
```

The skip flag distinguishes included-record replay from certification of absent external originals. CLI intervals are descriptive on the observed DEV records. [Original verification](records/verification.json) · [Executed CLI receipts](records/compare_verification.json) · [Review ZIP](../../downloads/case013_tool_contract_repair_review_v1.zip)

This implementation compares two independent changes on the original eight observed DEV tasks: explicit registry-aware error feedback and constrained decoding of tool-name strings. The quantized model artifacts, arguments, final-answer scorer and task fixtures remain unchanged. It is a system mitigation experiment, not a new quantizer or a held-out quality evaluation.

`BASELINE` uses the original contract. `FEEDBACK` distinguishes unknown tool names from invalid arguments and lists registered names. `ENUM` masks only the top-level tool-call name value; it does not force tool use or repair answers/arguments. `BOTH` combines the changes. BF16 and RTN have baseline controls; D16 and D4 run all four conditions.

See [the Korean execution guide](README.ko.md) for the actual prepare/launch/audit entry points. CPU tests require only Python; GPU execution requires the existing reference environment, model snapshot and original saved RTN/D16 projection artifacts. The 3,600-second conservative worker budget includes loading, CPU decoding and serialization. No model download, fitting, package upgrade or remote publication is performed.

The local review bundle contains raw synthetic-task events, inputs and a model-free CPU auditor. Its external-preservation skip flag explicitly distinguishes replay of included records from certification of absent original models/cases. Runtime is dequantized BF16 reference arithmetic, not packed serving performance. ASCII schema names/keys and batch1 are supported; escaped name/key forms and combined marker/name tokens are rejected rather than silently repaired.
