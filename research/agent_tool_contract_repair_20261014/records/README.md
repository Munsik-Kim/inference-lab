# Tool-name contract repair for the Case 013 follow-up

This implementation compares two independent changes on the original eight observed DEV tasks: explicit registry-aware error feedback and constrained decoding of tool-name strings. The quantized model artifacts, arguments, final-answer scorer and task fixtures remain unchanged. It is a system mitigation experiment, not a new quantizer or a held-out quality evaluation.

`BASELINE` uses the original contract. `FEEDBACK` distinguishes unknown tool names from invalid arguments and lists registered names. `ENUM` masks only the top-level tool-call name value; it does not force tool use or repair answers/arguments. `BOTH` combines the changes. BF16 and RTN have baseline controls; D16 and D4 run all four conditions.

See [the Korean execution guide](README.ko.md) for the actual prepare/launch/audit entry points. CPU tests require only Python; GPU execution requires the existing reference environment, model snapshot and original saved RTN/D16 projection artifacts. The 3,600-second conservative worker budget includes loading, CPU decoding and serialization. No model download, fitting, package upgrade or remote publication is performed.

The local review bundle contains raw synthetic-task events, inputs and a model-free CPU auditor. Its external-preservation skip flag explicitly distinguishes replay of included records from certification of absent original models/cases. Runtime is dequantized BF16 reference arithmetic, not packed serving performance. ASCII schema names/keys and batch1 are supported; escaped name/key forms and combined marker/name tokens are rejected rather than silently repaired.
