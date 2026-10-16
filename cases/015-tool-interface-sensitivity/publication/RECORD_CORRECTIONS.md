# Recorded diagnostic limitations and explicit post-hoc annotations

The A1 frozen runner starts with terminal_reason=BUDGET_EXHAUSTED. On8 malformed-only generated responses, it exits before reaching either request/tool limit but retains that initial label. The original raw JSON is preserved. postprocess/diagnostics.py derives PARSE_ERROR for these8 rows, each of which lacks a successful write, validation and final answer. Their primary failure is independently reproduced and unchanged.

One BF16 validation call uses an unknown order ID. The original event accurately records schema_result=UNKNOWN_ORDER and BAD_ARGUMENTS; the raw episode ARGUMENT_VALUE_WRONG field counts only successfully executed numerical-value errors and omitted this rejected ID. The separate annotation includes it as an argument value error.

These are metadata corrections, not changes to parser, feedback, generation or success scoring. No GPU rerun or performance-driven task change was made. The initial CPU annotation attempt missed UNKNOWN_ORDER and is retained in A1 results/attempts/diagnostic-annotations-attempt01.json; the corrected annotation has its own output identity. Public verification re-derives the current annotation from raw events. CPU audit independently reconstructs the unchanged primary endpoint for all96 episodes.
