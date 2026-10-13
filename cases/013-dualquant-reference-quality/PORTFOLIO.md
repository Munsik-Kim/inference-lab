# Case 013 — Implementation and evidence

**Problem:** a smaller weight error does not specify the quality or deployment cost of a complete model.

**Built:** a group64 INT4 solver, nibble codec with all scale metadata, scale-preserving Linear adapter, sequential full-model runner, paired long-input and local-tool fixtures, and a GPU-free recorded-results auditor.

**Stack:** PyTorch, Transformers, NumPy, SDPA; an existing vLLM/Marlin packed probe on RTX 5080 SM120.

**Observed:** D16 reduced weight MSE against RTN in both Qwen models. PPL improved in 0.6B and slightly worsened in 4B. The 4B D16 free-agent pilot completed 6/8 versus RTN 8/8; activation INT4 completed 0/8. Full packed-model numeric correspondence did not pass the frozen gate.

**Inspect:** [solver/adapter](source/paper_int4.py), [storage](source/reference_int4.py), [runner](source/paper_model_runner.py), [independent audit](analysis/audit.py), [tests](tests/test_records.py), [scope and exact figures](REPORT.md), [reuse/contributions](NOTICE.md).

This demonstrates implementation, artifact handling and failure-aware evaluation. It does not establish serving acceleration, stock GPTQ superiority, formal noninferiority or maximum context capacity.
