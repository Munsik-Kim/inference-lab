# Case018 — State Recovery and Continuation Bench

Correct inventory quantities do not necessarily mean recovery is complete: an active processing entry can still block the next transfer. This small synthetic environment separates a recovery declaration, restored business values, and the results of subsequent work.

**Scope: P0 CPU contracts only.** Scripted clients test the harness and evaluator; these are not model measurements. No model, tokenizer, server, grammar backend or GPU was run. No HOLDOUT input or gold was generated.

Six DEV tasks cover a wrong-target transfer, an extra duplicate after one legitimate completion, and a partially executed transfer. Public tools expose records, incident history and processing state. Explicit stock compensation and operation release are separate actions, with current revisions required. There is no reset, gold diff or automatic repair tool.

Two fixed evaluator programs run independently from cloned terminal and reference snapshots. Each reads the current revision before mutation; business arguments never depend on gold. These continuations are CPU programs, not autonomous agent behavior, and two successful finite paths do not prove equivalence for every future action.

The primary Q endpoint checks business restoration, protection throughout recovery and both continuations. W also requires an accepted, correct final response after observed validation of the current state. Q=1/W=0 is meaningful. A public invariant validator does not reveal or check checkpoint equality.

Future comparisons keep BF16 weights fixed and pair BF16/native NVFP4 KV with R0 (accept schema-valid final) and R1 (require observed current validation). All use the same public grammar. Actual integration gates remain NOT_RUN. The future DEV expansion gate has **no model-success threshold** (`null`); HOLDOUT generation is blocked until completion, freeze and technical identity gates pass.

See [the Korean P0 report](P0_REPORT.ko.md), [protocol](specs/protocol.json), [API](specs/tools.json), [source mapping](SOURCE_MAP.md), [reuse boundaries](REUSE_MAP.md), and [CPU commands](REPRODUCE.ko.md). This independently implements a limited principle inspired by ReTrace and a component-versus-joint evaluation view from ACE-Compression, not their full benchmarks or theory. Codex assisted implementation. Existing Case017 evidence is read-only and is not pooled into this case.
