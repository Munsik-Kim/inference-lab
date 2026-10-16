# Reuse and new implementation

| Component | Pinned source / identity | This study |
|---|---|---|
| Eight owned fixture tasks, gold, original4K/8K messages, six schemas | Case013 at `e647b474bdc9fa558d851bd9c998b664b0e33638`; [input identity](provenance/input_manifest.json) | Byte-preserving copies in `data/`; only model-visible typed API spans change |
| Six fixture functions and atomic JSON writer | [exact excerpt](src/interface_sensitivity/fixture_original.py), original `Worker.tool` and `atomic`; [hash](provenance/reuse.json) | Exact names checked first; then dispatch to the same implementation using semantic ID |
| Model loader, reference dtype and projection replacement | Three [byte-identical original modules](src/interface_sensitivity/upstream_case013/); original/public hashes matched | Read-only adapter, preflight504file SHA checks; solve/encode/serialize/full sweep guarded against execution |
| D16 arithmetic | `ReferenceLinear`: FP32 X times beta_inverse, cast BF16, BF16 W_pre; original bias | Unchanged;252projections, no activationINT4 or scale folding |
| Prior ENUM/feedback fixes | `research/agent_tool_contract_repair_20261014/guard.py` read for exclusion | Not imported or enabled; original generic error strings only |
| diova-compare prompt pairing | Existing0.1.2 core/readme read | Package unchanged. Per-interface adapter exports only exact shared inputs; cross-interface analyzer uses canonical task and mapping |
| Registry, typed transformation, stepwise equivalence, strict episode lifecycle, functional scorer | New Case015 CPU/GPU source | New independent evaluation code; no new quantizer |
| Analysis, saved-response replay, bilingual explorer | New Case015 source and publication scripts | CPU-only verification of saved evidence; not GPU logit replication |

No CIPR code was incorporated. The paper motivates cross-interface comparison; no training objective or reported trained result is reproduced. All reused DIOVA code retains the repository Apache-2.0 license; upstream models, paper and external source retain their own terms.
