# DIOVA · Model execution and change validation

English | [한국어](README.ko.md)

**Munsik Kim** · I build tools to run models, compare changed answers and diagnose failures. The work connects model storage and restart, matched-question evaluation and checks of actual task completion to code and auditable records.

[See selected work](https://munsik-kim.github.io/inference-lab/en/index.html#featured-stories) · [Run a model-free example](https://munsik-kim.github.io/inference-lab/en/compare.html#run)

## Choose a starting point

| What you want to check | Next step |
|---|---|
| Engineering skills and my contribution | [Design decisions, code and tests](docs/en/PORTFOLIO.md) |
| Changes in your two answer files | [diova-compare inputs and commands](packages/diova-compare/examples/compare/README.md) |
| Research findings and scope | [Cases and records](docs/en/CASEBOOK.md), [public research map](https://munsik-kim.github.io/inference-lab/en/index.html#research) |

## Selected work A · B · C

<!-- claims: c008-tracks c008-artifact-implementation -->
| Work | What I implemented and where to check it |
|---|---|
| A. Reload a smaller model | GPTQ integration, file checks and fresh-process reload: [Case008 Q](https://munsik-kim.github.io/inference-lab/en/case008.html#track-q) → [Case009 request costs and quality](https://munsik-kim.github.io/inference-lab/en/case009.html#serving). File size, GPU memory and changed answers are different measurements. |
| B. Changes hidden by an average | [diova-compare](https://munsik-kim.github.io/inference-lab/en/compare.html) checks matched inputs and reports lost and gained correct answers. Both runs score 2/4 in the small synthetic example, yet one correct answer is lost and another gained. |
| C. Completion claims and actual state | [Case017 report](cases/017-agent-contract-cost-bench/REPORT.md) · [CPU replay](cases/017-agent-contract-cost-bench/REPRODUCE.md) · [earlier implementation](research/agent_tool_contract_repair_20261014/README.md). [Case018](cases/018-agent-state-recovery/README.md) separates state/procedure fit from accurate final reporting. Case017 is exploratory, retaining an omitted expansion criterion and added costs. |

Recent **Case020** asks whether lower residency also means faster calls. **Case021** investigates where execution paths using the same compressed representation diverge. Its overall mean KL-reduction verdict remains **INCONCLUSIVE**: this evidence does not establish a reduction. [Read recent diagnostics](https://munsik-kim.github.io/inference-lab/en/index.html#latest-work).

## Tools you can use

- **diova-compare:** compares two already-run and graded JSONL files on CPU. After `validate → compare`, start with `report.md`, then inspect changes by question in `pairs.csv`. It does not run models or create ground truth for free-form answers.
- **[inference-engineering-kit](https://github.com/Munsik-Kim/inference-engineering-kit):** an installable evaluator with supported tasks and external-gold contracts. The default CPU path is separate from live model/HTTP execution; comparison reuses diova-compare. Its guide distinguishes [public v0.2.1](https://github.com/Munsik-Kim/inference-engineering-kit/releases/tag/v0.2.1) from current source.

## All cases

These 21 cases use different models, questions and conditions. Case numbers are not rankings. [Search or browse by topic](https://munsik-kim.github.io/inference-lab/en/index.html#cases).

<a id="capabilities"></a>
<a id="case-006--trace-model-changes-question-by-question"></a>
<a id="case-007--from-channel-selection-to-smaller-weight-matrices"></a>
<a id="case-008--build-reconstruct-reload"></a>
<a id="case-010--low-bit-state-storage-restart-and-memory-horizon"></a>
<a id="case-011--same-state-different-readout"></a>
<a id="case-012--image-generation-time-loops-vs-steps"></a>
<a id="case-013--does-lower-weight-error-improve-real-answers"></a>
<a id="case015--same-tools-different-interfaces"></a>
<a id="case016--native-kv-capacity-and-request-costs-at-the-same-memory-budget"></a>
<a id="case017--agent-contractcost-bench"></a>
<a id="case018--checking-inventory-recovery-and-completion-reports"></a>
<a id="case020--does-smaller-embedding-storage-also-run-faster"></a>
<a id="case021--why-can-the-same-int4-artifact-produce-different-results"></a>
<a id="compress-save-and-reload-models"></a>
<a id="contribution-and-sources"></a>
<a id="diova--model-compression-execution-and-evaluation"></a>
<a id="featured-work-a--b--c"></a>
<a id="follow-up--tool-names-and-verified-actions"></a>
<a id="go-deeper"></a>
<a id="gpu-inference-measurement-and-output-evaluation"></a>
<a id="implementation-and-technology-evidence"></a>
<a id="implementation-details-and-research-records"></a>
<a id="new-measurement--longer-decode-and-client-concurrency"></a>
<a id="project-implementation-and-upstream-components"></a>
<a id="projects"></a>
<a id="q--gptq-conversion-storage-and-fresh-runtime-execution"></a>
<a id="questions-and-recorded-tools"></a>
<a id="r--structure-aware-reload-and-fixed-weight-reconstruction"></a>
<a id="recorded-storage-and-runtime-memory"></a>
<a id="reproducible-analysis-and-result-explorers"></a>
<a id="research-and-numerical-engineering--case014"></a>
<a id="store-smaller-models-run-them-and-check-the-answers-that-changed"></a>
<a id="tech-stack"></a>
<a id="tech-stack-in-use"></a>
<a id="use-or-reproduce"></a>
<!-- claims: c007-implementation c006-implementation c006-readout c010-state-bytes -->
| Question | Observation and implementation scope |
|---|---|
| [001 — What if a model will not run on the GPU?](docs/en/CASEBOOK.md#case-001) | A before/after check of an upstream execution-path fix. Terms: SM120, vLLM, W8A8. |
| [002 — Do fewer bits save memory and time?](docs/en/CASEBOOK.md#case-002) | Official model comparisons on document extraction: lower resource use, but low task accuracy. Terms: BF16, FP8. |
| [003 — How close is a simpler calculation?](docs/en/CASEBOOK.md#case-003) | Numerical error checks with different comparison scales. Terms: EFQ-Softmax, FP32 simulation. |
| [004 — Does a faster core operation make a cheaper call?](docs/en/CASEBOOK.md#case-004) | Attention timing with preparation included; longer inputs gained speed while output error exceeded the fixed limits. Terms: kernel, complete-call cost. |
| [005 — Can a setting balance speed and error?](docs/en/CASEBOOK.md#case-005) | Measured trade-offs; no tested changed setting met both requirements, so fresh confirmation stopped. Terms: precision, local error. |
| [006 — Which individual answers change?](docs/en/CASEBOOK.md#case-006) | Paired answer scores, choices and equal-top-score diagnostics. Terms: NLL, top-score tie. |
| [007 — Which groups should a smaller block keep?](docs/en/CASEBOOK.md#case-007) | Group selection, actual matrix slicing and evaluation on unused inputs. Terms: MLP, pruning. |
| [008 — Can a changed model be saved and reloaded?](docs/en/CASEBOOK.md#case-008) | GPTQ checkpoint integration and fixed-MLP ridge reconstruction; smaller Q files and lower R local error, with mixed gold scores. |
| [009 — How do longer decode and concurrency change cost?](cases/009-q-serving-quality/README.md) | Matched graph/eager serving curves and official quality calculations with recorded runtime-exit failures. |
| [010 — Can low-bit state restart after failure?](docs/en/CASEBOOK.md#case-010) | Stored state for one stream: 12,305 → 3,137 bytes, 74.5% smaller. State bytes are separate from total RAM or VRAM. |
| [011 — What changes when only the readout changes?](docs/en/CASEBOOK.md#case-011) | Two-tensor adaptation and reload; paired gold-score and first-error comparisons on the same state. |
| [012 — Where should image-generation time go?](docs/en/CASEBOOK.md#case-012) | Same-noise generation-step/internal-repeat runner and timing. Assessment of requested count, color and left/right conditions in all 192 images, with paired constraint gains and losses. |
| [013 — Does lower weight error improve real answers?](docs/en/CASEBOOK.md#case-013) | INT4 scale-preserving conversion/reload and matched full-model, long-input and tool-quality records; packed gate remains blocked. |
| [014 — Can quality improve at the same 4-bit storage?](docs/en/CASEBOOK.md#case-014) | PG19 text-quality gains and ArXiv limits. Quality was evaluated after BF16 restoration; VRAM savings were not measured. |
| [015 — Can an AI finish the same task after a tool rename?](https://munsik-kim.github.io/inference-lab/en/case015.html#overview) | A purchase example: look up item count and price per item, calculate the total, save and validate it. Across 16 problems: BF16 15/16→16/16, RTN 16/16→16/16, D16 12/16→12/16. [Task, terms and CPU example](cases/015-tool-interface-sensitivity/README.md). |
| [016 — Native NVFP4 KV capacity and serving costs?](cases/016-kv-agent-bench/README.md) | BF16 weights,same4GiB pool:native KV read/write,capacity,36-block throughput/queue/numerical sanity. |
| [017 — Agent Contract–Cost Bench](cases/017-agent-contract-cost-bench/README.md) | Strict JSON, actual state, fresh validation and completion costs; exploratory measurement with a disclosed gate deviation. |
| [018 — Do recovered state and final reports both agree?](docs/en/CASEBOOK.md#case-018) | Separates state/procedure fit Q from W, which also requires accurate reporting. |
| [019 — Can learning progress while loss is flat?](docs/en/CASEBOOK.md#case-019) | Studies alignment, scale and loss through partial reproduction and original controls. |
| [020 — Does smaller storage also run faster?](docs/en/CASEBOOK.md#case-020) | Separates synthetic-embedding residency, calls and measured reuse costs. |
| [021 — Why can the same INT4 artifact differ?](docs/en/CASEBOOK.md#case-021) | Single terms matched; combinations differed. The overall precision effect remains INCONCLUSIVE. |

## Guides and attribution

[START_HERE](docs/en/START_HERE.md) chooses a route; [PORTFOLIO](docs/en/PORTFOLIO.md) connects contributions and decisions to code; [CASEBOOK](docs/en/CASEBOOK.md) indexes all cases. Continue with [CPU audits versus GPU reruns](docs/en/GETTING_STARTED.md), [model guides](docs/en/LLM_GUIDE.md) or the [glossary](docs/en/GLOSSARY.md) as needed.

DIOVA stands for Deep-learning Inference Optimization, Validation & Analysis. Public models, paper methods and runtimes are reused and distinguished from my adapters, evaluators and audits. OpenAI Codex assisted implementation, analysis and documentation. [Contribution and reuse](docs/en/PORTFOLIO.md#contribution-and-reuse) · [Apache-2.0](LICENSE) · [related work](docs/related-work/README.md). Per-case upstream terms, failures and unexecuted work remain recorded.
