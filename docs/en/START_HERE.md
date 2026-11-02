# Start with your question

English | [한국어](../ko/START_HERE.md) · [Home](../../README.md)

The [bilingual homepage](https://munsik-kim.github.io/inference-lab/en/index.html) opens the recorded results without installation. Choose one of three routes.

## Review the engineering

Read [selected work A/B/C](PORTFOLIO.md), the engineering decision, then its code and tests. A connects [Case008 Q storage and reload](https://munsik-kim.github.io/inference-lab/en/case008.html#track-q) → [Case009 actual requests](https://munsik-kim.github.io/inference-lab/en/case009.html#serving). C separates the [Case017 report](../../cases/017-agent-contract-cost-bench/REPORT.md) from [Case018 state and final reporting](CASEBOOK.md#case-018). The [earlier Case013 follow-up](../../research/agent_tool_contract_repair_20261014/README.md) is a separate study.

## Use a tool

Run the [small CPU example](https://munsik-kim.github.io/inference-lab/en/compare.html#run) to validate and compare two answer JSONL files. Start with `report.md`, inspect per-question changes in `pairs.csv`, and use `report.json` programmatically. Your records need matching question identifiers and ground-truth/grading rules. [Prepare your own records](https://munsik-kim.github.io/inference-lab/en/compare.html#own-records).

For supported-task evaluation, see [inference-engineering-kit](https://github.com/Munsik-Kim/inference-engineering-kit). For actual GPU execution requirements, use the [model guide](LLM_GUIDE.md).

## Review research and audit records

The [public research map](https://munsik-kim.github.io/inference-lab/en/index.html#research) leads from questions and bounded findings to the canonical repositories. Check this repository’s numbers with [CPU record audits](GETTING_STARTED.md#cpu-checks), or [open the site locally](GETTING_STARTED.md#offline-explorers). Recalculating stored records is neither a new GPU experiment nor external review.

## Find a case

These studies use different settings and inputs. You do not need to read them all in order. [Search and topic filters](https://munsik-kim.github.io/inference-lab/en/index.html#cases) · [Read the measurements](https://munsik-kim.github.io/inference-lab/en/measurements.html) · [Glossary](GLOSSARY.md).

<a id="1-what-problem-does-this-address"></a>
<a id="2-read-online-first"></a>
<a id="3-choose-an-instruction-path"></a>
<a id="3-reference-guides"></a>
<a id="4-how-do-the-cases-connect"></a>
<a id="5-download-or-check-records-when-needed"></a>
<a id="6-related-implementations-that-change-one-layer"></a>
<a id="api-rename-regression-testing"></a>
<a id="case015-cpu"></a>
<a id="case017--agent-contractcost-bench"></a>
<a id="case018--state-recovery-and-final-report-consistency"></a>
<a id="case019--learning-before-the-loss-moves"></a>
<a id="deeper-research-quality-compensation-at-identical-int4-storage"></a>
<a id="start-here-diova"></a>
- [001 — What if a model will not run on the GPU?](CASEBOOK.md#case-001)
- [002 — Do fewer bits save memory and time?](CASEBOOK.md#case-002)
- [003 — How close is a simpler calculation?](CASEBOOK.md#case-003)
- [004 — Does a faster core operation make a cheaper call?](CASEBOOK.md#case-004)
- [005 — Can a setting balance speed and error?](CASEBOOK.md#case-005)
- [006 — Which individual answers change?](CASEBOOK.md#case-006)
- [007 — Which groups should a smaller block keep?](CASEBOOK.md#case-007)
- [008 — Can a changed model be saved and reloaded?](CASEBOOK.md#case-008)
- [009 — How do longer decode and concurrency change cost?](CASEBOOK.md#case-009)
- [010 — Can low-bit state restart after failure?](CASEBOOK.md#case-010)
- [011 — What changes when only the readout changes?](CASEBOOK.md#case-011)
- [012 — Where should image-generation time go?](CASEBOOK.md#case-012)
- [013 — Does lower weight error improve real answers?](CASEBOOK.md#case-013)
- [014 — Can quality improve at the same 4-bit storage?](CASEBOOK.md#case-014)
- [015 — Can an AI finish the same task after a tool rename?](CASEBOOK.md#case-015)
- [016 — Native NVFP4 KV capacity and serving costs?](CASEBOOK.md#case-016)
- [017 — Agent Contract–Cost Bench](CASEBOOK.md#case-017)
- [018 — Do recovered state and final reports both agree?](CASEBOOK.md#case-018)
- [019 — Can learning progress while loss is flat?](CASEBOOK.md#case-019)
- [020 — Does smaller storage also run faster?](CASEBOOK.md#case-020)
- [021 — Why can the same INT4 artifact differ?](CASEBOOK.md#case-021)
