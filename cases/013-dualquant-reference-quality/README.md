# Case 013 — Does Lower Weight Error Improve Real Answers?

English | [한국어](README.ko.md)

Built tools to convert, pack and reload INT4 weights and to compare language scores, long-input answers and tool use on matched inputs. Applied the reference path to every Transformer projection in Qwen 0.6B and 4B on an RTX 5080, retaining failed outputs as well as successes.

The independent public-equation variant needs no private blockfmt. D16 improved PPL against RTN in 0.6B. In 4B, lower weight MSE and selected-position KL did not improve PPL or free-tool DEV. SM120 packed W4A16 ran, but full-model numeric correspondence failed; serving speed and context-capacity benefits remain unestablished.

**Row/column scales preserved · Real 4-bit packing · Matched-input quality comparison**

[Implementation](#implementation) · [Findings](#results) · [Check it yourself](#reproduce) · [Design and sources](#sources)

<a id="implementation"></a>
## What was built

| Function | Implementation and evidence |
|---|---|
| Weight-only calculation of row and column scales | [Solver](source/paper_int4.py), [equations and storage](source/reference_int4.py) |
| Save/reload codes and all required scales | [Nibble codec](source/reference_int4.py), [original artifact ledger](records/artifact_manifest.json) |
| Full-model adapter that also applies column scales to inputs | [ReferenceLinear](source/paper_int4.py), [sequential runner](source/paper_model_runner.py) |
| Separate answer/evidence scoring and actual tool-action checks | [Execution/scoring](source/paper_model_runner.py), [independent CPU audit](analysis/audit.py) |
| Separate actual packed CUDA execution from full-model correspondence | [Final probe source](provenance/source_snapshot/probe_public_marlin.py), [technical record](PACKED_PROBE.md) |

Uses PyTorch, Transformers, NumPy, SDPA and a vLLM/Marlin probe. DualQuant is the authors' method. This project implements an INT4 **reference variant** grounded in the public equations, plus storage, evaluation and audit tools. It does not reproduce the authors' complete program or private packed library.

<a id="results"></a>
## Findings

Weight reconstruction error decreased in both models, while answer quality moved in different directions.

| Model | Original BF16 PPL | Simple INT4 RTN | Row/column D16 | D16 + activation INT4 D4 |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 21.275 | 27.576 | 24.445 | 34.734 |
| Qwen3-4B-Instruct-2507 | 10.133 | 10.810 | 10.894 | 12.339 |

PPL summarizes next-symbol probability scores; **lower is better**. Each model uses a WikiText-2 subset of 32 windows of 2,048 symbols, with 65,504 scored positions. These are not full official benchmark scores or a cross-model ability ranking. [Unrounded results](records/quality_4b.csv) · [Shared display data](summary/project.json)

In the 4B free tool-use pilot, BF16 and RTN completed **8/8** tasks, D16 **6/8**, and D4 **0/8**. Extending four of the same tasks to 16K inputs gave **4/4 · 4/4 · 3/4 · 0/4**. Replay of already supplied tool results was correct in every condition; it answers a different question from choosing and executing tools.

An actual packed Marlin GEMM and one block passed the numeric gates. The full model had **2.61%** relative error against a predeclared **2%** limit. **Serving acceleration and context-capacity gains remain unestablished.** The main quality path uses reconstructed BF16 weights and matrix operations.

<a id="reproduce"></a>
## Check it yourself

Recorded-results checks use NumPy and require no GPU, model or Torch. All recorded outputs are under [records](records).

```bash
python3 -B cases/013-dualquant-reference-quality/analysis/audit.py --output /tmp/case013-audit.json
python3 -B -m unittest discover -s cases/013-dualquant-reference-quality/tests -v
python3 -B cases/013-dualquant-reference-quality/publication/verify.py --output /tmp/case013-public.json
```

Use new output paths. [Reproduction](REPRODUCTION.md) separates CPU dependencies, optional Torch synthetic checks and actual model rerun requirements. Weights and full logits are excluded from the public bundle.

<a id="sources"></a>
## Design and sources

[Full report](REPORT.md) · [Execution contract](records/reference_contract.json) · [Inputs/evaluation plan](records/experiment_manifest.json) · [Preservation/export map](provenance/export_map.json) · [Contributions and licensing](NOTICE.md)

[Paper/code/study contract](METHOD_CONTRACT.md) · [Evaluation protocol](EVALUATION_PROTOCOL.md) · [Failures and successful controls](FAILURE_ANALYSIS.md) · [Three recorded-results figures](figures/README.md) · [Proposed roadmap](ROADMAP.md)

Read the evidence in order: reconstruction error, language scores, exact long-input answers, then free tool use. The small DEV records show where those measures diverge; formal TEST and paired GPTQ serving comparisons were not run.
