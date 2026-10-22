# Case018 — State Recovery and Final-Report Consistency

## Question and implementation

We built a small tool environment for recovering wrong-target inventory changes, duplicate transfers and partial debits. The evaluator checks the terminal state, validation actually observed by the model, final quantities and two fixed CPU continuations. These continuations are evaluator programs, not autonomous follow-up work by the model.

A reverses the wrong target; intended work is deferred. B retains one valid transfer and compensates only the duplicate. C reverses a partial debit. Hidden checkpoints and correct scripts are not sent to the model. Validation checks only public invariants.

## Method and frozen contract

K0=BF16 KV; K1=native NVFP4 KV; both use Qwen3-4B-Instruct-2507 BF16 weights, revision cdbee75f17c01a7cc42f958dc650907174af0554. R0 accepts a well-formed final and scores afterward. R1 checks whether successful validation of the current scope/digest/version was actually observed before final. It neither checks final quantities against gold nor repairs state. Both use grammar G, strict parser, eight APIs, greedy decoding and common limits:16 requests,512 tokens/request,8192 output/episode,24576 context,180/900 seconds. The original model-success expansion threshold is null.

D is a recovered:true declaration, including rejected finals. B is business restoration, P is protection throughout recovery, Fresh is pre-generation observation of current successful validation. Q=B∧P∧C1∧C2; W=Q∧accepted final∧correct final∧Fresh. Without accepted final, terminal Q is still evaluated, W is0 and Fresh is a last-generation diagnostic. Restoring stock but reporting old quantities can yield Q1/W0.

## HOLDOUT results and paired comparison

On24 new synthetic HOLDOUT tasks, both policies scored Q24/W0 with BF16 KV and Q19/W5 with native NVFP4 KV. The freshness guard blocked15 ineligible final attempts across five NVFP4 episodes but recovered no Q/W successes; output cost increased18.69%. Weights were BF16 in both configurations.

| KV | Policy | Q /24 | W /24 | Requests | Input tokens | Output tokens | Guard rejections | Episode wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | R0 | 24 | 0 | 168 | 323375 | 10953 | 0 | 164.144812115 |
| BF16 | R1 | 24 | 0 | 168 | 323375 | 10953 | 0 | 164.157640174 |
| NVFP4 | R0 | 19 | 5 | 186 | 369881 | 11568 | 0 | 179.978763365 |
| NVFP4 | R1 | 19 | 5 | 206 | 439813 | 13730 | 15 | 213.922486411 |

Q checks business restoration, protection throughout recovery and the two fixed continuations. W additionally requires an accepted, correct final answer and observed current successful validation. This is exploratory measurement on a small synthetic sample; metric rankings differ. It does not establish KV quality equivalence, serving speed or internal memory mechanisms.

There are24 independent tasks ×four conditions=96 episodes, eight per family. The registered family-stratified paired task bootstrap uses10000 draws, seed61022018, retaining all four conditions jointly; intervals are exploratory pointwise percentile95%. R1−R0 Q/W is0pp [0,0] in both KV settings, not equivalence evidence. K1−K0 Q is−20.83pp [−33.33,−8.33]; W is+20.83pp [8.33,37.50]. B/C1/C2 success sets are identical in the actual P1/P2 samples; added discrimination by continuations was not observed. P0 synthetic counterexamples validate the evaluator separately.

## Posthoc quantity observation diagnostic

Posthoc analysis confirmed that all76 Q1/W0 episodes reported their first observed quantities. Among91 accepted finals, all13 with observed current record quantities matched the terminal quantity map; all78 without them differed. The13 include three accurate reports of incorrectly recovered states, so reporting consistency is not successful recovery. Five guard-limited episodes have no accepted-final comparison. This is an association, not a forced-reread intervention.

| KV | Policy | Accepted / no final | Read0 wrong | Read0 right | Read1 wrong | Read1 right |
|---|---|---|---:|---:|---:|---:|
| BF16 | R0 | 24 / 0 | 24 | 0 | 0 | 0 |
| BF16 | R1 | 24 / 0 | 24 | 0 | 0 | 0 |
| NVFP4 | R0 | 24 / 0 | 16 | 0 | 0 | 8 |
| NVFP4 | R1 | 19 / 5 | 14 | 0 | 0 | 5 |

Read0/1 denotes actual delivery of current quantities for every publicly required record through read_records. Any later quantity change, including change-and-restore, expires that record observation. A metadata-only global revision change does not. A stricter after-all-state-mutations read flag is stored separately; flags coincide in these measured samples. Neither final correctness nor checkpoint gold enters the flag. The anchor is accepted-final pre-generation available IDs; without final it is last-generation diagnostic and final comparisons are NA. Of five guard-limited cases, three have current-read1 and two current-read0. Observed0/correct-report and observed1/incorrect-report accepted cells are both zero; this is not causal evidence.

Separate P1 auxiliary analysis: per six BF16 episodes, read1/correct-report2 and read0/incorrect-report4; NVFP4 R0 has six read0/incorrect reports, R1 five plus one no-final. All18 P1 Q1/W0 cases match the first observed read. Phases are not pooled. Q1 subgroup tables are outcome-selected auxiliary summaries retained in the [ledger](publication/summary/quantity_ledger.json). The already-seen76 cases and A00 motivated this analysis; [definitions and registration](publication/analysis/ANALYSIS_FREEZE.json) precede the new aggregate, not unseen-results preregistration.

## Recorded episodes

A00 K0: after6/6 compensation and release, x/y are43/28; current valid:true is observed, but final reports37/34:Q1/Fresh1/W0. K1 reads43/28 after repair, validates and reports correctly:W1. A06 K1 applies40/30 compensation, leaving80/0 and increasing total stock by10:P0/Q0. Accurately reporting these wrong values is not recovery. C01 K1 subtracts4 from the destination while undoing a partial debit, leaving62/12. R0 accurately reports that wrong state:Q0/Fresh0/W0. R1 receives three final rejections without repairing it. [Selected excerpts](publication/summary/selected/) retain original raw text and observation IDs; complete records are archived. Summaries are not raw records.

## Costs, KV storage and technical limits

NVFP4 R0→R1:186→206 requests; input369881→439813(+18.91%), output11568→13730(+18.69%), episode wall179.978763365→213.922486411s(+18.86%), with unchanged Q19/W5. Costs cover all24 episodes including failures, not successful-only latency, billing or energy. Cost per zero successes is undefined. Input and output are not interchangeable compute costs.

P1 has180 DEV requests plus six TECH attempts (four complete/two cancelled); P2 has728 HOLDOUT plus three complete TECH requests. Cumulative917 attempts/915 complete, observed output lower bound57987 tokens; engine-generated token costs for the two cancellations remain null. P2 HOLDOUT input/output1456444/47204, TECH2901/39. Worker439.692244164+855.069382600=1294.761626764s includes failed probes, load/compile/shutdown and overlaps episode wall. New CPU publication cost is recorded separately.

Actual approximately4GiB pools: BF16 4293918720B,147456B/token,29120slots; NVFP4 4294508544B,41472B/token,103552slots. Historical receipts retain packed uint8+FP8 block scales,36-layer write/read and finite attention outputs. The3.556× ratio is token-slot capacity, not whole-VRAM savings, single100K context or speed superiority. Runtime is vLLM0.32.0/FlashInfer0.7.0.post1/xgrammar0.2.8; kernel paths also differ. Failed attempts and upstream force-kill/semaphore warnings are retained. No new profiling/GPU/model requests.

## Reproduction, sources and conclusion

```bash
python3 -B cases/018-agent-state-recovery/publication/run_checks.py --output /tmp/case018-public-check-new --tests
```

The public archive restores120 model endpoints and26 P0 synthetic traces, replayed with phase-specific frozen sources. ReferenceState transitions, protection, continuations and SSE assembly are independent; strict parser, serialization, public builder and wire rendering are shared. Stored booleans are comparison targets, not verdict inputs. Prior tokenizer re-encoding receipts are reused; this CPU path loads no model/tokenizer/SDK. This is same-machine clean-copy validation, not external human or GPU reproduction.

We observed an association between current quantity re-observation and final-report consistency in stored trajectories. We do not conclude that rereading caused correctness or that NVFP4 improved memory. Forced reread, new policies and P3 model experiments are NOT_RUN. This independently adapts recovery-followed-by-action evaluation to small tools; it does not reproduce the full GUI/on-policy paper. [SOURCE_MAP](SOURCE_MAP.md) · [REUSE_MAP](REUSE_MAP.md) · [NOTICE](NOTICE.md) · [Reproduction](REPRODUCE.md).
