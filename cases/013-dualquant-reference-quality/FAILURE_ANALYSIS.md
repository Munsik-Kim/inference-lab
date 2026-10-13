# Failure analysis / 무엇이 실패했는가?

**OBSERVED:** D16의 4B weight MSE와 표본 위치 KL은 RTN보다 작았지만 PPL은 조금 높고 자유 도구 작업은 6/8로 줄었습니다. 모든 방법의 replay는 8/8이지만 free rollout은 달랐습니다. Packed 전체 모델의 수치 대응은 사전 기준을 통과하지 못했습니다.

이 문서는 기존 기록에 대한 **POST_HOC_SAME_RECORDS** 분석입니다. 정답·parser·도구 alias·주 점수는 바꾸지 않습니다. 아래 유형은 겹칠 수 있어 합계를 독립 실패 수로 더하지 않습니다. Tool error events와 실패 episode 수는 별도입니다.

## Exact long answers: values and output contracts

| Method / task / length | Recorded output or event | Interpretation | Source |
|---|---|---|---|
| D16 / DEV-F3-0 /8192 | answer=`READY`, gold=`READY-0` | Identifier suffix loss; actual value error | [raw JSONL](records/raw/4b/D16/long_dev.jsonl) |
| RTN / DEV-F2-0 /16384 | answer=`7 * 3 = 21`, gold=`21` | Correct final number; exact string contract fails | [raw JSONL](records/raw/4b/RTN/long_dev.jsonl) |
| D16 / DEV-F2-0 /8192 | Correct expression with trailing extra characters | Expression contract failure plus strict-document JSON issue | [raw JSONL](records/raw/4b/D16/long_dev.jsonl) |
| D4 / DEV-F1-0 /8192 | `Answer`/`Evidence` instead of `answer`/`evidence` | Content can be right; key contract differs | [raw JSONL](records/raw/4b/D4/long_dev.jsonl) |
| D4 / DEV-F2-1 /8192 | answer=`8`, gold=`32` | Operand returned instead of product | [raw JSONL](records/raw/4b/D4/long_dev.jsonl) |
| D4 / DEV-F2-2 /16384 | Explanation used96outputtokens before required JSON | No extractable answer object; output budget reached | [raw JSONL](records/raw/4b/D4/long_dev.jsonl) |
| BF16 / DEV-F1-0 /8192 | answer=`GATE-41`, correct evidence | Successful retrieval control | [raw JSONL](records/raw/4b/BF16/long_dev.jsonl) |

The exact scorer allows the first valid JSON object; it does not require the whole response to be JSON. Consequently, strict-document invalidity alone is not automatically a primary failure. [Scalar/post-hoc columns](records/long_context_dev.csv) preserve the distinction.32K policy-skipped cells are not failures and do not enter these examples.

## D16 fixture editing: read the actual turns

Base FREE AGENT-6 follows this sequence in [the raw record](records/raw/4b/D16/agent_dev.jsonl):

1. First model turn calls `ApplyFixturePatch` and `ValidateFixtureTest`.
2. Both are rejected as unknown tool/incorrect argument keys.
3. Next turn says it will correct the names but again calls the same CamelCase identifiers; both are rejected.
4. Final answer is `valid:true`, while the saved fixture remains unchanged.

AGENT-7 has the same name-error/reselection pattern. The two episodes produce **8 tool-error events**, not8 failed tasks. Together they account for the base D16's two failed FREE episodes. Runtime retries are0; four repeated identical tool choices are a separate post-hoc count. Final JSON parsing succeeds, while action conditions fail.

The original registry requires `apply_fixture_patch` and `validate_fixture_test`. No aliases or automatic repair were inserted. The unchanged final fixture files are [AGENT-6](records/raw/4b/D16/AGENT-6-FREE-config.json) and [AGENT-7](records/raw/4b/D16/AGENT-7-FREE-config.json). This is observed model behavior under the shared tool contract, not proof of its numerical cause.

**Successful controls:** RTN AGENT-6/7 completes the required edit and validation in [its events](records/raw/4b/RTN/agent_dev.jsonl). D16 AGENT-2 correctly reads a record, calls calculate and returns total30. D16 therefore does not fail every task family. Replay supplies the relevant tool result and can succeed without demonstrating correct free tool selection.

## D4 and other failure types

[D4 base events](records/raw/4b/D4/agent_dev.jsonl) include statements that the model has no access instead of using an available tool, background text used as status, `Apply_fixture_patch` rather than the registered name, and validation on an unmodified fixture. Two parse-error events and two tool-error events occur across the base FREE cohort; all eight final task conditions fail. These overlapping causes are not summed into a new failure rate.

## What remains a hypothesis

| Level | Statement |
|---|---|
| OBSERVED | Weight MSE/selected-position KL decrease while4B PPL/free-task success worsens; replay and free differ; full packed gate fails. |
| HYPOTHESIS | Task-sensitive tool-identifier margins may change; FP32 rescale/BF16 rounding, scale representation or kernel arithmetic may accumulate differently. |
| NOT_TESTED | Identical-history token log-probabilities at the first action divergence; gold NLL at KL positions; first differing packed block; controlled inner-mode/BF16-scale comparisons. |

No single cause is established. These records do not show complete state-information loss, a failure of all decoders, or failure of the paper across models. Follow-up interventions are [proposed only](ROADMAP.md).
