# Case015 — Same Tools, Different Interfaces

Can an AI finish the same task when only its tool names change? This study gives an AI lookup, calculation and storage functions, keeps their behavior fixed, and compares task completion after changing how their names are written.

The concrete example is **calculating and saving the total for a purchase record**. The AI must look up the item count and price per item, multiply them, save the total, and check the saved record. We built tools to execute this sequence and replay the records to audit the outcome.

[Pages](https://munsik-kim.github.io/inference-lab/en/case015.html#overview) · [Report](REPORT.md) · [CPU / GPU](REPRODUCE.md) · [Stage map](STAGE_MAP.md) · [Runner](followups/order-api-regression-v1/src/order_regression/runner.py) · [Tests](followups/order-api-regression-v1/tests/test_order_contracts.py)

## The task given to the AI

1. **Read the purchase details:** retrieve the item count and price per item for the requested record.
2. **Calculate the total:** after receiving those details, pass the two numbers to the multiplication tool.
3. **Save the amount:** after receiving the product, update that purchase record’s total.
4. **Check the saved record:** use the validation tool to check the amount. Other purchase records must remain unchanged.
5. **Report completion:** after receiving validation, return the purchase ID, total and validation flag in the required format.

The experiment uses synthetic JSON files without connecting to a shop. JSON stores named fields and values as text. The initial request omits the item count and price, so the AI must receive them from the lookup tool. Amounts use integer units without decimal rounding.

## What changed in the tool names?

An API is the interface through which the AI requests a named function with input values. The two naming versions are **API_V1 (original names)** and **API_V2 (renamed)**. For example, the purchase lookup function changes from `read_order` to `readOrder`. Behavior, descriptions, input format and tool order stay fixed.

## Terms used below

| Term | Meaning in this case |
|---|---|
| Order / order ID | One purchase record / the ID identifying that record. |
| Quantity / unit price | Item count / the price of one item. |
| Tool call / API | A request from the AI to a named function, such as lookup or multiplication, with its input values. |
| Scenario / fixture | One test problem / the initial synthetic file used for it. |
| Episode | One run from the initial file until the AI stops answering. |
| Received-result history | Tool results the AI had already received **before** generating its next response, distinct from executed file changes. |
| Regression / gain | Success before the rename becoming failure / failure becoming success. |

The code fields `quantity`, `unit_price_minor` and `total_minor` mean item count, integer price per item and total amount. Original code and record field names are retained.

## Comparison and measured results

The three conditions use the same Qwen3-4B model. **BF16** is the original 16-bit model; **RTN** stores weights at four bits using nearest rounding; **D16** uses an independent reference based on public DualQuant equations, with four-bit stored weights and 16-bit activations. Stored quantized weights are restored to BF16 for this task-quality evaluation.

There are 16 test problems with different amounts and purchase IDs: four English instruction templates with four synthetic files each. Running each problem with two naming versions and three models gives **16×2×3=96 measured runs**. This application test is called **A1**.

| Model | Original names V1: success /16 | Renamed V2: success /16 | Both succeed | Success→failure | Failure→success | Both fail |
|---|---|---|---|---|---|---|
| BF16 | 15/16 | 16/16 | 15 | 0 | 1 | 0 |
| RTN | 16/16 | 16/16 | 16 | 0 | 0 | 0 |
| D16 | 12/16 | 12/16 | 12 | 0 | 0 | 4 |

Success requires the correct amount and actual saving/validation after receiving the required results. No success became a failure after the rename; one BF16 failure became a success. D16 failed on the same four problems with both naming versions.

## Implemented functions

1. Change tool names while checking that their executed behavior stays the same.
2. Record which results the AI had received before generating each response.
3. Replay recorded responses into fresh synthetic files and recompute actual state and task outcomes.

## Connection to the earlier study

The earlier **P1** stage separately varied names/order in four ways on eight existing tasks, for 96 runs across three models. Its comparison of additional sensitivity in a quantized model was +8.33 percentage points, with a 95% interval [-8.33,+25.00] including zero: inconclusive. P1 and A1 have different tasks and success criteria, so their success rates are not pooled.

| Earlier P1 task | What the AI had to do | Original task IDs |
|---|---|---|
| Read a document | Retrieve site and owner fields from a document | AGENT-0/1 |
| Read then calculate | Read item count and price, then multiply them using the calculation tool | AGENT-2/3 |
| Find the latest state | Return the latest status from an old/current status table | AGENT-4/5 |
| Update and validate settings | Change two values in a synthetic config file and run its validation tool | AGENT-6/7 |

[Original P1](publication/originals/p1_review/REPORT.md) · [Preservation map](publication/provenance/p1_original_map.json)
