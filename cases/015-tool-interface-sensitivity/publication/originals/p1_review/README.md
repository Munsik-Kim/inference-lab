# Case015 — Same Tools, Different Interfaces

Implemented a local comparison tool that changes the names and ordering of the same tools, then checks whether the agent actually completes the task. It connects exact interface mappings, stepwise equivalence, real fixture execution and saved-response replay.

Completed 96 FREE episodes. D16 recovered two patch failures under opaque names, while lowerCamelCase introduced a new loss. The primary signed-interaction interval includes zero.

[한국어](README.ko.md) · [ Results and episodes ](demo/en.html) · [ Report ](REPORT.md) · [ CPU reproduction ](REPRODUCE.md) · [Code](src/interface_sensitivity/cli.py)

| Method | I0 original | I1 order | I2 lowerCamelCase | I3 opaque |
|---|---:|---:|---:|---:|
| BF16 | 8/8 | 8/8 | 8/8 | 7/8 |
| RTN | 8/8 | 8/8 | 8/8 | 8/8 |
| D16 | 6/8 | 6/8 | 5/8 | 8/8 |


Each cell is functional success on the same eight existing DEV tasks;96 is not an independent task count. ENUM, enhanced feedback and aliases are off.

- Exact name/order transformations and semantic-ID mappings: [registry](src/interface_sensitivity/registry.py), [transform](src/interface_sensitivity/transform.py).
- Per-step actions, arguments, actual state and validation: [runner](src/interface_sensitivity/runner.py), [scoring](src/interface_sensitivity/scoring.py).
- Model-free response/action replay and paired-effect analysis: [audit](src/interface_sensitivity/audit.py), [analysis](src/interface_sensitivity/analysis.py).

P0 passed 27 unique CPU tests and 32 oracle episodes. GPU generation requests: 228 primary plus 4 separate smoke. Worker wall including model load/decode: 452.013 s. Execution uses saved INT4 with dense BF16 reload; these are not serving-speed measurements.

[Contract](specs/protocol.json) · [Raw records](results/p1_r1/episodes.jsonl) · [CPU replay audit](results/p1_r1/independent_audit.json) · [Decision](results/p1_r1/decision.json) · [Roadmap](ROADMAP.md)
