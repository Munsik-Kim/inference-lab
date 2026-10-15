# Case014 — Quantization Compensation: Cancellation, Re-quantization, and Generalization

[한국어](README.ko.md)

Quantization stores model numbers using fewer bits. Compensation adjusts later weights to reduce error inherited from earlier layers. This tool checks what survives after those adjusted weights are stored as INT4 again, and whether the benefit transfers to other documents.

Executed correction fitting, INT4 save/reload, error decomposition and common-input full-model comparison. CAL-selected strengths reduced local error on CAL, but increased aggregate local error on new EVAL documents at all six targets. Next-token scores moved differently across targets.

**Executed:**30/30 core settings,2,880 raw rows and32 original EVAL books. CAL-selected EVAL NLL improved at3 targets and worsened at3; all6 SHIFT mean NLL estimates increased. Lower ΔNLL is better.

|Block / projection|CAL α|CAL ΔSSE %|EVAL ΔSSE %|EVAL ΔNLL [95% CI]|SHIFT ΔNLL [95% CI]|
|---|---:|---:|---:|---|---|
|4 / down_proj|1|-53.34|+45.40|+0.017570 [+0.009267, +0.025416]|+0.047380 [+0.028125, +0.070095]|
|4 / o_proj|1|-52.06|+10.98|-0.006263 [-0.012331, -0.000537]|+0.010130 [+0.002822, +0.018001]|
|12 / down_proj|1|-55.72|+36.68|+0.011532 [+0.006298, +0.017117]|+0.016129 [+0.005145, +0.027515]|
|12 / o_proj|1|-55.25|+43.26|-0.004538 [-0.009533, +0.000430]|+0.005138 [-0.001055, +0.011251]|
|20 / down_proj|0.75|-45.33|+14.29|+0.003528 [-0.000984, +0.008394]|+0.056429 [+0.043199, +0.071002]|
|20 / o_proj|1|-66.15|+5.11|-0.007653 [-0.011391, -0.003748]|+0.023418 [+0.014054, +0.033364]|

[결과 보기 / Results](demo/p1_r2/en/index.html) · [상세 보고서 / Report](REPORT.md) · [CPU 검산 / Reproduce](REPRODUCE.md) · [실행 코드 / Runner](scripts/run_p1.py)

These results use INT4 storage and dense BF16 quality execution. P2/P3 and remote publication were not run. [Historical P0 report](publication/original_pre_p1/REPORT.md) is preserved.