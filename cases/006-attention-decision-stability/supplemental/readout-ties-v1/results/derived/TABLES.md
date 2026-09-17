# Result tables

POST-HOC SAME-INPUT READOUT DIAGNOSTIC. Original native results remain primary.

| Set | Readout | Arm | B correct | Candidate correct | Flips | Regression | Gain | Wrong-to-wrong | B / candidate top ties |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| standard | H_NATIVE | A_PUBLIC | 104/192 | 106/192 | 5 | 0 | 2 | 3 | 14 / 12 |
| standard | H_NATIVE | V4 | 104/192 | 108/192 | 8 | 0 | 4 | 4 | 14 / 9 |
| standard | H_FP32 | A_PUBLIC | 104/192 | 105/192 | 3 | 1 | 2 | 0 | 0 / 0 |
| standard | H_FP32 | V4 | 104/192 | 106/192 | 3 | 0 | 2 | 1 | 0 / 0 |
| boundary_pool | H_NATIVE | A_PUBLIC | 18/46 | 17/46 | 3 | 2 | 1 | 0 | 0 / 3 |
| boundary_pool | H_NATIVE | V4 | 18/46 | 19/46 | 3 | 1 | 2 | 0 | 0 / 4 |
| boundary_pool | H_FP32 | A_PUBLIC | 18/46 | 17/46 | 1 | 1 | 0 | 0 | 0 / 0 |
| boundary_pool | H_FP32 | V4 | 18/46 | 17/46 | 3 | 2 | 1 | 0 | 0 / 0 |

## Native tie-state partitions

Each cell reports scenario / flip / regression / gain / wrong-to-wrong counts. The four cells exhaust each paired comparison.

| Set / arm | Unique → unique | Tied → unique | Unique → tied | Both tied |
|---|---|---|---|---|
| standard / A_PUBLIC | 173 / 0 / 0 / 0 / 0 | 7 / 2 / 0 / 1 / 1 | 5 / 3 / 0 / 1 / 2 | 7 / 0 / 0 / 0 / 0 |
| standard / V4 | 174 / 0 / 0 / 0 / 0 | 9 / 5 / 0 / 2 / 3 | 4 / 3 / 0 / 2 / 1 | 5 / 0 / 0 / 0 / 0 |
| boundary_pool / A_PUBLIC | 43 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 3 / 3 / 2 / 1 / 0 | 0 / 0 / 0 / 0 / 0 |
| boundary_pool / V4 | 42 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 4 / 3 / 1 / 2 / 0 | 0 / 0 / 0 / 0 / 0 |

## Task-specific counts and scores

| Set / task | Readout / candidate | n | B correct | Candidate correct | Flips / regressions / gains | Mean Δ gold NLL (nats) | Mean Δ Brier | Mean Δ gold margin |
|---|---|---:|---:|---:|---|---:|---:|---:|
| standard / RETRIEVAL | H_NATIVE / A_PUBLIC | 64 | 59 | 59 | 0 / 0 / 0 | +0.00887064 | +0.00508331 | -0.03125000 |
| standard / RETRIEVAL | H_NATIVE / V4 | 64 | 59 | 59 | 0 / 0 / 0 | -0.00222008 | +0.00318091 | -0.02929688 |
| standard / RETRIEVAL | H_FP32 / A_PUBLIC | 64 | 59 | 59 | 0 / 0 / 0 | -0.00177429 | +0.00113410 | -0.01840332 |
| standard / RETRIEVAL | H_FP32 / V4 | 64 | 59 | 59 | 0 / 0 / 0 | -0.00633524 | +0.00066257 | -0.00239593 |
| standard / COMPARISON | H_NATIVE / A_PUBLIC | 64 | 30 | 31 | 2 / 0 / 1 | -0.00625716 | -0.00340585 | +0.00195312 |
| standard / COMPARISON | H_NATIVE / V4 | 64 | 30 | 32 | 4 / 0 / 2 | -0.02767042 | -0.01282879 | +0.03515625 |
| standard / COMPARISON | H_FP32 / A_PUBLIC | 64 | 30 | 30 | 0 / 0 / 0 | -0.01598925 | -0.00866496 | +0.02591494 |
| standard / COMPARISON | H_FP32 / V4 | 64 | 30 | 30 | 0 / 0 / 0 | -0.02725499 | -0.01312094 | +0.03858253 |
| standard / CODE | H_NATIVE / A_PUBLIC | 64 | 15 | 16 | 3 / 0 / 1 | +0.00066763 | +0.00116160 | -0.00195312 |
| standard / CODE | H_NATIVE / V4 | 64 | 15 | 17 | 4 / 0 / 2 | +0.00834396 | +0.00055636 | -0.01367188 |
| standard / CODE | H_FP32 / A_PUBLIC | 64 | 15 | 16 | 3 / 1 / 2 | -0.00534016 | -0.00465777 | +0.01421800 |
| standard / CODE | H_FP32 / V4 | 64 | 15 | 17 | 3 / 0 / 2 | -0.00085099 | -0.00667193 | +0.01002526 |
| boundary_pool / RETRIEVAL | H_NATIVE / A_PUBLIC | 10 | 8 | 8 | 0 / 0 / 0 | +0.04774010 | +0.02762736 | -0.05000000 |
| boundary_pool / RETRIEVAL | H_NATIVE / V4 | 10 | 8 | 8 | 0 / 0 / 0 | +0.01931867 | +0.01547817 | -0.08750000 |
| boundary_pool / RETRIEVAL | H_FP32 / A_PUBLIC | 10 | 8 | 8 | 0 / 0 / 0 | +0.01559037 | +0.01235371 | -0.00506248 |
| boundary_pool / RETRIEVAL | H_FP32 / V4 | 10 | 8 | 8 | 0 / 0 / 0 | +0.00693471 | +0.00891285 | -0.06447411 |
| boundary_pool / COMPARISON | H_NATIVE / A_PUBLIC | 18 | 6 | 7 | 1 / 0 / 1 | +0.00671891 | -0.00047081 | -0.00694444 |
| boundary_pool / COMPARISON | H_NATIVE / V4 | 18 | 6 | 8 | 2 / 0 / 2 | -0.00074810 | -0.00688352 | +0.01388889 |
| boundary_pool / COMPARISON | H_FP32 / A_PUBLIC | 18 | 6 | 6 | 0 / 0 / 0 | +0.00893259 | +0.00062120 | -0.02120294 |
| boundary_pool / COMPARISON | H_FP32 / V4 | 18 | 6 | 6 | 2 / 1 / 1 | +0.00821302 | +0.00207163 | -0.01265494 |
| boundary_pool / CODE | H_NATIVE / A_PUBLIC | 18 | 4 | 2 | 2 / 2 / 0 | +0.04730083 | +0.00994212 | -0.07638889 |
| boundary_pool / CODE | H_NATIVE / V4 | 18 | 4 | 3 | 1 / 1 / 0 | +0.07663202 | +0.01561728 | -0.10416667 |
| boundary_pool / CODE | H_FP32 / A_PUBLIC | 18 | 4 | 3 | 1 / 1 / 0 | +0.06628232 | +0.01798090 | -0.08926201 |
| boundary_pool / CODE | H_FP32 / V4 | 18 | 4 | 3 | 1 / 1 / 0 | +0.08251674 | +0.02078028 | -0.10901070 |

## STANDARD task-balanced score changes

5000 paired within-task scenario-cluster bootstrap draws; post-hoc pointwise descriptive intervals.

| Readout | Arm | Δ gold NLL [95% interval] | Δ Brier [95% interval] |
|---|---|---|---|
| H_NATIVE | A_PUBLIC | +0.00109370 [-0.01200852, +0.01430667] | +0.00094635 [-0.00644409, +0.00818496] |
| H_NATIVE | V4 | -0.00718218 [-0.01916248, +0.00462592] | -0.00303051 [-0.00917805, +0.00317810] |
| H_FP32 | A_PUBLIC | -0.00770123 [-0.01735549, +0.00242349] | -0.00406287 [-0.00911038, +0.00086250] |
| H_FP32 | V4 | -0.01148041 [-0.02073442, -0.00247898] | -0.00637677 [-0.01083025, -0.00197389] |
