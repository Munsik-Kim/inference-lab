# Representative approximation errors

These cases were selected from the completed fixed evaluation by the rules in `analyze.py`. The inspection reuses stored Q/K/V and the frozen arithmetic; it does not run the model again or change parameters.

## efq_mean: eval-13, length 512, layer 27, head 0

Selection: largest_absolute_relative_output_error. Unit error 0.35848818; same-scale nearest 0.16557932. Largest row error at query 170: 0.51704675. See [input document](../inputs/eval-13.txt).

| Key | Reference probability | Method probability | Nearest same-scale | Method code | Nearest code | Scale exponent |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.91572803 | 0.87041295 | 0.90639520 | 7 | 7 | -3 |
| 170 | 0.06970036 | 0.10880162 | 0.07553294 | 7 | 6 | -6 |
| 169 | 0.00439866 | 0.00906680 | 0.00944162 | 1 | 1 | -6 |
| 165 | 0.00076512 | 0.00000000 | 0.00000000 | 0 | 0 | -6 |
| 159 | 0.00072007 | 0.00085001 | 0.00088515 | 7 | 7 | -13 |
| 133 | 0.00062353 | 0.00085001 | 0.00088515 | 7 | 7 | -13 |
| 168 | 0.00050330 | 0.00000000 | 0.00000000 | 0 | 0 | -6 |
| 134 | 0.00048352 | 0.00085001 | 0.00059010 | 7 | 6 | -13 |
| 166 | 0.00040813 | 0.00000000 | 0.00000000 | 0 | 0 | -6 |
| 160 | 0.00037517 | 0.00000000 | 0.00000000 | 0 | 0 | -6 |
| 158 | 0.00034350 | 0.00056668 | 0.00044258 | 6 | 5 | -13 |
| 164 | 0.00029319 | 0.00000000 | 0.00000000 | 0 | 0 | -6 |
| 167 | 0.00023820 | 0.00000000 | 0.00000000 | 0 | 0 | -6 |
| 163 | 0.00022305 | 0.00000000 | 0.00000000 | 0 | 0 | -6 |
| 153 | 0.00021865 | 0.00042501 | 0.00029505 | 5 | 4 | -13 |
| 141 | 0.00021399 | 0.00042501 | 0.00029505 | 5 | 4 | -13 |

Removed reference mass: 0.00355860; score spread: 15.56761; entropy: 0.37750; maximum finite row-max increase: 0.00000. Reference output cancellation ratio: 0.62873. These are descriptive diagnostics, not additional pass/fail criteria.

## efq_calibrated: eval-15, length 512, layer 13, head 5

Selection: largest_absolute_relative_output_error. Unit error 0.18875005; same-scale nearest 0.17932130. Largest row error at query 34: 0.40496120. See [input document](../inputs/eval-15.txt).

| Key | Reference probability | Method probability | Nearest same-scale | Method code | Nearest code | Scale exponent |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.30976444 | 0.27428570 | 0.28235295 | 7 | 7 | -3 |
| 30 | 0.18922868 | 0.18285714 | 0.18823530 | 6 | 6 | -3 |
| 33 | 0.08257736 | 0.06857143 | 0.07058824 | 7 | 7 | -5 |
| 34 | 0.06780505 | 0.06857143 | 0.07058824 | 7 | 7 | -5 |
| 26 | 0.06334928 | 0.09142857 | 0.07058824 | 4 | 3 | -3 |
| 13 | 0.06052150 | 0.09142857 | 0.07058824 | 4 | 3 | -3 |
| 29 | 0.05500444 | 0.06857143 | 0.07058824 | 3 | 3 | -3 |
| 28 | 0.03076132 | 0.04571429 | 0.04705882 | 2 | 2 | -3 |
| 24 | 0.02356924 | 0.04571429 | 0.02352941 | 2 | 1 | -3 |
| 11 | 0.01360853 | 0.02285714 | 0.02352941 | 1 | 1 | -3 |
| 31 | 0.01336970 | 0.02285714 | 0.02352941 | 1 | 1 | -3 |
| 12 | 0.01204950 | 0.00000000 | 0.02352941 | 0 | 1 | -3 |
| 32 | 0.01007142 | 0.01714286 | 0.01176471 | 3 | 2 | -5 |
| 14 | 0.00990024 | 0.00000000 | 0.02352941 | 0 | 1 | -3 |
| 17 | 0.00760654 | 0.00000000 | 0.00000000 | 0 | 0 | -3 |
| 3 | 0.00518043 | 0.00000000 | 0.00000000 | 0 | 0 | -3 |

Removed reference mass: 0.08036902; score spread: 7.61966; entropy: 2.35523; maximum finite row-max increase: 0.00000. Reference output cancellation ratio: 0.47877. These are descriptive diagnostics, not additional pass/fail criteria.

## efq_calibrated: eval-15, length 2048, layer 0, head 5

Selection: largest_calibrated_to_same_scale_nearest_ratio. Unit error 0.02874992; same-scale nearest 0.00888626. Largest row error at query 1091: 0.14303030. See [input document](../inputs/eval-15.txt).

| Key | Reference probability | Method probability | Nearest same-scale | Method code | Nearest code | Scale exponent |
|---|---:|---:|---:|---:|---:|---:|
| 1091 | 0.89776677 | 0.82808071 | 0.88956726 | 7 | 7 | -3 |
| 1090 | 0.07046168 | 0.13801345 | 0.07413060 | 2 | 1 | -3 |
| 1009 | 0.00411934 | 0.00379959 | 0.00408172 | 7 | 7 | -3 |
| 1089 | 0.00113695 | 0.00000000 | 0.00000000 | 0 | 0 | -3 |
| 1086 | 0.00107066 | 0.00107823 | 0.00115829 | 6 | 6 | -12 |
| 1087 | 0.00104336 | 0.00107823 | 0.00115829 | 6 | 6 | -12 |
| 350 | 0.00083955 | 0.00077438 | 0.00083188 | 7 | 7 | -3 |
| 1085 | 0.00075281 | 0.00080867 | 0.00086872 | 5 | 5 | -12 |
| 1082 | 0.00063818 | 0.00080867 | 0.00086872 | 5 | 5 | -12 |
| 1056 | 0.00060833 | 0.00080867 | 0.00086872 | 5 | 5 | -12 |
| 1073 | 0.00057958 | 0.00080867 | 0.00086872 | 5 | 5 | -12 |
| 1079 | 0.00043928 | 0.00053912 | 0.00057915 | 4 | 4 | -12 |
| 997 | 0.00041756 | 0.00063326 | 0.00068029 | 2 | 2 | -3 |
| 991 | 0.00038427 | 0.00031663 | 0.00051021 | 6 | 7 | -6 |
| 1066 | 0.00037411 | 0.00053912 | 0.00043436 | 4 | 3 | -12 |
| 1080 | 0.00032648 | 0.00053912 | 0.00043436 | 4 | 3 | -12 |

Removed reference mass: 0.00291461; score spread: 29.50253; entropy: 0.55180; maximum finite row-max increase: 5.38422. Reference output cancellation ratio: 0.87060. These are descriptive diagnostics, not additional pass/fail criteria.
