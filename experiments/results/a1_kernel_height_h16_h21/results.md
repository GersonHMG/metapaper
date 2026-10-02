# Experiment A1: Kernel height results

`SpatialNet(spatial="height", kernel_height=h)`, recording CV (5 folds), patients chb01, chb18, chb06. See `experiments/experiment_a1_kernel_height.md`.

Cells are F1 mean ± variance. Best mean per window in bold. Per-fold rows with all metrics: `a1_kernel_height_h16_h21_folds.csv`.

## Mean over patients

**All patients (mean of the per-patient fold means ± variance across patients)**

| Window | h=16 | h=21 |
|---|---|---|
| 1 s | **0.748** ± 0.014 | 0.729 ± 0.011 |
| 3 s | **0.804** ± 0.012 | 0.795 ± 0.012 |
| 5 s | 0.766 ± 0.041 | **0.783** ± 0.024 |
| 8 s | 0.727 ± 0.044 | **0.825** ± 0.010 |
| 10 s | 0.657 ± 0.161 | **0.878** ± 0.010 |

## Per patient

**chb01 (mean ± variance over folds)**

| Window | h=16 | h=21 |
|---|---|---|
| 1 s | **0.873** ± 0.002 | 0.813 ± 0.004 |
| 3 s | **0.915** ± 0.001 | 0.889 ± 0.001 |
| 5 s | **0.940** ± 0.001 | 0.922 ± 0.000 |
| 8 s | **0.955** ± 0.003 | 0.916 ± 0.009 |
| 10 s | 0.955 ± 0.003 | **0.988** ± 0.001 |

**chb18 (mean ± variance over folds)**

| Window | h=16 | h=21 |
|---|---|---|
| 1 s | 0.730 ± 0.049 | **0.763** ± 0.040 |
| 3 s | 0.798 ± 0.038 | **0.823** ± 0.043 |
| 5 s | 0.813 ± 0.063 | **0.814** ± 0.036 |
| 8 s | 0.686 ± 0.038 | **0.839** ± 0.039 |
| 10 s | **0.815** ± 0.034 | 0.800 ± 0.036 |

**chb06 (mean ± variance over folds)**

| Window | h=16 | h=21 |
|---|---|---|
| 1 s | **0.640** ± 0.029 | 0.612 ± 0.035 |
| 3 s | **0.699** ± 0.055 | 0.673 ± 0.054 |
| 5 s | 0.543 ± 0.023 | **0.613** ± 0.052 |
| 8 s | 0.540 ± 0.132 | **0.720** ± 0.081 |
| 10 s | 0.200 ± 0.200 | **0.844** ± 0.121 |
