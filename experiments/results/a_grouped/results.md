# Experiment A: Grouped kernel results

`SpatialNet(spatial="grouped")`, recording CV (5 folds), patients chb01, chb18, chb06. See `experiments/experiment_a.md`.

Cells are F1 mean ± variance. Per-fold rows with all metrics: `a_grouped_folds.csv`.

## Mean over patients

**All patients (mean of the per-patient fold means ± variance across patients)**

| Window | grouped |
|---|---|
| 1 s | 0.759 ± 0.009 |
| 3 s | 0.815 ± 0.010 |
| 5 s | 0.801 ± 0.026 |
| 10 s | 0.859 ± 0.012 |

## Per patient

**Mean ± variance over folds**

| Patient | 1 s | 3 s | 5 s | 10 s |
|---|---|---|---|---|
| chb01 | 0.871 ± 0.004 | 0.913 ± 0.006 | 0.941 ± 0.000 | 0.970 ± 0.001 |
| chb18 | 0.712 ± 0.058 | 0.815 ± 0.055 | 0.838 ± 0.039 | 0.752 ± 0.034 |
| chb06 | 0.694 ± 0.044 | 0.716 ± 0.056 | 0.624 ± 0.049 | 0.856 ± 0.063 |
