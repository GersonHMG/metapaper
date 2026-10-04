# Experiment A4: Grouped kernel at 1 s windows

`AsymSETNetGroupedSegments`, 1 s segments, 0 TCN layers, Linear-ReLU-Linear head; window probability = mean of the segment probabilities (a single segment at 1 s). Same model, folds, seeds and `TrainConfig` as `b_tcn` 0 layers. Recording CV (5 folds, leave-one-recording-out when a patient has fewer recordings). See `experiments/experiment_a4_grouped_1s.md`.

Cells are the mean of the per-patient fold means ± **standard deviation** across patients. Per-fold rows: `a4_grouped_1s_folds.csv`.

## Table 3.1 / 3.2 values

| Window | Patients | Accuracy | Sensitivity | Specificity | F1 | AUROC |
|---|---|---|---|---|---|---|
| 1 s | 24 | 0.900 ± 0.074 | 0.835 ± 0.090 | 0.913 ± 0.075 | 0.762 ± 0.124 | 0.928 ± 0.089 |
| 3 s | 2 | 0.970 ± 0.017 | 0.886 ± 0.106 | 0.986 ± 0.001 | 0.885 ± 0.087 | 0.965 ± 0.048 |

## Comparison with the existing grouped cells (F1, same patients)

| Window | Patients | A4 (this run) | `b_tcn` 0 layers | `SpatialNet(grouped)` (`a_grouped_all`) |
|---|---|---|---|---|
| 1 s | 24 | 0.762 ± 0.124 | — | 0.762 ± 0.120 |
| 3 s | 2 | 0.885 ± 0.087 | 0.885 ± 0.072 | 0.863 ± 0.074 |

## Sanity check

GPU training is not bit-reproducible: expect about ±0.02 F1.

**3 s window: A4 vs. `b_tcn` 0 layers, F1 per patient (fold mean)**

| Patient | A4 | `b_tcn` | Difference |
|---|---|---|---|
| chb01 | 0.946 | 0.936 | +0.011 |
| chb18 | 0.823 | 0.834 | -0.010 |

Identical test recordings in every fold: **True**.

## Per patient (F1, fold mean)

| Patient | 1 s | 3 s |
|---|---|---|
| chb01 | 0.863 | 0.946 |
| chb02 | 0.731 | — |
| chb03 | 0.914 | — |
| chb04 | 0.448 | — |
| chb05 | 0.863 | — |
| chb06 | 0.688 | — |
| chb07 | 0.795 | — |
| chb08 | 0.756 | — |
| chb09 | 0.745 | — |
| chb10 | 0.818 | — |
| chb11 | 0.826 | — |
| chb12 | 0.746 | — |
| chb13 | 0.707 | — |
| chb14 | 0.656 | — |
| chb15 | 0.902 | — |
| chb16 | 0.413 | — |
| chb17 | 0.809 | — |
| chb18 | 0.714 | 0.823 |
| chb19 | 0.779 | — |
| chb20 | 0.844 | — |
| chb21 | 0.770 | — |
| chb22 | 0.871 | — |
| chb23 | 0.886 | — |
| chb24 | 0.735 | — |
