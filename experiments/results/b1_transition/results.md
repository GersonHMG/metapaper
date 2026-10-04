# Experiment B1: Transition windows results

`AsymSETNetGroupedSegments`, 1 s segments, non-causal TCN (width 32), one prediction per segment. Transition dataset (`fold_windows`), 5 folds by recording, patients chb01, chb18, chb06. Boundary segments: within ±2 s of an onset or offset. See `experiments/experiment_b1_transition.md`.

**Pooled F1** is computed from TP/FP/FN summed over every patient and fold. It is the most reliable number for boundary segments, which are few per patient. Per-fold counts: `b1_transition_folds.csv`.

Boundary segments per patient (all test folds, one window length): chb01 56, chb18 48, chb06 80.

## Boundary segments

**Pooled F1, boundary segments. Best per row in bold.**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.606 | 0.596 | 0.614 | **0.651** |
| 5 s | 0.590 | 0.614 | 0.624 | **0.630** |
| 8 s | 0.614 | 0.573 | 0.571 | **0.623** |
| 10 s | 0.577 | 0.603 | 0.622 | **0.650** |

**Boundary F1, mean of per-patient F1 ± variance across patients**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.598 ± 0.007 | 0.598 ± 0.002 | 0.617 ± 0.007 | **0.647** ± 0.003 |
| 5 s | 0.584 ± 0.004 | 0.596 ± 0.014 | **0.616** ± 0.007 | 0.609 ± 0.019 |
| 8 s | 0.594 ± 0.018 | 0.567 ± 0.005 | 0.562 ± 0.011 | **0.625** ± 0.005 |
| 10 s | 0.567 ± 0.005 | 0.587 ± 0.017 | 0.614 ± 0.008 | **0.624** ± 0.029 |

## All segments

**Pooled F1, all test segments. Best per row in bold.**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.735 | 0.768 | 0.794 | **0.795** |
| 5 s | 0.762 | 0.775 | **0.836** | 0.809 |
| 8 s | 0.739 | 0.790 | 0.806 | **0.811** |
| 10 s | 0.736 | 0.807 | **0.827** | 0.810 |

## Paired comparison with 0 layers

Per-patient F1 (folds pooled): mean difference (positive = TCN better), patients where the TCN is better, and the two-sided Wilcoxon signed-rank p-value over patients (— with fewer than 5 patients). No correction for multiple comparisons.

**Boundary segments**

| Window | 1 layers − 0 | wins | p | 2 layers − 0 | wins | p | 3 layers − 0 | wins | p |
|---|---|---|---|---|---|---|---|---|---|
| 3 s | +0.001 | 1 / 3 | — | +0.019 | 2 / 3 | — | +0.049 | 3 / 3 | — |
| 5 s | +0.012 | 2 / 3 | — | +0.032 | 3 / 3 | — | +0.025 | 2 / 3 | — |
| 8 s | -0.026 | 1 / 3 | — | -0.032 | 1 / 3 | — | +0.031 | 1 / 3 | — |
| 10 s | +0.020 | 1 / 3 | — | +0.046 | 3 / 3 | — | +0.057 | 2 / 3 | — |

**All segments**

| Window | 1 layers − 0 | wins | p | 2 layers − 0 | wins | p | 3 layers − 0 | wins | p |
|---|---|---|---|---|---|---|---|---|---|
| 3 s | +0.021 | 2 / 3 | — | +0.049 | 3 / 3 | — | +0.050 | 2 / 3 | — |
| 5 s | +0.006 | 1 / 3 | — | +0.072 | 3 / 3 | — | +0.034 | 2 / 3 | — |
| 8 s | +0.039 | 2 / 3 | — | +0.053 | 2 / 3 | — | +0.061 | 2 / 3 | — |
| 10 s | +0.055 | 2 / 3 | — | +0.079 | 3 / 3 | — | +0.059 | 3 / 3 | — |

## Shuffle test

Pooled F1 with the segments of every test window in their original order → in a random order (drop). Each segment is scored against its own label. 0 layers must not change (control).

**Boundary segments**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.606 → 0.606 (+0.000) | 0.596 → 0.610 (-0.013) | 0.614 → 0.612 (+0.003) | 0.651 → 0.643 (+0.008) |
| 5 s | 0.590 → 0.590 (+0.000) | 0.614 → 0.578 (+0.036) | 0.624 → 0.568 (+0.056) | 0.630 → 0.626 (+0.005) |
| 8 s | 0.614 → 0.614 (+0.000) | 0.573 → 0.481 (+0.093) | 0.571 → 0.490 (+0.081) | 0.623 → 0.562 (+0.060) |
| 10 s | 0.577 → 0.577 (+0.000) | 0.603 → 0.555 (+0.048) | 0.622 → 0.559 (+0.063) | 0.650 → 0.562 (+0.087) |

**All segments**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.735 → 0.735 (+0.000) | 0.768 → 0.771 (-0.002) | 0.794 → 0.794 (-0.000) | 0.795 → 0.796 (-0.001) |
| 5 s | 0.762 → 0.762 (+0.000) | 0.775 → 0.773 (+0.001) | 0.836 → 0.832 (+0.004) | 0.809 → 0.805 (+0.003) |
| 8 s | 0.739 → 0.739 (+0.000) | 0.790 → 0.781 (+0.008) | 0.806 → 0.782 (+0.023) | 0.811 → 0.800 (+0.010) |
| 10 s | 0.736 → 0.736 (+0.000) | 0.807 → 0.790 (+0.016) | 0.827 → 0.802 (+0.025) | 0.810 → 0.797 (+0.014) |

## Per patient

**chb01: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.689 | 0.633 | 0.700 | **0.710** |
| 5 s | 0.656 | 0.678 | 0.698 | **0.730** |
| 8 s | **0.719** | 0.645 | 0.679 | 0.700 |
| 10 s | 0.633 | 0.724 | 0.702 | **0.764** |

**chb18: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.524 | **0.619** | **0.619** | 0.605 |
| 5 s | 0.524 | 0.462 | **0.533** | 0.462 |
| 8 s | 0.450 | 0.524 | 0.474 | **0.615** |
| 10 s | 0.488 | 0.462 | **0.524** | 0.432 |

**chb06: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.581 | 0.542 | 0.531 | **0.627** |
| 5 s | 0.571 | **0.647** | 0.615 | 0.635 |
| 8 s | **0.613** | 0.533 | 0.533 | 0.559 |
| 10 s | 0.581 | 0.576 | 0.615 | **0.677** |
