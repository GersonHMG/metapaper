# Experiment B1: Transition windows results

`AsymSETNetGroupedSegments`, 1 s segments, non-causal TCN (width 32), one prediction per segment. Transition dataset (`fold_windows`), 5 folds by recording, patients chb01, chb18, chb06, chb02, chb03, chb04, chb05, chb07, chb08, chb09, chb10, chb11, chb12, chb13, chb14, chb15, chb16, chb17, chb19, chb20, chb21, chb22, chb23, chb24. Boundary segments: within ±2 s of an onset or offset. See `experiments/experiment_b1_transition.md`.

**Pooled F1** is computed from TP/FP/FN summed over every patient and fold. It is the most reliable number for boundary segments, which are few per patient. Per-fold counts: `b1_transition_all_folds.csv`.

Boundary segments per patient (all test folds, one window length): chb01 56, chb18 48, chb06 80, chb02 24, chb03 56, chb04 32, chb05 40, chb07 24, chb08 40, chb09 32, chb10 56, chb11 24, chb12 216, chb13 80, chb14 64, chb15 160, chb16 64, chb17 24, chb19 24, chb20 64, chb21 32, chb22 24, chb23 56, chb24 128.

## Boundary segments

**Pooled F1, boundary segments. Best per row in bold.**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.557 | 0.571 | **0.578** | 0.569 |
| 5 s | 0.546 | 0.565 | 0.581 | **0.592** |
| 8 s | 0.562 | 0.554 | 0.549 | **0.579** |
| 10 s | 0.532 | 0.556 | 0.568 | **0.577** |

**Boundary F1, mean of per-patient F1 ± variance across patients**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.547 ± 0.012 | 0.565 ± 0.015 | **0.570** ± 0.009 | 0.553 ± 0.016 |
| 5 s | 0.542 ± 0.013 | 0.542 ± 0.019 | 0.558 ± 0.009 | **0.578** ± 0.010 |
| 8 s | 0.542 ± 0.017 | 0.534 ± 0.017 | 0.519 ± 0.019 | **0.567** ± 0.012 |
| 10 s | 0.529 ± 0.017 | 0.534 ± 0.018 | 0.537 ± 0.011 | **0.554** ± 0.010 |

## All segments

**Pooled F1, all test segments. Best per row in bold.**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.747 | 0.777 | **0.780** | 0.777 |
| 5 s | 0.754 | 0.773 | **0.792** | 0.788 |
| 8 s | 0.736 | 0.770 | 0.780 | **0.785** |
| 10 s | 0.735 | 0.745 | **0.790** | 0.787 |

## Paired comparison with 0 layers

Per-patient F1 (folds pooled): mean difference (positive = TCN better), patients where the TCN is better, and the two-sided Wilcoxon signed-rank p-value over patients (— with fewer than 5 patients). No correction for multiple comparisons.

**Boundary segments**

| Window | 1 layers − 0 | wins | p | 2 layers − 0 | wins | p | 3 layers − 0 | wins | p |
|---|---|---|---|---|---|---|---|---|---|
| 3 s | +0.017 | 15 / 24 | 0.201 | +0.023 | 15 / 24 | 0.132 | +0.005 | 11 / 24 | 0.664 |
| 5 s | -0.000 | 13 / 24 | 0.584 | +0.016 | 14 / 24 | 0.484 | +0.036 | 15 / 24 | 0.069 |
| 8 s | -0.008 | 11 / 24 | 0.627 | -0.023 | 8 / 24 | 0.189 | +0.025 | 9 / 24 | 0.465 |
| 10 s | +0.006 | 10 / 24 | 0.812 | +0.008 | 14 / 24 | 0.658 | +0.025 | 15 / 24 | 0.290 |

**All segments**

| Window | 1 layers − 0 | wins | p | 2 layers − 0 | wins | p | 3 layers − 0 | wins | p |
|---|---|---|---|---|---|---|---|---|---|
| 3 s | +0.034 | 20 / 24 | <0.001 | +0.035 | 20 / 24 | <0.001 | +0.035 | 18 / 24 | <0.001 |
| 5 s | +0.025 | 17 / 24 | 0.029 | +0.043 | 21 / 24 | <0.001 | +0.034 | 20 / 24 | <0.001 |
| 8 s | +0.039 | 19 / 24 | <0.001 | +0.049 | 20 / 24 | <0.001 | +0.053 | 21 / 24 | <0.001 |
| 10 s | +0.023 | 15 / 24 | 0.065 | +0.055 | 23 / 24 | <0.001 | +0.045 | 22 / 24 | <0.001 |

## Shuffle test

Pooled F1 with the segments of every test window in their original order → in a random order (drop). Each segment is scored against its own label. 0 layers must not change (control).

**Boundary segments**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.557 → 0.557 (+0.000) | 0.571 → 0.563 (+0.009) | 0.578 → 0.574 (+0.003) | 0.569 → 0.574 (-0.004) |
| 5 s | 0.546 → 0.546 (+0.000) | 0.565 → 0.557 (+0.008) | 0.581 → 0.561 (+0.020) | 0.592 → 0.570 (+0.022) |
| 8 s | 0.562 → 0.562 (+0.000) | 0.554 → 0.523 (+0.031) | 0.549 → 0.529 (+0.020) | 0.579 → 0.557 (+0.022) |
| 10 s | 0.532 → 0.532 (+0.000) | 0.556 → 0.533 (+0.023) | 0.568 → 0.552 (+0.016) | 0.577 → 0.544 (+0.033) |

**All segments**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.747 → 0.747 (+0.000) | 0.777 → 0.777 (-0.000) | 0.780 → 0.778 (+0.002) | 0.777 → 0.777 (-0.000) |
| 5 s | 0.754 → 0.754 (+0.000) | 0.773 → 0.772 (+0.001) | 0.792 → 0.791 (+0.001) | 0.788 → 0.788 (+0.000) |
| 8 s | 0.736 → 0.736 (+0.000) | 0.770 → 0.768 (+0.003) | 0.780 → 0.776 (+0.003) | 0.785 → 0.781 (+0.004) |
| 10 s | 0.735 → 0.735 (+0.000) | 0.745 → 0.742 (+0.003) | 0.790 → 0.781 (+0.010) | 0.787 → 0.775 (+0.011) |

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

**chb02: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.471 | **0.556** | 0.471 | 0.471 |
| 5 s | 0.471 | 0.471 | 0.526 | **0.600** |
| 8 s | 0.353 | **0.444** | 0.353 | 0.353 |
| 10 s | 0.500 | 0.286 | **0.526** | 0.471 |

**chb03: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.742 | 0.783 | 0.746 | **0.806** |
| 5 s | **0.787** | 0.746 | 0.696 | 0.735 |
| 8 s | **0.762** | 0.722 | 0.722 | 0.704 |
| 10 s | **0.787** | 0.727 | 0.714 | 0.722 |

**chb04: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.560 | 0.522 | 0.538 | **0.600** |
| 5 s | **0.560** | 0.545 | 0.435 | 0.538 |
| 8 s | 0.643 | **0.667** | 0.538 | 0.640 |
| 10 s | **0.615** | 0.583 | 0.500 | 0.593 |

**chb05: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.529 | **0.588** | **0.588** | 0.438 |
| 5 s | **0.632** | 0.556 | 0.564 | 0.611 |
| 8 s | **0.667** | 0.611 | 0.529 | 0.571 |
| 10 s | **0.667** | 0.588 | 0.485 | 0.471 |

**chb07: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | **0.727** | 0.696 | 0.636 | 0.692 |
| 5 s | 0.632 | **0.720** | 0.600 | 0.609 |
| 8 s | 0.667 | 0.696 | **0.741** | 0.667 |
| 10 s | **0.700** | 0.696 | 0.600 | 0.522 |

**chb08: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | **0.529** | 0.345 | 0.471 | 0.357 |
| 5 s | **0.606** | 0.286 | 0.424 | 0.400 |
| 8 s | **0.516** | 0.438 | 0.438 | 0.485 |
| 10 s | **0.606** | 0.457 | 0.471 | 0.438 |

**chb09: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.522 | **0.643** | 0.538 | 0.522 |
| 5 s | 0.522 | **0.643** | 0.560 | 0.545 |
| 8 s | 0.476 | **0.583** | 0.316 | 0.545 |
| 10 s | 0.476 | 0.545 | 0.500 | **0.583** |

**chb10: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.625 | 0.727 | **0.741** | 0.731 |
| 5 s | **0.667** | 0.619 | 0.553 | 0.549 |
| 8 s | 0.667 | **0.680** | 0.596 | 0.655 |
| 10 s | 0.524 | **0.667** | 0.604 | 0.549 |

**chb11: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.583 | **0.609** | 0.560 | 0.583 |
| 5 s | 0.609 | **0.667** | 0.609 | **0.667** |
| 8 s | 0.583 | 0.640 | **0.667** | **0.667** |
| 10 s | 0.636 | 0.609 | 0.609 | **0.667** |

**chb12: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.592 | 0.596 | **0.622** | 0.584 |
| 5 s | 0.586 | 0.679 | **0.689** | 0.676 |
| 8 s | 0.617 | **0.673** | 0.627 | 0.643 |
| 10 s | 0.549 | 0.657 | **0.661** | 0.637 |

**chb13: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.438 | **0.525** | 0.400 | 0.381 |
| 5 s | 0.429 | 0.286 | 0.369 | **0.532** |
| 8 s | **0.406** | 0.290 | 0.310 | 0.286 |
| 10 s | 0.328 | 0.271 | **0.345** | 0.333 |

**chb14: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.520 | 0.500 | **0.538** | 0.449 |
| 5 s | 0.431 | 0.520 | **0.526** | 0.480 |
| 8 s | 0.333 | 0.528 | **0.607** | 0.576 |
| 10 s | 0.375 | 0.531 | **0.632** | 0.536 |

**chb15: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | **0.590** | 0.528 | 0.549 | 0.561 |
| 5 s | 0.532 | 0.547 | **0.602** | 0.580 |
| 8 s | **0.584** | 0.520 | 0.535 | 0.568 |
| 10 s | 0.539 | 0.530 | 0.566 | **0.603** |

**chb16: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.233 | 0.261 | **0.400** | 0.304 |
| 5 s | 0.311 | 0.340 | 0.383 | **0.483** |
| 8 s | 0.244 | 0.348 | **0.383** | 0.377 |
| 10 s | 0.244 | 0.326 | 0.417 | **0.444** |

**chb17: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.421 | 0.421 | 0.500 | **0.609** |
| 5 s | 0.421 | 0.444 | 0.667 | **0.750** |
| 8 s | 0.500 | 0.353 | 0.222 | **0.667** |
| 10 s | 0.333 | 0.444 | 0.455 | **0.571** |

**chb19: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.476 | 0.417 | **0.519** | 0.348 |
| 5 s | 0.381 | 0.300 | **0.455** | 0.364 |
| 8 s | **0.500** | 0.300 | 0.364 | 0.480 |
| 10 s | 0.400 | 0.316 | 0.348 | **0.455** |

**chb20: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.562 | **0.696** | 0.685 | 0.639 |
| 5 s | 0.567 | 0.590 | **0.606** | 0.603 |
| 8 s | **0.636** | 0.603 | **0.636** | 0.585 |
| 10 s | 0.606 | 0.566 | **0.645** | 0.610 |

**chb21: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.611 | 0.632 | **0.667** | 0.650 |
| 5 s | **0.649** | 0.632 | 0.647 | 0.632 |
| 8 s | 0.606 | **0.629** | 0.571 | 0.556 |
| 10 s | 0.625 | **0.647** | 0.600 | 0.581 |

**chb22: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.571 | 0.609 | **0.636** | 0.500 |
| 5 s | 0.500 | **0.636** | 0.571 | 0.571 |
| 8 s | 0.421 | 0.421 | **0.636** | **0.636** |
| 10 s | 0.500 | **0.667** | 0.476 | 0.545 |

**chb23: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.610 | **0.644** | 0.536 | 0.561 |
| 5 s | 0.596 | 0.520 | 0.510 | **0.600** |
| 8 s | **0.586** | 0.571 | 0.490 | 0.519 |
| 10 s | **0.576** | 0.468 | 0.372 | 0.520 |

**chb24: boundary F1 (folds pooled)**

| Window | 0 layers | 1 layers | 2 layers | 3 layers |
|---|---|---|---|---|
| 3 s | 0.429 | 0.462 | 0.495 | **0.533** |
| 5 s | 0.370 | 0.471 | **0.542** | 0.515 |
| 8 s | 0.465 | 0.400 | 0.485 | **0.564** |
| 10 s | 0.404 | 0.485 | 0.528 | **0.569** |
