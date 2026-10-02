# Experiment A: Spatial module

**Question:** Which spatial module configuration works best for AsymSETNet?

**Goal:** Refine the AsymSETNet spatial module, tested on its own (no temporal layer).

## Models

Both use `SpatialNet` (`models/spatial_net.py`): spatial module → global average pooling → linear.
The model has no TCN and no `n_segments`, so it works with every window length.

| Model | Config | Spatial kernel |
|---|---|---|
| Grouped kernel | `SpatialNet(spatial="grouped")` | One kernel per bipolar electrode chain (`ELECTRODE_GROUPS`) |
| Kernel height | `SpatialNet(spatial="height", kernel_height=h)` | One kernel of height `h` sliding over all electrodes, `h` ∈ {1, 4, 8, 16, 21} |

Choosing the best `h` is a separate sub-experiment: [experiment_a1_kernel_height.md](experiment_a1_kernel_height.md).

## Data and protocol

- **Dataset:** CHB-MIT, `recording_cv` (`datasets/recording_cv.py`). Windows: 1, 3, 5, 10 s.
- **Patient dependent:** each patient is trained and tested on their own data.
- **Splits by recording:** a recording is never in both train and test.
  - 5 folds when the patient has ≥ 5 seizure recordings.
  - Otherwise leave-one-recording-out.
- **Training:** remaining recordings split 80/20 into train/validation (validation used only for early stopping), Focal Loss, `TrainConfig` defaults (`experiments/training.py`).
- **Test set** keeps its natural ≤ 1:5 seizure:normal ratio.
- **Exclusion:** chb16 at 10 s (only 1 usable recording).

## Metrics

Save these per fold: tp, tn, fp, fn, accuracy, sensitivity, specificity, F1, AUROC.

## Report

Best kernel height model vs. grouped kernel, F1 per window length:

| Patient | Model | 1 s | 3 s | 5 s | 10 s |
|---|---|---|---|---|---|
| chb01 | Kernel height (best `h`) | | | | |
| chb01 | Grouped kernel | | | | |
| … | | | | | |
| **Mean ± std** | Kernel height (best `h`) | | | | |
| **Mean ± std** | Grouped kernel | | | | |
