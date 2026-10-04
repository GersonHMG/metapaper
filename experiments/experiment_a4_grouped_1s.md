# Experiment A4: Grouped kernel at 1 s windows

Fills the missing cell in Table 3.1 (grouped kernel, 1 s window).

**Question:** What F1 does the grouped spatial module reach at 1 s windows, using the same model as the 3 s and 5 s cells?

## Why it is missing

The grouped cells at 3 s and 5 s come from `b_tcn` with 0 TCN layers: `AsymSETNetGroupedSegments`, 1 s segments, with the window probability being the mean of the segment probabilities. `b_tcn` only ran windows of 3, 5, 8 and 10 s, so there is no 1 s run.

The 1 s value in `a_grouped_all` (0.762) is not a substitute. It comes from `SpatialNet(spatial="grouped")`, which has a single Linear head and no segments.

## Model

- `AsymSETNetGroupedSegments(n_segments=1)`, with 1 s segments, so each window is a single segment.
- **TCN:** 0 layers.
- **Head:** Linear → ReLU → Linear per segment, the same as `b_tcn`.
- **Everything else:** identical to the 0-layer model in `b_tcn` (width, dropout, `TrainConfig` defaults).

This is the same configuration as the B3 encoder, but trained on `recording_cv` windows instead of continuous seconds.

## Data and protocol

Same as Experiment A, so the cell is comparable with the rest of Table 3.1:

- **Dataset:** CHB-MIT, `recording_cv`, 1 s windows.
- **Patients:** all 24.
- **Folds:** the same folds and seeds as `a_grouped_all` and `b_tcn`. That is 5 folds by recording, or leave-one-recording-out when a patient has fewer than 5 seizure recordings.
- **Training:** the remaining recordings are split 80/20 into train and validation, with validation used only for early stopping. Loss is focal loss.
- **Test set:** keeps its natural ≤ 1:5 seizure:normal ratio.
- No shuffle test (unlike `b_tcn`): a single segment cannot be reordered.

## Metrics

Save per fold: tp, tn, fp, fn, accuracy, sensitivity, specificity, F1, AUROC.

Report F1 as the mean of the per-patient fold means ± **standard deviation** across patients (not variance).

## Sanity check

Also run the 3 s window with this script for one or two patients (chb01 and chb18). The result should match `b_tcn` 0 layers within GPU noise (about ±0.02 F1). If it does, the new 1 s cell is comparable with the 3 s and 5 s cells.

## Run

```bash
python -m experiments.experiment_a4_grouped_1s --windows 3 --patients chb01 chb18   # sanity check
python -m experiments.experiment_a4_grouped_1s --windows 1 --patients all           # Table 3.1 cell
```

## Output

- `results/a4_grouped_1s/a4_grouped_1s_folds.csv`
- `results/a4_grouped_1s/results.md`

**Goes into:** Table 3.1, row "Kernel por grupo", column 1 s. Optionally also Table 3.2 (Acc, Sen, Spec, F1, AUROC at 1 s).
