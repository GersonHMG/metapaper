# Experiment B: Temporal layer

**Question:** Is the temporal layer (TCN) worth adding to AsymSETNet?

**Sub-questions:**
- Does the window length change how much the TCN helps?
- How many TCN layers are needed?
- Does the segment length (1 s vs. 2 s) matter?

## Model

`AsymSETNetGroupedSegments` (`models/asymsetnet_grouped_segments.py`):

- The window is cut into segments, and the grouped spatial module (the winner of Experiment A) runs on each segment.
- A non-causal TCN runs over the segments (`causal=False`), with 0, 1, 2 or 3 layers: `tcn_channels` = `()`, `(32,)`, `(32, 32)` or `(32, 32, 32)`.
- 0 layers is the baseline: each segment is classified from its own spatial features only.
- **Head:** Linear → ReLU → Linear (hidden 64), the same as `SpatialNet` and shared by every segment.
- **Each segment is classified:** the output is one logit per segment, shape (B, N).
  - Training: `recording_cv` windows never cross a seizure boundary, so every segment gets the window's label in the focal loss.
  - Evaluation: the window probability is the mean of its segment probabilities (`model.predict_proba`). Window-level metrics are therefore comparable with Experiment A.

**Reference (no new runs):** `SpatialNet(spatial="grouped")` from Experiment A (`results/a_grouped_all/`), which processes the whole window without segments.

## Windows and segments

`n_segments` = window ÷ segment length.

| Segment | Windows | n_segments |
|---|---|---|
| 1 s | 3, 5, 8, 10 s | 3, 5, 8, 10 |
| 2 s | 8, 10 s | 4, 5 |

- 1 s windows are left out: they contain a single segment, so the TCN has nothing to model.
- 2 s segments use only the existing datasets.

## Data and protocol

Same as Experiment A, so the results can be compared fold by fold:

- **Dataset:** CHB-MIT `recording_cv`, all 24 patients.
- **Splits:** 5 folds by recording, or leave-one-recording-out when a patient has fewer than 5 recordings.
- **Training:** the remaining recordings are split 80/20 into train/validation (stratified), focal loss, `TrainConfig` defaults, same seeds.
- **Exclusion:** chb16 at 10 s.
- **Size:** 6 window/segment combinations × 4 TCN settings × about 100 folds ≈ 2,400 runs.

## Metrics

Per fold: tp, tn, fp, fn, accuracy, sensitivity, specificity, F1, AUROC.

## Shuffle test

Does the TCN actually use the order of the segments?

- Every trained model is evaluated twice on the same test windows: once as is, and once with the segments of each window in a random order (never the original order). The content of each segment is unchanged; only its position moves. Nothing is retrained.
- **Reading it:**
  - F1 barely drops: the TCN isn't using time. Any gain over 0 layers comes from capacity, not temporal modelling.
  - F1 drops clearly: the model relies on segment order.
- **Control:** the 0-layer model classifies each segment on its own and averages, so shuffling must not change it. If it does, the test is broken.
- **Expectation:** `recording_cv` windows are entirely seizure or entirely normal, so little change is expected here. Transition windows (B1) are where order should matter.

## Report

- F1, shown as mean ± variance across patients. One row per window/segment combination, one column per number of TCN layers, plus the `SpatialNet` reference.
- Paired comparison of each TCN setting against 0 layers: F1 difference, number of patients improved, and the Wilcoxon p-value.
- Shuffle test: F1 as is vs. with shuffled segments, and the drop, per TCN setting.
- A per-patient table.

## Sub-experiments

- **B1:** what happens when a window contains both classes (seizure onset or offset): [experiment_b1_transition.md](experiment_b1_transition.md).
- **B2:** a TCN and head trained on top of a frozen, pretrained spatial module: [experiment_b2_frozen_spatial.md](experiment_b2_frozen_spatial.md).
- **B3:** long temporal context (13–125 s) on continuous EEG, with event-level metrics (false alarms per hour, latency) and a moving-average control: [experiment_b3_long_context.md](experiment_b3_long_context.md).
