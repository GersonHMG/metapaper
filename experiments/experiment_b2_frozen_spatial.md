# Experiment B2: TCN on a frozen spatial module

Variant of [experiment_b.md](experiment_b.md). Same windows, segments, data, protocol and metrics.

**Question:** Does the TCN add anything on top of an already trained spatial module, if only the TCN and the classifier are trained?

In Experiment B, everything is trained end to end, so the TCN's effect is mixed with the spatial module learning differently. Here the spatial module is fixed, so any change comes from the TCN and the head only.

## Procedure (inside every fold)

The spatial module is trained only on that fold's training recordings, so no test data leaks into it.

1. **Pretrain:** train `AsymSETNetGroupedSegments(n_segments, tcn_channels=())` on the fold's train/validation split. This is exactly B's 0-layer model, with the same seed.
2. **Freeze:** take its spatial module with `freeze_spatial()`: no gradients, and kept in eval mode so BatchNorm statistics stay fixed. Drop its classifier.
3. **Train the TCN and head:** `AsymSETNetGroupedSegments.from_pretrained_spatial(pretrained, tcn_channels)` creates a new non-causal TCN with 1, 2 or 3 layers (`(32,)`, `(32, 32)`, `(32, 32, 32)`) and a new Linear → ReLU → Linear head. Train them on the same train/validation split, with focal loss and early stopping.

Each segment is classified, and the window probability is the mean of its segment probabilities, as in B.

## Windows and segments

| Segment | Windows | n_segments |
|---|---|---|
| 1 s | 3, 5, 8, 10 s | 3, 5, 8, 10 |
| 2 s | 8, 10 s | 4, 5 |

All 24 patients, chb16 excluded at 10 s. Per fold: 1 pretraining run and 3 TCN runs, about 2,400 runs in total. The pretraining runs are also B's 0-layer results.

## Report

- F1 mean ± variance across patients: one row per window/segment combination, columns 0 layers (pretrained only), and frozen + 1, 2 and 3 TCN layers.
- Paired comparison per patient (F1 difference, patients improved, Wilcoxon p):
  - frozen + TCN vs. 0 layers: does a TCN help a fixed spatial module?
  - frozen + TCN vs. B end-to-end with the same number of layers: is it better to freeze or to train everything?
- A per-patient table.
