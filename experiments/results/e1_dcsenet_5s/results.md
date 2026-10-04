# Experiment E1: DCSENet baseline at 5 s windows

Recording CV at 5 s (5 folds, leave-one-recording-out when a patient has fewer seizure recordings), 1 patients, same folds and test sets as `a_grouped_all` and `b_tcn`. See `experiments/experiment_e1_dcsenet_5s.md`.

## DCSENet settings

- **Preprocessing:** FIR band-pass 0.5–30 Hz, Hamming window (1,691 taps, zero phase), applied to each 5 s channel signal with reflect padding.
- **Spectrogram (Algorithm 1):** **Gaussian taper, σ = 0.1, STFT window WL = 5 s**, the paper's best setting for WL = 5 s (Table 3, Fig. 6). On a 5 s window this is a single STFT frame: the image is one log-frequency spectrum (224 bins, 0.5–40 Hz, dB relative to its mean) stretched to 224 × 224 × 3.
- **Network:** DCSENet as in the paper (5,763,361 parameters), one output.
- **Channels:** one image per channel (21 channels in our montage), window probability = mean of the 21 image probabilities.
- **Training:** `TrainConfig` defaults: focal loss (α 0.75, γ 2.0), Adam lr 0.001, weight decay 0.0001, batch 64 images, ≤ 100 epochs, early stopping (patience 15) on validation loss.

Identical test recordings to `b_tcn` in every fold: **True** (5 folds compared).

## Table 3.4, 5 s (window level)

Mean of the per-patient fold means ± standard deviation across patients, in %.

| Model | Acc | Sen | Spec | F1 | AUROC |
|---|---|---|---|---|---|
| DCSENet | 91.3 ± nan | 58.3 ± nan | 97.9 ± nan | 61.1 ± nan | 88.5 ± nan |
| Bloque espacial (`b_tcn`, 0 layers) | 98.7 ± nan | 96.9 ± nan | 99.1 ± nan | 96.2 ± nan | 99.6 ± nan |

## Paired test per patient

Per-patient fold means; difference in percentage points (positive = spatial block better); two-sided Wilcoxon signed-rank test over patients.

| Metric | Spatial − DCSENet | Spatial wins | Ties | p (Wilcoxon) |
|---|---|---|---|---|
| Acc | +7.4 | 1 / 1 | 0 | 1.000 |
| Sen | +38.6 | 1 / 1 | 0 | 1.000 |
| Spec | +1.2 | 1 / 1 | 0 | 1.000 |
| F1 | +35.2 | 1 / 1 | 0 | 1.000 |
| AUROC | +11.2 | 1 / 1 | 0 | 1.000 |

## DCSENet per image (paper style)

Every channel image scored on its own; mean of per-patient fold means ± std, in %.

| Acc | Sen | Spec | F1 |
|---|---|---|---|
| 83.9 ± nan | 51.9 ± nan | 90.4 ± nan | 46.5 ± nan |

Folds where DCSENet predicts no seizure at all: **1** of 5 (chb01 f1).

## Per patient

| Patient | Folds | DCSENet F1 | Spatial F1 | DCSENet AUROC | Spatial AUROC |
|---|---|---|---|---|---|
| chb01 | 5 | 0.611 | 0.962 | 0.885 | 0.996 |

Training time, all folds: 5.4 min. Per-fold rows: `e1_dcsenet_5s_folds.csv`.
