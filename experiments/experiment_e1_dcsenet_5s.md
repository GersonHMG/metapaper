# Experiment E1: DCSENet baseline at 5 s windows

Fills Table 3.4 (DCSENet vs. the spatial block, CHB-MIT), 5 s rows only for now.

**Question:** Under the same folds and test sets as Experiment A, does the grouped spatial block beat DCSENet at 5 s windows?

## Why it is needed

The published DCSENet numbers (87.29 % accuracy LOPOCV, 94.48 % accuracy 10-fold) cannot be compared directly with ours:

- DCSENet balances its test data 1:1. Ours keeps the natural ≤ 1:5 ratio, where predicting "normal" for every window already gives 83 % accuracy.
- DCSENet splits by window. We split by recording.

So DCSENet has to be re-run inside our protocol. The earlier run (`a_dcsenet/`: chb01 and chb02, 1–10 s, Hann taper with WL = 1 s, no band-pass, the paper's 20 epochs) used different settings, so it is not reused.

## Model: DCSENet (`models/dcsenet.py`), Aboyeji et al. (2025)

- **Preprocessing:** FIR band-pass 0.5–30 Hz with a Hamming window. That gives passband ripple 0.0194 and stopband attenuation 53 dB, as in the paper. The filter has 1,691 taps and zero phase, with MNE-style transition bands.
  - `recording_cv` stores cut windows, so the filter runs on each 5 s channel signal with reflect padding. The paper filters the continuous recording.
- **Input:** one STFT spectrogram image per channel (Algorithm 1). Frequencies are log-spaced from 0.5 to 40 Hz, in dB relative to the mean power, then min-max scaled and resized to 224 × 224 × 3.
  - The paper's 22 channels become our 21.
- **STFT taper and window length:** **Gaussian taper, σ = 0.1, WL = 5 s.** This is the paper's best setting for WL = 5 s: 86.99 % accuracy, against 84.88 % for Hann (Table 3, Fig. 6).
  - On a 5 s window this is **one STFT frame**. Each image is a single spectrum stretched across the time axis.
- **Network:** 2 × [dilated Conv2D, 32 filters, 3 × 3, dilation 2 → MaxPool 2 × 2 → SE block] → Dropout → Dense 64 → 1 output, giving 5,763,361 parameters.
- **Window prediction:** the mean of the 21 channel-image probabilities, as in `a_dcsenet`. Both image-level counts (`img_TP`, …) and window-level counts (`TP`, …) are saved.

## Data and protocol

Identical to the 5 s cells of Tables 3.1 and 3.2, so the comparison is paired:

- **Dataset:** CHB-MIT, `recording_cv`, 5 s windows.
- **Patients:** all 24.
- **Folds:** the same folds and seeds as `a_grouped_all` and `b_tcn`: 5 folds by recording, or leave-one-recording-out when a patient has fewer than 5 seizure recordings. The report checks that every fold's test recordings match `b_tcn`.
- **Training:** the remaining recordings are split 80/20 into train and validation, with validation used only for early stopping. `TrainConfig` defaults:
  - focal loss, α 0.75, γ 2
  - Adam, lr 1e-3, weight decay 1e-4
  - batch of 64 images
  - at most 100 epochs, patience 15
- **Test set:** keeps its natural ≤ 1:5 seizure:normal ratio.

## Metrics

Saved per fold: tp, tn, fp, fn, accuracy, sensitivity, specificity, F1 and AUROC (window level), plus the image-level counts.

The report gives the mean of the per-patient fold means ± **standard deviation** across patients.

## Comparison

- **Spatial block, 5 s:** `b_tcn`, 0 TCN layers, 1 s segments.
- **Paired test per patient:** the difference (spatial − DCSENet) and the number of patients where the spatial block wins, with a two-sided Wilcoxon signed-rank test over the 24 patients. This is done for Acc, Sen, Spec, F1 and AUROC.

## Cost

chb01 took 5.4 min (5 folds). chb01 has 4.2 % of the 5 s windows, so the full run takes about 2.2 h.

## Run

```bash
python -m experiments.experiment_e1_dcsenet_5s --patients chb01   # timing
python -m experiments.experiment_e1_dcsenet_5s --patients all     # resumes from the checkpoint
```

## Output

- `results/e1_dcsenet_5s/e1_dcsenet_5s_folds.csv`
- `results/e1_dcsenet_5s/results.md`, with the summary table and the paired test

**Goes into:** Table 3.4, rows "5 s, DCSENet" and "5 s, Bloque espacial".
