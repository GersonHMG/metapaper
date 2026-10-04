# Experiment B1 baseline: DCSENet on transition windows

Baseline for [experiment_b1_transition.md](experiment_b1_transition.md).

**Question:** How well does a published spectrogram-image baseline, DCSENet (Aboyeji et al., *Computers in Biology and Medicine* 185, 2025; `dcsenet_pape.pdf`), label each 1 s segment on the transition data? The comparison is with AsymSETNetGroupedSegments under the same protocol.

## Model (`models/dcsenet.py`)

The model is ported from `notebooks/dcsenet.ipynb`.

- **Spectrogram, Algorithm 1 of the paper:**
  - Hann taper, WL = 1 s.
  - 224 log-spaced frequencies from 0.5 to 40 Hz.
  - dB relative to the mean power.
  - Min-max to uint8, bilinear resize to 224×224, then 3 copies of the grayscale image.
- **Body:** dilated CNN with SE blocks, unchanged from the paper (5,763,361 parameters with one output).
- **One image per channel, as in the paper.** Each channel's spectrogram is a separate sample. Training uses every (window, channel) image. At test time the prediction for a segment is the **mean of the 21 channel probabilities**. Per-image metrics, as the paper reports them, are stored too.

## Protocol

The protocol is the same as B1 (`datasets/transition.py`, `fold_windows`):
- 5 folds by recording.
- Training and validation windows at stride 1, with validation taken from 60 s blocks.
- Test windows are non-overlapping tiles.
- Normal windows are capped at 1:5.
- Windows of 3, 5, 8 and 10 s, one label per second.

Training follows the paper: Adam, lr 1e-3, batch of 32 images, 20 epochs.

## Deviations from the paper

| Paper | Here | Why |
|---|---|---|
| One logit per image | Last FC has W logits, one per 1 s segment | Per-segment labels on the transition data |
| BCE on balanced data | Focal loss (α 0.75, γ 2), the same as B1 | The test data are 1:5 |
| Best epoch chosen on **test** accuracy (Algorithm 2) | Best epoch chosen on **validation** loss | Avoids test leakage |
| Flip, rotation and zoom augmentation | None | Flipping a spectrogram changes what it means |
| STFT hop not given; images of a whole seizure event | Hop 0.125 s inside each 3–10 s window | With hop = WL, a window has only W columns |

## Run

```bash
python -m models.dcsenet                                   # self-test vs. the notebook implementation
python -m experiments.experiment_b1_dcsenet --patients all # results/b1_dcsenet/
```

chb01 holds 4.9 % of all training windows, so the full run takes about 20.6× chb01's time.
