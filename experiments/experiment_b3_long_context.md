# Experiment B3: Long temporal context on continuous EEG

Sub-experiment of [experiment_b.md](experiment_b.md).

**Question:** Does a temporal layer with long context (up to 2 minutes) improve seizure detection on continuous EEG, measured as a clinical detector is: false alarms per hour, seizures detected, and latency?

**Motivation:**
- **B:** windows were entirely one class, and the TCN didn't help.
- **B1:** on windows of at most 10 s, the TCN helped overall, mostly through neighbouring context rather than order.
- **Here:** seizures evolve over tens of seconds, so longer context is where order and dynamics should matter. Continuous evaluation, with seizure-free hours included, is where a temporal layer can suppress false alarms.

## Data

`datasets/continuous.py` (built by `datasets/build_continuous_dataset.py`):
- **Coverage:** every recording of every patient, with and without seizures, about 990 h in total.
- **Labels:** one per second.

**Folds:**
- Seizure recordings are split exactly as in B1: 5 folds by recording, or leave-one-recording-out when a patient has fewer.
- Seizure-free recordings are dealt round-robin to the folds, so every test fold also contains seizure-free hours.

## Models (inside every fold)

1. **Encoder:** `AsymSETNetGroupedSegments(n_segments=1)`, the grouped spatial module classifying single 1 s segments.
   - **Training data:** the training recordings: all seizure seconds plus up to 5× as many normal seconds.
   - **Validation:** about 20% of the ~300 s time blocks.
   - **Its own output is the `none` model,** with no temporal context.
2. **Freeze** the encoder and embed every second: 32 features per second.
3. **`TemporalHead`** (`models/temporal_head.py`): a TCN over the embeddings, then Linear → ReLU → Linear per second.
   - **Context:** 2, 3, 4 or 5 layers, giving a receptive field of **13, 29, 61 or 125 s**.
   - **Direction:** causal (past only, a real-time detector) and non-causal (both sides).
   - **Training:** on 240 s sequences: all sequences with a seizure plus up to 5× as many without. Per-second focal loss, early stopping on validation sequences.
4. **Control (`avg`):** a non-learned moving average of the `none` probabilities over the same window, causal or centred. It tests whether the TCN does more than averaging.

## Evaluation

- **Inference:** each model runs over whole recordings.
- **Threshold:** per model and fold, the best per-second F1 on the validation seconds.
- **Per second:** F1 on all test seconds, and on boundary seconds (within ±2 s of an onset or offset).
- **Events:** positive seconds, with gaps under 10 s merged.
  - A seizure is **detected** if an event overlaps it.
  - **Latency** is the first detected second minus the onset.
  - An event overlapping no seizure is a **false alarm**. Report false alarms per hour of test EEG.
- **Paired tests over patients:** each model vs. `none`, and each TCN vs. `avg` with the same context. Wilcoxon signed-rank, no correction for multiple comparisons.

## Not included yet

- **Shuffle test with long context.**
- **Multiple-comparison correction:** with 16 comparisons, apply Holm–Bonferroni before claiming significance.
- **Event-merging and tolerance choices:** the 10 s merge gap and overlap-based detection are fixed choices. A sensitivity analysis on them would strengthen the claims.
