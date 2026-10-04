# Experiment B1: Transition windows

Sub-experiment of [experiment_b.md](experiment_b.md).

**Question:** When a window contains both classes (it crosses a seizure onset or offset), does a TCN help label each segment correctly?

## Starting point

- **Dataset:** `transition` (`datasets/transition.py`). It keeps continuous recordings with one label per second, so windows that cross an onset or offset are kept.
- **Model:** `AsymSETNetSegments` outputs one logit per segment.
- **Earlier run:** `experiments/transition_segment_experiment.ipynb` already compared 0 TCN layers with causal and non-causal TCNs of 1–3 layers, at 3, 5 and 10 s windows, using the kernel-height spatial module.
- **Metric:** F1 on all segments, and on **boundary** segments (within ±2 s of an onset or offset). Few segments per patient are boundary segments, so use the pooled F1 across patients.

## First run (`experiment_b1_transition.py`)

Settings chosen to match Experiment B; revisit with the open questions below.

- **Model:** `AsymSETNetGroupedSegments`, the grouped spatial module with one prediction per segment.
- **TCN:** non-causal, 0–3 layers.
- **Data:** 1 s segments, windows of 3, 5, 8 and 10 s.
- **Patients:** chb01, chb18 and chb06.
- **Protocol:** the same as the notebook (`fold_windows`), focal loss on every segment.
- **Shuffle test:** each model is also evaluated with the segments of every test window in a random order. Predictions are mapped back, so each segment is scored against its own label.
- **Results:** `results/b1_transition/`.

## Open questions (to discuss)

- Re-run with the grouped spatial module? That needs a grouped version of `AsymSETNetSegments`.
- Which patients: the earlier set, or all 24?
- Which windows and segment lengths: the same as Experiment B?
- Causal and non-causal, or only non-causal, as in Experiment B?
