# Experiment B3: Models compared at the same sensitivity

24 patients, 952 test hours, 181 seizures. Each model's probabilities are thresholded with **one global threshold** (swept 0.01–0.99) over all patients and folds. For each target, the highest threshold that still detects at least that share of seizures (pooled) is used, i.e. the fewest false alarms at that sensitivity. Events as in B3: gaps < 10 s merged, a seizure is detected if an event overlaps it, latency = first detected second − onset (mean over detected seizures).

The threshold is read from the test curve, so this is a threshold-free comparison of models (like comparing ROC curves), not a deployable operating point. Curves: `operating_curves.csv`. Source run: `experiment_b3_long_context.py --save-probs` (`b3_long_context_probs/`).

## False alarms per hour at a fixed seizure sensitivity

| Model | FA/h at ≥80 % | latency (s) | FA/h at ≥90 % | latency (s) | FA/h at ≥95 % | latency (s) |
|---|---|---|---|---|---|---|
| none | 6.41 | 4.5 | 16.56 | 3.1 | 42.91 | 2.1 |
| 13s causal avg | 0.19 | 12.8 | 1.21 | 8.3 | 3.13 | 7.0 |
| 13s non-causal avg | 0.18 | 7.2 | 1.19 | 3.4 | 3.10 | 2.4 |
| 29s causal avg | 0.42 | 15.5 | 2.29 | 10.4 | 4.10 | 7.6 |
| 29s non-causal avg | 0.22 | 5.0 | 1.16 | 1.9 | 2.70 | 1.2 |
| 61s causal avg | 1.53 | 18.6 | 2.04 | 13.8 | 4.09 | 7.0 |
| 61s non-causal avg | 0.69 | 1.7 | 1.64 | 0.9 | 2.73 | 0.2 |
| 125s causal avg | 1.57 | 19.3 | 2.54 | 9.4 | 3.01 | 6.6 |
| 125s non-causal avg | 0.88 | 0.9 | 1.62 | 0.2 | 2.07 | 0.4 |
| 13s causal | 0.63 | 7.6 | 2.59 | 5.0 | 6.82 | 3.8 |
| 13s non-causal | 0.68 | 5.1 | 1.83 | 4.3 | 5.53 | 2.8 |
| 29s causal | 0.62 | 7.5 | 2.93 | 5.4 | 4.76 | 4.7 |
| 29s non-causal | 0.47 | 4.5 | 1.57 | 3.4 | 6.76 | 2.3 |
| 61s causal | 1.05 | 6.8 | 3.25 | 4.9 | 7.92 | 4.0 |
| 61s non-causal | 0.37 | 4.7 | 0.93 | 3.7 | 7.04 | 1.9 |
| 125s causal | 0.86 | 6.1 | 3.04 | 5.4 | 6.46 | 4.4 |
| 125s non-causal | 0.39 | 4.2 | 0.95 | 3.5 | 6.06 | 1.6 |

## Paired per patient

Per-patient FA/h, each model at its own pooled operating point: mean difference, patients where the TCN has fewer false alarms, two-sided Wilcoxon signed-rank p-value. No correction for multiple comparisons.

**TCN vs. moving average, same context** (negative = TCN has fewer false alarms)

| TCN | ΔFA/h at ≥80 % | fewer FA | p | ΔFA/h at ≥90 % | fewer FA | p | ΔFA/h at ≥95 % | fewer FA | p |
|---|---|---|---|---|---|---|---|---|---|
| 13s causal | +0.42 | 2 / 24 | 0.001 | +1.28 | 2 / 24 | <0.001 | +3.51 | 1 / 24 | <0.001 |
| 13s non-causal | +0.47 | 1 / 24 | <0.001 | +0.37 | 8 / 24 | 0.029 | +1.64 | 8 / 24 | 0.031 |
| 29s causal | +0.14 | 8 / 24 | 0.224 | +0.32 | 10 / 24 | 0.422 | +0.21 | 9 / 24 | 0.439 |
| 29s non-causal | +0.18 | 5 / 24 | 0.011 | +0.19 | 11 / 24 | 0.422 | +3.93 | 5 / 24 | 0.003 |
| 61s causal | -0.43 | 16 / 24 | 0.290 | +1.13 | 5 / 24 | 0.015 | +3.84 | 2 / 24 | <0.001 |
| 61s non-causal | -0.35 | 17 / 24 | 0.004 | -0.78 | 14 / 24 | 0.039 | +4.12 | 10 / 24 | 0.056 |
| 125s causal | -0.79 | 19 / 24 | 0.002 | +0.65 | 11 / 24 | 0.229 | +3.63 | 4 / 24 | <0.001 |
| 125s non-causal | -0.64 | 19 / 24 | <0.001 | -0.71 | 20 / 24 | <0.001 | +3.85 | 7 / 24 | 0.004 |

**TCN vs. no context** (negative = TCN has fewer false alarms)

| TCN | ΔFA/h at ≥80 % | fewer FA | p | ΔFA/h at ≥90 % | fewer FA | p | ΔFA/h at ≥95 % | fewer FA | p |
|---|---|---|---|---|---|---|---|---|---|
| 13s causal | -6.04 | 22 / 24 | <0.001 | -14.00 | 23 / 24 | <0.001 | -35.41 | 24 / 24 | <0.001 |
| 13s non-causal | -6.01 | 22 / 24 | <0.001 | -14.94 | 23 / 24 | <0.001 | -37.32 | 24 / 24 | <0.001 |
| 29s causal | -6.07 | 22 / 24 | <0.001 | -13.83 | 23 / 24 | <0.001 | -37.53 | 24 / 24 | <0.001 |
| 29s non-causal | -6.25 | 23 / 24 | <0.001 | -15.19 | 23 / 24 | <0.001 | -35.41 | 23 / 24 | <0.001 |
| 61s causal | -5.55 | 21 / 24 | <0.001 | -13.24 | 21 / 24 | <0.001 | -34.27 | 24 / 24 | <0.001 |
| 61s non-causal | -6.28 | 22 / 24 | <0.001 | -15.74 | 23 / 24 | <0.001 | -35.22 | 23 / 24 | <0.001 |
| 125s causal | -5.82 | 22 / 24 | <0.001 | -13.56 | 21 / 24 | <0.001 | -35.57 | 23 / 24 | <0.001 |
| 125s non-causal | -6.33 | 22 / 24 | <0.001 | -15.72 | 23 / 24 | <0.001 | -36.21 | 24 / 24 | <0.001 |
