# Experiment A on Siena Scalp: Spatial module

Repeats [experiment_a.md](experiment_a.md) (grouped kernel vs. kernel height, `SpatialNet`) on the
**Siena Scalp** recording-CV data (`/home/gmarihuan/SIENNA_SCALP_PROCESSED`, `siena.recording_cv`),
in both of its montages.

**Question:** Which spatial module works best on Siena, for the bipolar and for the referential montage?

## Models

| Montage | Grouped kernel | Kernel height |
|---|---|---|
| bipolar (21 ch, CHB-MIT order; F9/F10 stand in for FT9/FT10) | `ELECTRODE_GROUPS` + `FLIP_CHANNELS`, as on CHB-MIT | `h` ∈ {1, 4, 8, 16, 21} |
| referential (19 electrodes) | `REFERENTIAL_GROUPS`: the double-banana chains on the electrodes | `h` ∈ {1, 4, 8, 16, 19} |

`REFERENTIAL_GROUPS` (`models/asymsetnet_grouped.py`): left temporal FP1 F7 T7 P7 O1, left parasagittal
FP1 F3 C3 P3 O1, right parasagittal FP2 F4 C4 P4 O2, right temporal FP2 F8 T8 P8 O2, midline FZ CZ PZ.
FP1/FP2/O1/O2 end two chains each, so groups overlap. There is no transverse chain and no flip.

## Data and protocol

Same as Experiment A: 5 folds by recording (leave-one-recording-out with fewer recordings), 80/20
stratified train/validation for early stopping, Focal Loss, `TrainConfig` defaults, same seeds. All models
of a fold share the split. Windows: 1, 3, 5, 8, 10 s.

- **Patients (pilot):** PN00, PN06, PN10, PN12, PN14, the patients with the most recordings.
- PN01, PN07 and PN11 have a single recording, so they can't be used (`--patients all` skips them).
- No window exclusions: every Siena recording keeps seizure windows at 10 s.
- The best `h` is picked per window on the test-fold mean F1, the same (optimistic) rule as `compare_a.py`.

## Run

```bash
cd /home/gmarihuan/metapaper && python -m experiments.experiment_a_siena --out a_siena_5p
python -m experiments.experiment_a_siena --patients all --out a_siena_all
```
Results: `experiments/results/<out>/results.md` (one section per montage) and `<out>_folds.csv`.
