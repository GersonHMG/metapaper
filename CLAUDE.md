# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# metapaper

Research code for **EEG seizure detection on the CHB-MIT scalp EEG dataset** (24 pediatric patients), with deep models trained on fixed-length multichannel windows.
The main models are AsymSETNet (asymmetric conv + Squeeze-Excitation spatial module feeding a TCN) and a grouped variant with one spatial block per bipolar montage chain.

## Environment
- Conda env `torchy` (Python 3.10, torch 2.11 with CUDA, mne 1.12, sklearn 1.7). GPU: RTX 5090.
- Raw data: `/home/gmarihuan/chbmit/chbXX/*.edf` (43 GB, read-only). Seizure annotations: `/home/gmarihuan/chbmit/chbmit_summary.json`, with an identical copy in `datasets/`.
- `datasets/` is a regular package (it has an `__init__.py`). Without that file, the pip-installed HuggingFace `datasets` package shadows it. Import as `from datasets.data_loader import ...`, with `metapaper/` first on `sys.path`: run from `metapaper/` or set `PYTHONPATH=/home/gmarihuan/metapaper`.
- No test suite, linter or build system. "Testing" means running a build script or experiment on one patient first.

## Commands
```bash
conda activate torchy
# Build datasets (the scripts put metapaper/ on sys.path themselves, so they run from any directory)
python datasets/build_balanced_windows_dataset.py
python datasets/build_recording_cv_dataset.py --windows 3 5      # default: all of 1 3 5 8 10
python datasets/build_transition_dataset.py --patients chb01     # default: all patients
# LOPO-CV runs (imports lopocv.* and models.*)
cd notebooks && PYTHONPATH=/home/gmarihuan/metapaper python lopocv_experiments.py
```

## Layout
- `datasets/data_description.md` is the reference for the three generated datasets (sizes, labels, balance, intended CV). Read it before changing a build script or loader.
- `datasets/`: the data pipeline, the canonical code for data prep.
  - `data_loader.py`: `load_patient_data(patient, only_seizure_files=False, return_recordings=False)`.
    - Reads the EDFs with MNE and keeps only files that have all 21 `REQUIRED_CHANNELS`, in that fixed order.
    - Converts **volts to microvolts (×1e6)**. This is required: at volt scale the signal is below BatchNorm's eps and training stalls at a loss of ln 2.
    - Returns variable-length segments: one per seizure, one per normal gap between seizures, one per fully normal file.
  - `preprocessing.py`:
    - `segments_to_windows(..., groups=)` cuts segments into windows and carries the recording id per window.
    - `balance_dataset` gives 1:1 by undersampling or oversampling.
    - `cap_class_ratio(max_ratio=)` limits the majority:minority ratio.
  - `build_balanced_windows_dataset.py` writes `datasets/data/balanced_windows/`.
    - 3 s windows, 1:1 per patient, every recording.
    - Outputs `chbmit_windows_all.npz` (`X, y, patient_id, patients, window_len`) and `per_patient/`.
    - Used for **leave-one-patient-out** CV.
  - `build_recording_cv_dataset.py` writes `datasets/data/recording_cv/win{1,3,5,8,10}s/per_patient/chbXX.npz` (`X, y, recording, window_len, window_sec`).
    - Uses only recordings that contain seizures, with at most 5 normal windows per seizure window **within each recording**.
    - Used for **within-patient CV grouped by recording**: `recording_folds(y, recording, n_splits=5|10|None)`, where `None` is leave-one-recording-out.
    - `per_patient_dir(w)` gives the folder for a window length. `--windows 3 5` builds only some lengths.
  - `build_transition_dataset.py` writes `datasets/data/transition/per_patient/chbXX.npz` (about 6.7 GB in total).
    - Stores each patient's seizure recordings as one **continuous** signal: `X` (21, S·256) float16, plus `sec_label` (one label per second), `rec_names` and `rec_bounds`.
    - Windows are cut at load time, so one dataset serves every window length and stride.
    - Used for labelling each 1 s segment, including windows that cross seizure onsets and offsets.
  - `build_continuous_dataset.py` writes `datasets/data/continuous/per_patient/chbXX/` (`X.npy`, a float16 memmap of shape (21, S·256), plus `meta.npz` with `sec_label`, `rec_names`, `rec_bounds` and `rec_has_seizure`). About 35 GB.
    - Stores **every** recording, seizure-free ones included, as one continuous signal with one label per second, for long-context models and event-level metrics.
    - Builds patients in parallel (`--jobs`) and skips patients that are already built.
  - **Loaders (use these to read the data; they don't import MNE):**
    - `balanced_windows.py`: `load(patients=, exclude=)` returns `X, y, patient`. Also `load_patient(name)` and `lopo_splits(patient)`.
    - `recording_cv.py`: `load_patient(name, window_sec)` returns `X, y, recording`. Also `iter_folds(name, window_sec, n_splits, balance_train=)`, `iter_patients(window_sec, exclude=)` and `recording_folds`.
    - `continuous.py`: `load_patient(name)` (`X` is a read-only memmap) and `split_folds(d)`. Seizure recordings are split exactly as in `transition.split_folds`; seizure-free recordings go round-robin to the folds.
    - `transition.py`: `load_patient(name)`, `split_folds(d, n_splits)` (by recording), and `fold_windows(d, train_recs, test_recs, window_sec, ...)`.
      - `fold_windows` returns train/val/test dicts with `X`, `Y` (n, W) per-segment labels, and `boundary` (n, W).
      - Train and validation use stride-1 windows, with validation taken from ~60 s time blocks so the two sets share no seconds.
      - Test uses non-overlapping tiles, and tiles that touch a boundary are always kept.
      - All three sets use the 1:5 window cap.
    - The loaders return `X` as float32. The paths and helpers live here, and the build scripts import them.
  - `data/`: generated `.npz` files, which are gitignored. Rebuild them with the scripts; never edit them by hand.
- `experiments/`: patient-dependent experiments on the recording-CV data.
  - Protocol: 5 folds by recording; the rest is split 80/20 into train/validation (stratified, validation only for early stopping); Focal Loss.
  - `training.py` has the shared `FocalLoss`, `TrainConfig`, `train_model`, `confusion_counts` and `add_metrics`.
  - `experiment_a.md`: spatial-module comparison (grouped kernel vs. kernel height) with `SpatialNet`, recording CV at 1/3/5/10 s. `experiment_a1_kernel_height.md` covers choosing the kernel height; its selection rule is still open.
  - `transition_segment_experiment.ipynb` labels each 1 s segment on the transition data, comparing 0 TCN layers with causal and non-causal TCNs of 1 to 3 layers, at 3, 5 and 10 s windows.
    - It reports F1 on all segments and on **boundary** segments (within ±2 s of an onset or offset).
    - Each patient has few boundary segments, so use the pooled F1 across patients.
  - Long notebooks save progress to a `*_folds_partial.csv` checkpoint and resume from it. Delete that file to start a fresh run.
  - GPU training isn't bit-reproducible, so the same configuration varies by about ±0.02 in F1 between runs.
- `models/`: `asymsetnet.py`.
  - `AsymSETNet`: `tcn_channels=()` means no TCN, and the segment features are averaged instead. The TCN takes `causal=True/False`; non-causal padding is symmetric.
  - `AsymSETNetSegments`: one logit per segment, shape (B, N).
  - Also `asymsetnet_grouped.py` (`AsymSETNetGrouped`, with `ELECTRODE_GROUPS` and `FLIP_CHANNELS` indexed into the 21-channel order).
  - `spatial_net.py` (`SpatialNet(spatial="height"|"grouped", kernel_height=)`): the spatial module on the whole window, then GAP and a linear head. It has no TCN and no `n_segments`, so any window length works.
  - `asymsetnet_grouped_segments.py` (`AsymSETNetGroupedSegments(n_segments, tcn_channels=(), causal=False)`): the grouped spatial module per segment, an optional TCN, and the `SpatialNet` head shared by every segment. It returns logits of shape (B, N). `predict_proba` averages the segment probabilities into one window probability. `FocalLoss` broadcasts window labels (B,) over (B, N) logits. `freeze_spatial()` and `from_pretrained_spatial(pretrained, tcn_channels)` support Experiment B2 (frozen spatial module, new TCN and head). Freezing also keeps the spatial module in eval mode, so its BatchNorm statistics stay fixed.
  - `temporal_head.py` (`TemporalHead`): a TCN over per-second embeddings with one logit per second; it runs on sequences of any length. Each layer has two dilated convolutions, so the receptive field is 1 + 4·(2ⁿ − 1) s: 13/29/61/125 s for 2–5 layers.
- `notebooks/`: experiments.
  - `lopocv_experiments.py` runs N repeated LOPO-CV runs per model and writes `<model>_lopocv/run_<x>.csv`. It imports `lopocv.*` and `models.*`, so run it from `notebooks/` with `metapaper/` on `PYTHONPATH`.
  - `lopocv/run_lopocv_fold.py` has `run_fold` (Adam, BCEWithLogits, early stopping). It returns a dict: `tp, tn, fp, fn, acc, sen, spec, f1`.
  - The `.ipynb` files are exploratory: DCSENet, CNN baselines, adversarial learning, kernel experiments.
  - `notebooks/load_patient.py` is an older copy of the pipeline; `datasets/` is canonical.
  - `lopocv/grouped_asymsetnet_lopocv.py` is stale and won't run: it never imports `torch`, hardcodes the data path, and unpacks `run_fold` as a 3-tuple although it returns a dict. Use `lopocv_experiments.py`.
- `example_datasets_usage.ipynb`: worked examples for building and loading both datasets.

## Data conventions
- Window arrays are `(n_windows, 21, window_samples)` at 256 Hz. Labels: `1` = seizure (ictal), `0` = normal.
- `X` is stored as **float16**. The loader modules cast it to float32.
- Channel order is `REQUIRED_CHANNELS` in `data_loader.py`. The grouped model depends on these indices, so don't reorder them.
- Models expect `(B, 21, T)` and return logits of shape `(B,)`. **`T` must be divisible by `n_segments`**. The experiments use `n_segments=3`, which works for 3 s windows (768 samples) but not for 1, 5 or 10 s (256, 1280, 2560 samples). Pick `n_segments` to match the window length.

## Evaluation pitfalls to keep in mind
- The balanced dataset is balanced **before** splitting, so LOPO test folds are 50/50. Accuracy there overstates real-world performance.
- Recording CV: balance only the **training** fold; the test fold keeps its 1:5 ratio. Never let a recording appear in both train and test.
- Many patients have only 3 or 4 seizure recordings, so they allow only leave-one-recording-out; 10-fold is possible only for chb12, chb15 and chb24.
- Recordings whose seizures are all shorter than the window are dropped. At 8 s only chb16 is affected (5 recordings become 4, leaving 6 seizure windows). At 10 s chb02 goes from 3 recordings to 2, and chb16 from 5 to 1. Exclude chb16 at 10 s.

## Working conventions
- Keep data prep in `datasets/`, not in notebooks. New datasets get a `build_*_dataset.py` script that writes to `datasets/data/<name>/`.
- Use repo-relative output paths (`Path(__file__).resolve().parent / "data" / ...`). Some notebooks still hardcode `/home/gmarihuan/metapaper/datasets/data/...`.
- Building from the EDFs is slow (minutes per dataset). Test on one patient first (e.g. `process_patient("chb01")`) before running the full build.
