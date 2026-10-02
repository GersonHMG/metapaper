# Datasets

Three window datasets generated from the **CHB-MIT Scalp EEG Database** and stored in `datasets/data/`.
The data is gitignored, so rebuild it with the `build_*_dataset.py` scripts. Read it with the loader modules, which don't need MNE.

```
datasets/data/
├── balanced_windows/            306 MB
│   ├── chbmit_windows_all.npz
│   └── per_patient/chbXX.npz
├── recording_cv/                2.1 GB
│   └── win{1,3,5,8,10}s/per_patient/chbXX.npz
└── transition/                  6.7 GB
    └── per_patient/chbXX.npz
```

## Source and common preprocessing

- **Raw data:** `/home/gmarihuan/chbmit/chbXX/*.edf`, 24 cases (chb01–chb24), 256 Hz.
  - Seizure annotations are in `chbmit_summary.json`, as start/end times in whole seconds.
  - chb01 and chb21 are the same person, recorded 1.5 years apart.
- **Channels:** 21 bipolar channels, in the fixed order of `REQUIRED_CHANNELS` (`data_loader.py`). Recordings missing any of them are skipped.
- **Units:** microvolts (MNE's volts × 1e6). This is needed for training, because at volt scale the signal sits below BatchNorm's eps.
- **Storage:** `X` is stored as **float16**. The loaders return float32.
- **Labels:** `1` = seizure (ictal), `0` = normal.

| Dataset | Recordings | Label | Windows crossing seizure boundaries | Balance | Intended CV |
|---|---|---|---|---|---|
| `balanced_windows` | all | per window | no | 1:1 per patient | leave-one-patient-out |
| `recording_cv` | seizure recordings only | per window | no | ≤ 1:5 per recording | within patient, by recording |
| `transition` | seizure recordings only | per 1 s segment | yes | ≤ 1:5 per recording (applied when windows are cut) | within patient, by recording |

---

## 1. `balanced_windows/`

**Build:** `python datasets/build_balanced_windows_dataset.py` · **Load:** `datasets/balanced_windows.py`

**How it's built:** every recording of each patient is cut into **3 s non-overlapping windows** (768 samples). Each window lies entirely inside a seizure or a normal stretch. Normal windows are then undersampled to **1:1** for each patient.

| File | Arrays |
|---|---|
| `chbmit_windows_all.npz` | `X` (7208, 21, 768), `y` (7208,), `patient_id` (7208,), an index into `patients` (24,), and `window_len` |
| `per_patient/chbXX.npz` | `X` (n, 21, 768), `y` (n,) |

There are 7,208 windows in total (3,604 per class).

```python
from datasets import balanced_windows as bw
X, y, patient = bw.load(exclude=["chb16"])        # patient = name per window
for test_p, train_idx, test_idx in bw.lopo_splits(patient): ...
```

**Caveat:** balancing happens before any split, so the test folds are also 50/50. Accuracy and F1 there overstate real-world performance.

---

## 2. `recording_cv/`

**Build:** `python datasets/build_recording_cv_dataset.py [--windows 1 3 5 8 10]` · **Load:** `datasets/recording_cv.py`

**How it's built:**
- Only **recordings that contain seizures** are used.
- They're cut into non-overlapping windows of **1, 3, 5, 8 or 10 s**, one dataset per length, and each window lies entirely inside a seizure or a normal stretch.
- Within each recording, normal windows are capped at **5 per seizure window**.
- Recordings whose seizures are all shorter than the window are dropped at that window length. At 8 s this affects only chb16 (5 recordings become 4, with 6 seizure windows). At 10 s it affects chb02 (3 recordings become 2) and chb16 (5 become 1, so exclude chb16 at 10 s).

| Folder | Window shape | Total windows (seizure / normal) | Size |
|---|---|---|---|
| `win1s/` | (21, 256) | 64,437 (11,015 / 53,422) | 433 MB |
| `win3s/` | (21, 768) | 21,075 (3,604 / 17,471) | 424 MB |
| `win5s/` | (21, 1280) | 12,516 (2,141 / 10,375) | 419 MB |
| `win10s/` | (21, 2560) | 5,997 (1,027 / 4,970) | 401 MB |

**Each `per_patient/chbXX.npz` holds:**
- `X` (n, 21, window_sec·256)
- `y` (n,)
- `recording` (n,), the EDF filename of each window
- `window_len` (in samples) and `window_sec`

```python
from datasets import recording_cv as rcv
X, y, rec = rcv.load_patient("chb01", window_sec=5)
for X_tr, y_tr, X_te, y_te in rcv.iter_folds("chb01", window_sec=5, n_splits=5,
                                             balance_train=True): ...
```

- `recording_folds(y, rec, n_splits)` gives grouped folds; `n_splits=None` is leave-one-recording-out.
- Patients with fewer recordings than `n_splits` fall back to leave-one-recording-out. Ten patients have only 3–4 seizure recordings, and only chb12, chb15 and chb24 have 10 or more.

---

## 3. `transition/`

**Build:** `python datasets/build_transition_dataset.py [--patients chb01 ...]` · **Load:** `datasets/transition.py`

**How it's built:**
- Each patient's **seizure recordings** are stored as **one continuous signal** with a **label for every second**.
- Windows are cut when loading, not stored, so one dataset serves every window length and stride.
- Because the annotations are whole seconds, a 1 s segment is never half seizure.
- Windows can cross onsets and offsets, which makes the dataset suited to labelling each segment.

**Each `per_patient/chbXX.npz` holds:**

| Array | Shape | Meaning |
|---|---|---|
| `X` | (21, S·256) float16 | continuous signal, recordings concatenated |
| `sec_label` | (S,) int8 | label of every second |
| `rec_names` | (R,) | EDF filename of each recording |
| `rec_bounds` | (R, 2) | `[start, end)` second of each recording in `X` |
| `sfreq` | scalar | 256 |

There are 185.5 h of seizure recordings and 11,015 seizure seconds, from 3 to 14 recordings per patient. The largest file is chb06 (25.9 h).

```python
from datasets import transition as tr
d = tr.load_patient("chb01")
for fold, (train_recs, test_recs) in enumerate(tr.split_folds(d, n_splits=5)):
    w = tr.fold_windows(d, train_recs, test_recs, window_sec=5, seed=fold)
    w["train"]["X"], w["train"]["Y"]      # (n, 21, 1280), (n, 5) per-segment labels
    w["test"]["boundary"]                 # (n, 5) segments within +-2 s of an onset/offset
```

**The protocol `fold_windows` applies:**
- **Train and validation:** windows with a 1 s stride. Validation is about 20% of ~60 s time blocks from the training recordings, so the two sets never share a second.
- **Test:** non-overlapping tiles, so every second is scored once. Tiles that touch a boundary are always kept.
- **Balance:** every window with a seizure segment is kept, and all-normal windows are capped at 5× that count per recording.

---

## Rebuilding

| Dataset | Command | Notes |
|---|---|---|
| all | — | needs the `torchy` environment and the raw EDFs |
| `balanced_windows` | `python datasets/build_balanced_windows_dataset.py` | reads every EDF (all recordings) |
| `recording_cv` | `python datasets/build_recording_cv_dataset.py` | seizure recordings only; all 4 lengths from one read |
| `transition` | `python datasets/build_transition_dataset.py` | seizure recordings only; writes 6.7 GB |

Every script writes into its own folder in `datasets/data/`, whatever directory you run it from. Builds overwrite existing files.
