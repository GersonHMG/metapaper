"""
Load the recording-CV CHB-MIT datasets (built by build_recording_cv_dataset.py):
seizure recordings only, at most 1:5 seizure:normal per recording, one
dataset per window length (1, 3, 5, 8, 10 s), for within-patient CV grouped by
recording.

    from datasets import recording_cv as rcv

    X, y, rec = rcv.load_patient("chb01", window_sec=5)

    for X_tr, y_tr, X_te, y_te in rcv.iter_folds("chb01", window_sec=5,
                                                 n_splits=5, balance_train=True):
        ...

    for name, X, y, rec in rcv.iter_patients(window_sec=10, exclude=["chb16"]):
        ...

X is returned as float32 (stored as float16), shape (n, 21, window_sec*256).
`rec` holds the EDF filename (recording) of every window.
"""

import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from datasets.preprocessing import balance_dataset

SFREQ = 256
WINDOW_SECONDS = [1, 3, 5, 8, 10]
DATA_DIR = Path(__file__).resolve().parent / "data" / "recording_cv"


def per_patient_dir(window_sec: int) -> Path:
    """Folder holding the per-patient files for one window length."""
    return DATA_DIR / f"win{window_sec}s" / "per_patient"


def load_windows(path, dtype=np.float32):
    """Load a saved .npz as a dict, casting X to `dtype` (float16 on disk)."""
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["X"] = out["X"].astype(dtype)
    return out


def available_patients(window_sec: int = 3):
    """Patient names with a saved file for this window length."""
    return sorted(p.stem for p in per_patient_dir(window_sec).glob("chb*.npz"))


def load_patient(name, window_sec=3, dtype=np.float32):
    """Load one patient -> (X, y, recording)."""
    path = per_patient_dir(window_sec) / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; build it with "
            f"`python datasets/build_recording_cv_dataset.py --windows {window_sec}`.")
    d = load_windows(path, dtype)
    return d["X"], d["y"], d["recording"]


def iter_patients(window_sec=3, patients=None, exclude=(), dtype=np.float32):
    """Yield (name, X, y, recording) for each patient, one at a time."""
    names = patients if patients is not None else available_patients(window_sec)
    for name in names:
        if name in exclude:
            continue
        yield (name, *load_patient(name, window_sec, dtype))


def recording_folds(y, recording, n_splits=5, seed=0):
    """
    Yield (train_idx, test_idx) splits where no recording is in both sets.

    n_splits=None -> leave-one-recording-out. Otherwise StratifiedGroupKFold;
    if the patient has fewer recordings than n_splits, n_splits is capped at
    the number of recordings (i.e. leave-one-recording-out) with a warning.
    """
    y = np.asarray(y)
    recording = np.asarray(recording)
    n_rec = len(np.unique(recording))

    if n_splits is None or n_splits >= n_rec:
        if n_splits is not None and n_splits > n_rec:
            warnings.warn(f"Only {n_rec} recordings; using leave-one-recording-out "
                          f"instead of {n_splits}-fold.")
        splitter = LeaveOneGroupOut()
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=seed)

    yield from splitter.split(np.zeros(len(y)), y, groups=recording)


def iter_folds(name, window_sec=3, n_splits=5, seed=0, balance_train=False,
               dtype=np.float32):
    """
    Load one patient and yield (X_train, y_train, X_test, y_test) per fold.

    balance_train=True undersamples the training fold to 1:1; the test fold
    always keeps its original (up to 1:5) ratio.
    """
    X, y, rec = load_patient(name, window_sec, dtype)
    for k, (tr, te) in enumerate(recording_folds(y, rec, n_splits, seed)):
        X_tr, y_tr = X[tr], y[tr]
        if balance_train:
            X_tr, y_tr = balance_dataset(X_tr, y_tr, seed=seed + k)
        yield X_tr, y_tr, X[te], y[te]
