"""
Load the continuous dataset (built by build_continuous_dataset.py): every
recording of a patient, with and without seizures, as one continuous signal
with a label per second.

    from datasets import continuous as ct

    d = ct.load_patient("chb01")           # X is a read-only memmap
    for fold, (train_recs, test_recs) in enumerate(ct.split_folds(d, n_splits=5)):
        ...

Folds: seizure recordings are split exactly as in datasets/transition.py
(split_folds on seizure seconds), so test folds match Experiment B1; seizure-free
recordings are then dealt round-robin over the folds (filename order), so every
test fold also holds seizure-free hours for false alarms per hour.
No MNE import: loading is fast.
"""

from pathlib import Path

import numpy as np

from datasets.recording_cv import recording_folds

SFREQ = 256
DATA_DIR = Path(__file__).resolve().parent / "data" / "continuous"
PER_PATIENT_DIR = DATA_DIR / "per_patient"


def available_patients():
    return sorted(p.name for p in PER_PATIENT_DIR.glob("chb*") if (p / "meta.npz").exists())


def load_patient(name):
    """
    -> dict with X (21, S*256) float16 memmap, sec_label (S,), rec_names (R,),
       rec_bounds (R, 2), rec_has_seizure (R,), sec_rec (S,) recording of every second.
    """
    folder = PER_PATIENT_DIR / name
    if not (folder / "meta.npz").exists():
        raise FileNotFoundError(
            f"{folder} not built; run `python datasets/build_continuous_dataset.py "
            f"--patients {name}`.")
    f = np.load(folder / "meta.npz", allow_pickle=True)
    d = {k: f[k] for k in f.files}
    d["X"] = np.load(folder / "X.npy", mmap_mode="r")
    lengths = d["rec_bounds"][:, 1] - d["rec_bounds"][:, 0]
    d["sec_rec"] = np.repeat(np.arange(len(lengths)), lengths)
    d["name"] = name
    return d


def split_folds(d, n_splits=5, seed=0):
    """Yield (train_recs, test_recs) recording indices. Seizure recordings are
    split by recording_folds on their seconds (as transition.split_folds);
    seizure-free recordings go round-robin to the folds."""
    sz = np.flatnonzero(d["rec_has_seizure"])
    free = np.flatnonzero(~d["rec_has_seizure"])
    in_sz = np.isin(d["sec_rec"], sz)
    folds = [np.unique(d["sec_rec"][in_sz][te])
             for _, te in recording_folds(d["sec_label"][in_sz], d["sec_rec"][in_sz],
                                          n_splits, seed)]
    for k, r in enumerate(free):
        folds[k % len(folds)] = np.append(folds[k % len(folds)], r)
    all_recs = np.arange(len(d["rec_names"]))
    for test in folds:
        test = np.sort(test)
        yield np.setdiff1d(all_recs, test), test
