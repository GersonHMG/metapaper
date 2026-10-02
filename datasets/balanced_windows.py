"""
Load the balanced-window CHB-MIT dataset (built by
build_balanced_windows_dataset.py): 3 s windows, 1:1 seizure/normal per
patient, for leave-one-patient-out CV.

    from datasets import balanced_windows as bw

    X, y, patient = bw.load()                      # all patients
    X, y, patient = bw.load(exclude=["chb16"])     # subset
    X, y = bw.load_patient("chb01")                # one patient

    for test_p, train_idx, test_idx in bw.lopo_splits(patient):
        ...

X is returned as float32 (stored as float16), shape (n, 21, 768).
`patient` holds the patient name (e.g. "chb01") of every window.
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data" / "balanced_windows"
COMBINED_PATH = DATA_DIR / "chbmit_windows_all.npz"
PER_PATIENT_DIR = DATA_DIR / "per_patient"


def load_windows(path, dtype=np.float32):
    """Load a saved .npz as a dict, casting X to `dtype` (float16 on disk)."""
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["X"] = out["X"].astype(dtype)
    return out


def available_patients():
    """Patient names with a saved per-patient file."""
    return sorted(p.stem for p in PER_PATIENT_DIR.glob("chb*.npz"))


def load(patients=None, exclude=(), dtype=np.float32):
    """
    Load the combined dataset.

    Parameters
    ----------
    patients : list[str], optional
        Keep only these patients (default: all).
    exclude : list[str]
        Patients to drop.

    Returns
    -------
    X : (n, 21, 768) array, y : (n,) int8, patient : (n,) str
    """
    d = load_windows(COMBINED_PATH, dtype)
    patient = d["patients"][d["patient_id"]]

    keep = np.ones(len(patient), dtype=bool)
    if patients is not None:
        keep &= np.isin(patient, patients)
    if exclude:
        keep &= ~np.isin(patient, exclude)

    return d["X"][keep], d["y"][keep], patient[keep]


def load_patient(name, dtype=np.float32):
    """Load one patient's windows -> (X, y)."""
    d = load_windows(PER_PATIENT_DIR / f"{name}.npz", dtype)
    return d["X"], d["y"]


def lopo_splits(patient):
    """Yield (test_patient, train_idx, test_idx), one fold per patient."""
    patient = np.asarray(patient)
    for p in np.unique(patient):
        yield p, np.flatnonzero(patient != p), np.flatnonzero(patient == p)
