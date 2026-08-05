"""
Build the windowed, class-balanced CHB-MIT dataset for all patients and
save it to disk.

Per patient it runs:
    load_patient_data -> segments_to_windows(..., 256*3) -> balance_dataset
Saves each patient individually AND a combined dataset with a patient_id
array (needed for leave-one-patient-out cross-validation).

Windows are stored as float16 to save space; cast back to float32 when
loading for training (see load helper at the bottom).
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.append("..")          # make `datasets` importable

from datasets.data_loader import load_patient_data          # noqa: E402
from datasets.preprocessing import segments_to_windows, balance_dataset  # noqa: E402

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
WINDOW_LEN = 256 * 3                 # 768 samples per window
N_PATIENTS = 24                      # chb01 .. chb24
PATIENTS = [f"chb{i:02d}" for i in range(1, N_PATIENTS + 1)]

OUT_DIR = Path("./processed")                    # <- Path, not str
PER_PATIENT_DIR = OUT_DIR / "per_patient"


def process_patient(name: str):
    """Return (X, y): balanced (n, 21, 768) windows, X as float16."""
    X, y = load_patient_data(name, progress=False)
    X, y = segments_to_windows(X, y, WINDOW_LEN)
    X, y = balance_dataset(X, y)

    X = np.asarray(X, dtype=np.float16)          # storage precision
    y = np.asarray(y).ravel().astype(np.int8)
    return X, y


def main():
    PER_PATIENT_DIR.mkdir(parents=True, exist_ok=True)

    all_X, all_y, all_pid = [], [], []
    summary = []

    for pid, name in enumerate(PATIENTS):
        t0 = time.time()
        try:
            X, y = process_patient(name)
        except Exception as exc:                 # keep going if one fails
            print(f"[SKIP] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            summary.append((name, "FAILED", 0, 0, 0))
            continue

        np.savez_compressed(PER_PATIENT_DIR / f"{name}.npz", X=X, y=y)

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        print(f"[ OK ] {name}: {X.shape[0]:6d} windows "
              f"(seizure={n_pos}, normal={n_neg})  "
              f"shape={tuple(X.shape[1:])}  {time.time() - t0:.1f}s")

        all_X.append(X)
        all_y.append(y)
        all_pid.append(np.full(X.shape[0], pid, dtype=np.int16))
        summary.append((name, "ok", X.shape[0], n_pos, n_neg))

    if not all_X:
        raise RuntimeError("No patients were processed successfully.")

    # Combined dataset with patient ids for leave-one-patient-out.
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    pid_all = np.concatenate(all_pid, axis=0)

    np.savez_compressed(
        OUT_DIR / "chbmit_windows_all.npz",
        X=X_all, y=y_all, patient_id=pid_all,
        patients=np.array(PATIENTS),
        window_len=WINDOW_LEN,
    )

    # ---- report ------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'patient':10s} {'status':8s} {'windows':>8s} {'seizure':>8s} {'normal':>8s}")
    for name, status, n, pos, neg in summary:
        print(f"{name:10s} {status:8s} {n:>8d} {pos:>8d} {neg:>8d}")
    print("-" * 60)
    print(f"TOTAL: {X_all.shape[0]} windows, shape per window {tuple(X_all.shape[1:])}")
    print(f"Saved combined -> {OUT_DIR / 'chbmit_windows_all.npz'}")
    print(f"Saved per-patient -> {PER_PATIENT_DIR}/")


def load_windows(path):
    """Load a saved .npz and cast X back to float32 for training."""
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["X"] = out["X"].astype(np.float32)       # float16 -> float32
    return out


if __name__ == "__main__":
    main()