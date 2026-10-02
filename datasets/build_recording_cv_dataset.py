"""
Build a windowed CHB-MIT dataset for within-patient, recording-level
cross-validation (leave-one-recording-out, or k-fold grouped by recording).

Only recordings that contain seizures are used. Per patient it runs:
    load_patient_data(only_seizure_files=True) -> segments_to_windows(...)
    -> cap_class_ratio(max_ratio=5) applied per recording
so every recording keeps at most 5 normal windows per seizure window.

One dataset is built per window length in WINDOW_SECONDS (1, 3, 5, 8, 10 s); the
EDFs are read once per patient and windowed at every length. Recordings whose
seizures are all shorter than the window yield no seizure windows and are
dropped for that window length.

Output: data/recording_cv/win{N}s/per_patient/chbXX.npz, one file per patient
with a `recording` array (EDF filename per window).

Usage:
    python build_recording_cv_dataset.py              # all window lengths
    python build_recording_cv_dataset.py --windows 3 5

Load it with datasets/recording_cv.py (load_patient, iter_folds,
recording_folds); balancing the training fold is left to the training code.

Windows are stored as float16 to save space; cast back to float32 when
loading for training (the loader does this).
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# put metapaper/ first so it wins over the pip `datasets` (HuggingFace) package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.data_loader import load_patient_data          # noqa: E402
from datasets.preprocessing import segments_to_windows, cap_class_ratio  # noqa: E402
# paths and loading helpers live in the loader module (re-exported here)
from datasets.recording_cv import (SFREQ, WINDOW_SECONDS, DATA_DIR as OUT_DIR,  # noqa: E402,F401
                                   per_patient_dir, load_windows, recording_folds)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
MAX_RATIO = 5                        # at most 5 normal windows per seizure window
N_PATIENTS = 24                      # chb01 .. chb24
PATIENTS = [f"chb{i:02d}" for i in range(1, N_PATIENTS + 1)]


def load_segments(name: str):
    """Read a patient's seizure recordings once -> (X_seg, y_seg, rec_seg)."""
    return load_patient_data(name, only_seizure_files=True,
                             return_recordings=True, progress=False)


def process_patient(name: str, window_sec: int = 3, seed: int = 0, segments=None):
    """Return (X, y, recording, dropped): (n, 21, window_sec*SFREQ) windows as
    float16, capped at 1:MAX_RATIO seizure:normal within each recording.
    `dropped` lists recordings with no seizure window at this length.
    Pass `segments` (from load_segments) to avoid re-reading the EDFs."""
    X, y, rec = segments if segments is not None else load_segments(name)
    X, y, rec = segments_to_windows(X, y, window_sec * SFREQ, groups=rec)

    X_parts, y_parts, rec_parts, dropped = [], [], [], []
    for r in np.unique(rec):                     # cap the ratio per recording
        m = rec == r
        if not (y[m] == 1).any():                # seizures shorter than the window
            dropped.append(str(r))
            continue
        Xr, yr, recr = cap_class_ratio(X[m], y[m], rec[m],
                                       max_ratio=MAX_RATIO, seed=seed)
        X_parts.append(Xr)
        y_parts.append(yr)
        rec_parts.append(recr)

    if not X_parts:
        raise ValueError(f"No recording has a seizure window at {window_sec} s.")

    X = np.concatenate(X_parts, axis=0).astype(np.float16)   # storage precision
    y = np.concatenate(y_parts).astype(np.int8)
    rec = np.concatenate(rec_parts)
    return X, y, rec, dropped


def main(window_secs=WINDOW_SECONDS):
    for w in window_secs:
        per_patient_dir(w).mkdir(parents=True, exist_ok=True)

    summary = {w: [] for w in window_secs}

    for name in PATIENTS:
        try:
            segments = load_segments(name)       # read the EDFs once
        except Exception as exc:                 # keep going if one fails
            print(f"[SKIP] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            for w in window_secs:
                summary[w].append((name, "FAILED", 0, 0, 0, 0, 0))
            continue

        for w in window_secs:
            t0 = time.time()
            try:
                X, y, rec, dropped = process_patient(name, w, segments=segments)
            except Exception as exc:
                print(f"[SKIP] {name} {w}s: {type(exc).__name__}: {exc}")
                summary[w].append((name, "FAILED", 0, 0, 0, 0, 0))
                continue

            np.savez_compressed(per_patient_dir(w) / f"{name}.npz",
                                X=X, y=y, recording=rec,
                                window_len=w * SFREQ, window_sec=w)

            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            n_rec = len(np.unique(rec))
            print(f"[ OK ] {name} {w:2d}s: {X.shape[0]:6d} windows "
                  f"(seizure={n_pos}, normal={n_neg}, recordings={n_rec}"
                  f"{f', dropped={dropped}' if dropped else ''})  "
                  f"{time.time() - t0:.1f}s")
            summary[w].append((name, "ok", X.shape[0], n_pos, n_neg, n_rec, len(dropped)))
        del segments

    # ---- report ------------------------------------------------------
    for w in window_secs:
        print("\n" + "=" * 66)
        print(f"window = {w} s ({w * SFREQ} samples)")
        print(f"{'patient':10s} {'status':8s} {'windows':>8s} {'seizure':>8s} "
              f"{'normal':>8s} {'recs':>5s} {'dropped':>8s}")
        for name, status, n, pos, neg, nrec, ndrop in summary[w]:
            print(f"{name:10s} {status:8s} {n:>8d} {pos:>8d} {neg:>8d} "
                  f"{nrec:>5d} {ndrop:>8d}")
        print("-" * 66)
        print(f"Saved per-patient -> {per_patient_dir(w)}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOW_SECONDS,
                        help="window lengths in seconds (default: 1 3 5 8 10)")
    main(parser.parse_args().windows)
