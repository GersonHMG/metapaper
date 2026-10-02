"""
Build the transition dataset: every seizure recording of each patient stored
as ONE continuous signal with a label per second, for experiments that label
each 1 s segment (normal / seizure), including windows that cross seizure
onsets and offsets.

Per patient:
    load_seizure_recordings -> trim each recording to whole seconds
    -> sec_label[t] = 1 if start <= t < end (annotations are integer seconds,
       so a 1 s segment is never half seizure) -> concatenate recordings

Output: data/transition/per_patient/chbXX.npz with
    X          (21, S*256) float16   continuous signal, microvolts
    sec_label  (S,) int8             label of every second
    rec_names  (R,) str              EDF filename of each recording
    rec_bounds (R, 2) int64          [start, end) second of each recording in X
    sfreq      int

Windows are NOT stored: cut them at load time with datasets/transition.py,
so one dataset serves every window length and stride.

Usage:
    python build_transition_dataset.py                # all patients
    python build_transition_dataset.py --patients chb01 chb02
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# put metapaper/ first so it wins over the pip `datasets` (HuggingFace) package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.data_loader import load_seizure_recordings   # noqa: E402
# paths and loading helpers live in the loader module (re-exported here)
from datasets.transition import SFREQ, DATA_DIR as OUT_DIR, PER_PATIENT_DIR  # noqa: E402,F401

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
N_PATIENTS = 24                      # chb01 .. chb24
PATIENTS = [f"chb{i:02d}" for i in range(1, N_PATIENTS + 1)]


def process_patient(name: str):
    """Return (X, sec_label, rec_names, rec_bounds) for one patient."""
    X_parts, label_parts, names, bounds = [], [], [], []
    cursor = 0
    for rec_name, data, sfreq, starts, ends in load_seizure_recordings(name):
        if int(sfreq) != SFREQ:
            raise ValueError(f"{rec_name}: sfreq {sfreq} != {SFREQ}")
        n_sec = data.shape[1] // SFREQ
        X_parts.append(data[:, :n_sec * SFREQ].astype(np.float16))

        label = np.zeros(n_sec, dtype=np.int8)
        for s, e in zip(starts, ends):
            label[max(0, int(s)):min(n_sec, int(e))] = 1
        label_parts.append(label)

        names.append(rec_name)
        bounds.append((cursor, cursor + n_sec))
        cursor += n_sec

    if not X_parts:
        raise ValueError(f"No seizure recordings with the required channels for {name}.")

    X = np.concatenate(X_parts, axis=1)
    sec_label = np.concatenate(label_parts)
    return X, sec_label, np.array(names), np.array(bounds, dtype=np.int64)


def main(patients=PATIENTS):
    PER_PATIENT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for name in patients:
        t0 = time.time()
        try:
            X, sec_label, rec_names, rec_bounds = process_patient(name)
        except Exception as exc:                 # keep going if one fails
            print(f"[SKIP] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            summary.append((name, "FAILED", 0, 0, 0))
            continue

        np.savez(PER_PATIENT_DIR / f"{name}.npz", X=X, sec_label=sec_label,
                 rec_names=rec_names, rec_bounds=rec_bounds, sfreq=SFREQ)

        n_sec, n_sz = len(sec_label), int(sec_label.sum())
        print(f"[ OK ] {name}: {len(rec_names):2d} recordings, {n_sec / 3600:5.1f} h, "
              f"seizure={n_sz} s  {time.time() - t0:.1f}s")
        summary.append((name, "ok", len(rec_names), n_sec, n_sz))

    # ---- report ------------------------------------------------------
    print("\n" + "=" * 56)
    print(f"{'patient':10s} {'status':8s} {'recs':>5s} {'hours':>8s} {'seizure_s':>10s}")
    for name, status, nrec, n_sec, n_sz in summary:
        print(f"{name:10s} {status:8s} {nrec:>5d} {n_sec / 3600:>8.1f} {n_sz:>10d}")
    print("-" * 56)
    print(f"Saved per-patient -> {PER_PATIENT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patients", nargs="+", default=PATIENTS)
    main(parser.parse_args().patients)
