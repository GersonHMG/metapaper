"""
Build the continuous dataset: EVERY recording of each patient (with and without
seizures) as one continuous signal with a label per second, for long-context
temporal models and event-level evaluation (false alarms per hour, latency).

Same per-second labelling as build_transition_dataset.py, but seizure-free
recordings are kept too. Recordings missing a required channel are skipped.

Output: data/continuous/per_patient/chbXX/
    X.npy      (21, S*256) float16   continuous signal, microvolts (load with mmap)
    meta.npz   sec_label (S,) int8, rec_names (R,), rec_bounds (R, 2) [start, end)
               seconds, rec_has_seizure (R,) bool, sfreq

Usage:
    python build_continuous_dataset.py                       # all patients
    python build_continuous_dataset.py --patients chb01 --jobs 1
Patients whose meta.npz exists are skipped (delete the folder to rebuild).
"""

import argparse
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# put metapaper/ first so it wins over the pip `datasets` (HuggingFace) package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.data_loader import iter_recordings            # noqa: E402
from datasets.continuous import SFREQ, PER_PATIENT_DIR      # noqa: E402

PATIENTS = [f"chb{i:02d}" for i in range(1, 25)]


def process_patient(name: str):
    out = PER_PATIENT_DIR / name
    if (out / "meta.npz").exists():
        return name, "exists", 0, 0, 0, 0.0
    t0 = time.time()
    try:
        X_parts, label_parts, names, bounds, has_sz = [], [], [], [], []
        cursor = 0
        for rec_name, data, sfreq, starts, ends in iter_recordings(name):
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
            has_sz.append(bool(label.any()))
            cursor += n_sec
            del data

        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "X.npy", np.concatenate(X_parts, axis=1))
        del X_parts
        sec_label = np.concatenate(label_parts)
        # meta.npz last: its presence marks a complete patient
        np.savez(out / "meta.npz", sec_label=sec_label, rec_names=np.array(names),
                 rec_bounds=np.array(bounds, dtype=np.int64),
                 rec_has_seizure=np.array(has_sz), sfreq=SFREQ)
        return name, "ok", len(names), len(sec_label), int(sec_label.sum()), time.time() - t0
    except Exception as exc:                     # keep going if one fails
        traceback.print_exc()
        return name, f"FAILED {type(exc).__name__}: {exc}", 0, 0, 0, time.time() - t0


def main(patients=PATIENTS, jobs=6):
    PER_PATIENT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    with Pool(jobs) as pool:
        for name, status, nrec, n_sec, n_sz, dt in pool.imap_unordered(process_patient, patients):
            print(f"[{status[:6]:>6}] {name}: {nrec:3d} recordings, {n_sec / 3600:6.1f} h, "
                  f"seizure={n_sz} s  {dt:.0f}s", flush=True)
            summary.append((name, status, nrec, n_sec, n_sz))

    print("\n" + "=" * 60)
    print(f"{'patient':10s} {'status':8s} {'recs':>5s} {'hours':>8s} {'seizure_s':>10s}")
    for name, status, nrec, n_sec, n_sz in sorted(summary):
        print(f"{name:10s} {status[:8]:8s} {nrec:>5d} {n_sec / 3600:>8.1f} {n_sz:>10d}")
    print("-" * 60)
    print(f"Saved per-patient -> {PER_PATIENT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patients", nargs="+", default=PATIENTS)
    parser.add_argument("--jobs", type=int, default=6, help="patients built in parallel")
    args = parser.parse_args()
    main(args.patients, args.jobs)
