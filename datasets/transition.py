"""
Load the transition dataset (built by build_transition_dataset.py) and cut it
into windows with a label per 1 s segment, including windows that cross
seizure onsets and offsets.

    from datasets import transition as tr

    d = tr.load_patient("chb01")
    for fold, (train_recs, test_recs) in enumerate(tr.split_folds(d, n_splits=5)):
        w = tr.fold_windows(d, train_recs, test_recs, window_sec=5, seed=fold)
        w["train"]["X"], w["train"]["Y"]      # (n, 21, 5*256), (n, 5) per-segment labels
        w["test"]["boundary"]                 # (n, 5) True where a segment is near a boundary

Protocol implemented by fold_windows:
  - train / val: stride-1 s windows. Validation is ~20 % of ~60 s time blocks
    of the training recordings (stratified by "block has seizure"); a window
    is used only if all its seconds fall in one set (no shared seconds).
  - test: recordings tiled with non-overlapping windows (stride = window), so
    every second is scored once. Tiles that touch a boundary are always kept.
  - every set keeps all windows with a seizure segment and caps all-normal
    windows at max_ratio x that count, per recording.

No MNE import: loading is fast.
"""

from pathlib import Path

import numpy as np

from datasets.recording_cv import recording_folds

SFREQ = 256
DATA_DIR = Path(__file__).resolve().parent / "data" / "transition"
PER_PATIENT_DIR = DATA_DIR / "per_patient"


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def available_patients():
    return sorted(p.stem for p in PER_PATIENT_DIR.glob("chb*.npz"))


def load_patient(name):
    """
    Load one patient -> dict with
        X (21, S*256) float16, sec_label (S,), rec_names (R,), rec_bounds (R, 2),
        sec_rec (S,) recording index of every second.
    """
    path = PER_PATIENT_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; build it with "
            f"`python datasets/build_transition_dataset.py --patients {name}`.")
    f = np.load(path, allow_pickle=True)
    d = {k: f[k] for k in f.files}
    lengths = d["rec_bounds"][:, 1] - d["rec_bounds"][:, 0]
    d["sec_rec"] = np.repeat(np.arange(len(lengths)), lengths)
    d["name"] = name
    return d


# ----------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------
def window_starts(rec_bounds, window_sec, stride=1, recordings=None):
    """Start seconds of windows lying entirely inside one recording.
    `recordings`: recording indices to use (default: all)."""
    recs = range(len(rec_bounds)) if recordings is None else recordings
    starts = [np.arange(rec_bounds[r, 0], rec_bounds[r, 1] - window_sec + 1, stride)
              for r in recs]
    return np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)


def cut_windows(X, sec_label, starts, window_sec, dtype=np.float32):
    """-> Xw (n, 21, window_sec*256), Yw (n, window_sec) per-segment labels, sec_idx (n, window_sec)."""
    sec_idx = starts[:, None] + np.arange(window_sec)            # (n, W) second of each segment
    X3 = X.reshape(X.shape[0], -1, SFREQ)                        # (21, S, 256) view
    Xw = X3[:, sec_idx, :]                                       # (21, n, W, 256)
    Xw = np.ascontiguousarray(Xw.transpose(1, 0, 2, 3)).reshape(len(starts), X.shape[0], -1)
    return Xw.astype(dtype), sec_label[sec_idx], sec_idx


def boundary_mask(sec_label, rec_bounds, k=2):
    """True for seconds within k s of an onset or offset (k seconds on each
    side of the change), inside the same recording."""
    mask = np.zeros(len(sec_label), dtype=bool)
    for s, e in rec_bounds:
        lab = sec_label[s:e]
        for t in np.flatnonzero(np.diff(lab) != 0) + 1:          # first second of the new state
            mask[s + max(0, t - k):s + min(e - s, t + k)] = True
    return mask


def cap_windows(Yw, rec_of_window, max_ratio=5, keep=None, seed=0):
    """
    Indices of windows to keep: every window with a seizure segment, plus
    all-normal windows up to max_ratio x the seizure-window count, per
    recording. `keep` (bool per window) forces windows to be kept; forced
    all-normal windows count toward the cap. Returned sorted.
    """
    rng = np.random.default_rng(seed)
    has_sz = Yw.any(axis=1)
    keep = np.zeros(len(Yw), dtype=bool) if keep is None else keep
    kept = []
    for r in np.unique(rec_of_window):
        in_r = rec_of_window == r
        forced = np.flatnonzero(in_r & (has_sz | keep))
        normal = np.flatnonzero(in_r & ~has_sz & ~keep)
        budget = max(0, max_ratio * int((in_r & has_sz).sum()) - int((in_r & ~has_sz & keep).sum()))
        chosen = rng.choice(normal, size=min(budget, len(normal)), replace=False)
        kept.append(np.concatenate([forced, chosen]))
    return np.sort(np.concatenate(kept)) if kept else np.zeros(0, dtype=np.int64)


# ----------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------
def split_folds(d, n_splits=5, seed=0):
    """Yield (train_recs, test_recs) recording indices, folds by recording
    (StratifiedGroupKFold on seconds; leave-one-recording-out if fewer
    recordings than n_splits)."""
    for tr, te in recording_folds(d["sec_label"], d["sec_rec"], n_splits, seed):
        yield np.unique(d["sec_rec"][tr]), np.unique(d["sec_rec"][te])


def time_block_split(sec_label, rec_bounds, train_recs, block_sec=60, val_frac=0.2, seed=0):
    """
    Split the seconds of the training recordings into contiguous blocks of
    ~block_sec and send ~val_frac of them to validation, separately for
    blocks with and without seizure (so both sets get seizure blocks when
    there are >= 2). Returns an int8 array per second: -1 = not a training
    recording, 0 = train, 1 = validation.
    """
    rng = np.random.default_rng(seed)
    split = np.full(len(sec_label), -1, dtype=np.int8)
    blocks = []                                                  # (start, end)
    for r in train_recs:
        s, e = rec_bounds[r]
        split[s:e] = 0
        blocks += [(b, min(b + block_sec, e)) for b in range(s, e, block_sec)]
    blocks = np.array(blocks)
    has_sz = np.array([sec_label[s:e].any() for s, e in blocks])

    for cls in (True, False):
        idx = np.flatnonzero(has_sz == cls)
        n_val = int(round(val_frac * len(idx)))
        if len(idx) >= 2:
            n_val = min(max(n_val, 1), len(idx) - 1)             # both sets get this class
        else:
            n_val = 0
        for b in rng.choice(idx, size=n_val, replace=False):
            split[blocks[b, 0]:blocks[b, 1]] = 1
    return split


def fold_windows(d, train_recs, test_recs, window_sec, stride=1, max_ratio=5,
                 block_sec=60, val_frac=0.2, boundary_k=2, seed=0):
    """
    Build train / val / test windows for one fold. Returns
        {"train": {...}, "val": {...}, "test": {...}}
    each with X (n, 21, W*256) float32, Y (n, W) int8, boundary (n, W) bool,
    starts (n,) start second.
    """
    X, sec_label, rec_bounds, sec_rec = d["X"], d["sec_label"], d["rec_bounds"], d["sec_rec"]
    bmask = boundary_mask(sec_label, rec_bounds, boundary_k)
    split = time_block_split(sec_label, rec_bounds, train_recs, block_sec, val_frac, seed)

    def make(starts, keep=None):
        sec_idx = starts[:, None] + np.arange(window_sec)
        Yw = sec_label[sec_idx]
        forced = None if keep is None else keep(sec_idx)
        idx = cap_windows(Yw, sec_rec[starts], max_ratio, forced, seed)
        starts = starts[idx]
        Xw, Yw, sec_idx = cut_windows(X, sec_label, starts, window_sec)
        return {"X": Xw, "Y": Yw, "boundary": bmask[sec_idx], "starts": starts}

    # train / val: stride-1 windows whose seconds are all in one set
    starts = window_starts(rec_bounds, window_sec, stride, train_recs)
    win_split = split[starts[:, None] + np.arange(window_sec)]
    out = {
        "train": make(starts[(win_split == 0).all(axis=1)]),
        "val": make(starts[(win_split == 1).all(axis=1)]),
        # test: non-overlapping tiles; tiles touching a boundary are always kept
        "test": make(window_starts(rec_bounds, window_sec, window_sec, test_recs),
                     keep=lambda sec_idx: bmask[sec_idx].any(axis=1)),
    }
    return out
