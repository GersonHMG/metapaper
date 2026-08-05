import numpy as np


def balance_dataset(X, y, method="undersample", seed=0):
    """
    Balance a windowed dataset so both classes have equal counts.

    Parameters
    ----------
    X : np.ndarray, shape (n_windows, n_channels, window_samples)
    y : np.ndarray, shape (n_windows,)   binary labels (0/1)
    method : {"undersample", "oversample"}
        "undersample" — randomly drop majority windows down to the minority
                        count (no duplicates, but discards data).
        "oversample"  — randomly duplicate minority windows up to the majority
                        count (keeps all data, but repeats minority windows).
    seed : int
        RNG seed for reproducible sampling.

    Returns
    -------
    X_bal, y_bal : balanced arrays, shuffled, same dtypes as inputs.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"Expected 2 classes, found {classes.tolist()}.")

    target = counts.min() if method == "undersample" else counts.max()
    if method not in ("undersample", "oversample"):
        raise ValueError("method must be 'undersample' or 'oversample'.")

    keep_idx = []
    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        # replace=True only matters for oversampling a minority class
        replace = method == "oversample" and len(cls_idx) < target
        chosen = rng.choice(cls_idx, size=target, replace=replace)
        keep_idx.append(chosen)

    keep_idx = rng.permutation(np.concatenate(keep_idx))   # shuffle classes together
    return X[keep_idx], y[keep_idx]



def segments_to_windows(
    X_segments: np.ndarray,
    y_segments: np.ndarray,
    window_samples: int,
    overlap_sec: float = 0.0,
    sfreq: float = 256.0,
    n_channels: int = 21,
    overlap_seizure_only: bool = False,
    seizure_label: int = 1,
):
    """
    Turn variable-length segments into a fixed-size 3D array of windows.

    Parameters
    ----------
    X_segments : np.ndarray, shape (n_segments,), dtype=object
        Output of load_patient_data — each element is a 1D vector of length
        (n_channels * n_times_i), channel-major (channel 0's samples first).
    y_segments : np.ndarray, shape (n_segments,)
        Per-segment labels (1 = seizure, 0 = normal).
    window_samples : int
        Window length in TIME samples (e.g. 256*4 = 1024 for 4 s at 256 Hz).
    overlap_sec : float
        Overlap between consecutive windows, in SECONDS. 0.0 = no overlap.
        Must be less than the window length in seconds.
    sfreq : float
        Sampling frequency in Hz, used to convert overlap_sec to samples.
    n_channels : int
        Channels per segment, needed to un-flatten. Defaults to len(REQUIRED_CHANNELS).
    overlap_seizure_only : bool
        If True, apply `overlap_sec` only to seizure segments; normal segments
        are windowed with no overlap. Useful to up-sample the minority (seizure)
        class without inflating the majority class. Default False (overlap
        applied to all segments).
    seizure_label : int
        Label value that marks a seizure segment. Default 1.

    Returns
    -------
    X : np.ndarray, shape (n_windows, n_channels, window_samples), float
    y : np.ndarray, shape (n_windows,), int8
        Each window inherits the label of the segment it came from.
    """
    if window_samples <= 0:
        raise ValueError("window_samples must be positive.")

    overlap_samples = int(round(overlap_sec * sfreq))
    if overlap_samples < 0:
        raise ValueError("overlap_sec must be non-negative.")
    if overlap_samples >= window_samples:
        raise ValueError(
            f"overlap ({overlap_samples} samples) must be smaller than "
            f"window_samples ({window_samples})."
        )

    # hop between window starts: with overlap vs. without
    overlap_step = window_samples - overlap_samples
    no_overlap_step = window_samples

    X_windows, y_windows = [], []

    for flat, label in zip(X_segments, y_segments):
        # un-flatten (channels × time) → (n_channels, n_times)
        seg = np.asarray(flat).reshape(n_channels, -1)
        n_times = seg.shape[1]

        # too short for even one window → skip
        if n_times < window_samples:
            continue

        # pick the hop for THIS segment
        if overlap_seizure_only and label != seizure_label:
            step = no_overlap_step       # normal segment: no overlap
        else:
            step = overlap_step          # seizure (or overlap-for-all): use overlap

        for start in range(0, n_times - window_samples + 1, step):
            X_windows.append(seg[:, start:start + window_samples])
            y_windows.append(label)

    if not X_windows:
        raise ValueError(
            "No windows produced — every segment was shorter than "
            f"window_samples ({window_samples})."
        )

    X = np.stack(X_windows, axis=0).astype(np.float64)   # (n_windows, n_channels, window)
    y = np.asarray(y_windows, dtype=np.int8)
    return X, y