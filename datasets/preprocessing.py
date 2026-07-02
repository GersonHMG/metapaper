import json
import warnings
import numpy as np
import mne
from pathlib import Path

BASE_DIR  = Path("/home/gmarihuan/chbmit")
JSON_PATH = BASE_DIR / "chbmit_summary.json"

# MNE's get_data() returns volts (~1e-5). The network's BatchNorm uses
# eps=1e-5, so volts-scale inputs sit below eps and never normalize -> the
# loss locks at ln(2). Converting to microvolts (x1e6) puts the signal at
# ~tens of uV, well above eps, and preserves cross-window amplitude (the
# ictal vs. inter-ictal amplitude difference). No data statistics are used,
# so this introduces no train/test leakage.
VOLTS_TO_MICROVOLTS = 1e6

REQUIRED_CHANNELS = [
    "FP1-F7", "F7-T7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8",
]


def _load_summary() -> dict:
    with open(JSON_PATH) as f:
        return json.load(f)


def _read_required(filepath: Path, lowpass_hz: float | None = 64.0) -> mne.io.BaseRaw:
    """Read an EDF, collapse the duplicate T8-P8 name, keep+reorder required
    channels, and (optionally) apply the paper's lowpass filter.

    lowpass_hz : float or None
        Cutoff for the lowpass filter in Hz. The paper (Section III-A-1)
        filters out noise above 64 Hz. Pass None to skip filtering (e.g. to
        isolate the effect of the microvolt rescaling).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Channel names are not unique",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
    if "T8-P8-0" in raw.ch_names:
        raw.rename_channels({"T8-P8-0": "T8-P8"})
    raw.pick(REQUIRED_CHANNELS)   # selects + reorders to match REQUIRED_CHANNELS

    # Paper preprocessing: lowpass < 64 Hz. Done on the full recording before
    # segmentation so the filter has the whole signal to work with.
    if lowpass_hz is not None:
        raw.filter(l_freq=None, h_freq=lowpass_hz, verbose=False)
    return raw


def _channel_names(filepath: Path) -> list[str]:
    """Return channel names from an EDF file (after collapsing the duplicate)."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Channel names are not unique",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
    if "T8-P8-0" in raw.ch_names:
        raw.rename_channels({"T8-P8-0": "T8-P8"})
    return raw.ch_names


def _norm(ch: str) -> str:
    # collapse en-dash (–) and em-dash (—) to a plain hyphen, then strip/upper
    return ch.strip().upper().replace("\u2013", "-").replace("\u2014", "-")


def filter_files_with_channels(file_list, required=REQUIRED_CHANNELS):
    required_set = {_norm(c) for c in required}
    kept = []
    for f in file_list:
        channels = {_norm(c) for c in _channel_names(f)}
        if required_set.issubset(channels):
            kept.append(f)
    return kept


def _extract_seizure_times(seizure_info: dict):
    """
    Pull seizure start/end times (in seconds) out of a summary file-entry.

    Expected JSON shape:
        "chb01_03.edf": {
            ...,
            "seizures": {"1": {"start_sec": 2996, "end_sec": 3036}, ...}
        }
    """
    starts, ends = [], []
    if not seizure_info:
        return starts, ends

    seizures = seizure_info.get("seizures", {})
    for key in sorted(seizures, key=lambda k: int(k)):
        ev = seizures[key]
        starts.append(float(ev["start_sec"]))
        ends.append(float(ev["end_sec"]))

    return starts, ends


def split_seizure_segments(data, sfreq, seizure_starts, seizure_ends):
    """
    Split already-loaded EEG data into seizure and normal segments by time.

    Returns
    -------
    seizure_segments : list[np.ndarray]   one (n_channels, n_times) per seizure
    normal_segments  : list[np.ndarray]   gaps before/between/after seizures
    """
    if len(seizure_starts) != len(seizure_ends):
        raise ValueError("seizure_starts and seizure_ends must be the same length.")

    n_times = data.shape[1]

    intervals = sorted(
        (int(round(s * sfreq)), int(round(e * sfreq)))
        for s, e in zip(seizure_starts, seizure_ends)
    )
    intervals = [(max(0, s), min(n_times, e)) for s, e in intervals]

    seizure_segments, normal_segments = [], []
    cursor = 0
    for s, e in intervals:
        if s > cursor:
            normal_segments.append(data[:, cursor:s])
        seizure_segments.append(data[:, s:e])
        cursor = e
    if cursor < n_times:
        normal_segments.append(data[:, cursor:n_times])

    return seizure_segments, normal_segments


def load_patient_data(patient_name: str, base_dir: str = str(BASE_DIR),
                      normal_from_seizure_only: bool = False,
                      lowpass_hz: float | None = 64.0,
                      to_microvolts: bool = True):
    """
    Load and label all EEG data for one patient as per-segment flattened vectors.

    Parameters
    ----------
    normal_from_seizure_only : bool, default False
        If True, drop every fully-normal (non-seizure) recording and keep only
        the segments carved out of seizure recordings. The seizure (label 1)
        segments ARE still returned; only the standalone normal recordings are
        discarded. Result: a mix of label 0 (inter-ictal gaps) and label 1
        (seizure) windows, with the majority-class normal recordings removed.
    lowpass_hz : float or None, default 64.0
        Lowpass cutoff in Hz applied per recording (paper Section III-A-1).
        Pass None to skip filtering.
    to_microvolts : bool, default True
        Multiply the volts-scale data from MNE by 1e6 -> microvolts. Required
        for the model to train (see VOLTS_TO_MICROVOLTS note above). Set False
        only if your raw data is already in non-volt units.

    Returns
    -------
    X : np.ndarray, shape (n_segments,), dtype=object
    y : np.ndarray, shape (n_segments,), dtype int8
    """
    summary         = _load_summary()
    patient_summary = summary.get(patient_name, {})

    patient_dir = Path(base_dir) / patient_name
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    all_edfs = sorted(patient_dir.glob("*.edf"))

    def _has_seizures(name: str) -> bool:
        info = patient_summary.get(name)
        return bool(info and info.get("seizures"))

    seizure_edfs = [f for f in all_edfs if _has_seizures(f.name)]
    normal_edfs  = [f for f in all_edfs if not _has_seizures(f.name)]

    seizure_edfs = filter_files_with_channels(seizure_edfs, REQUIRED_CHANNELS)
    normal_edfs  = filter_files_with_channels(normal_edfs,  REQUIRED_CHANNELS)

    scale = VOLTS_TO_MICROVOLTS if to_microvolts else 1.0

    X_segments: list[np.ndarray] = []
    y_labels:   list[int]        = []

    for filepath in seizure_edfs:
        raw   = _read_required(filepath, lowpass_hz=lowpass_hz)
        data  = raw.get_data() * scale          # (n_channels, n_times), microvolts
        sfreq = raw.info["sfreq"]

        starts, ends = _extract_seizure_times(patient_summary.get(filepath.name))
        seiz_segs, norm_segs = split_seizure_segments(data, sfreq, starts, ends)

        for seg in seiz_segs:
            X_segments.append(seg.reshape(-1))   # flatten (channels × time)
            y_labels.append(1)

        for seg in norm_segs:
            X_segments.append(seg.reshape(-1))
            y_labels.append(0)

    # Skip the fully-normal recordings entirely when we only want the
    # inter-ictal segments carved out of seizure files.
    if not normal_from_seizure_only:
        for filepath in normal_edfs:
            raw  = _read_required(filepath, lowpass_hz=lowpass_hz)
            data = raw.get_data() * scale            # one normal segment per file
            X_segments.append(data.reshape(-1))
            y_labels.append(0)

    if not X_segments:
        raise ValueError(
            f"No segments produced for patient '{patient_name}' "
            f"(check required channels / normal_from_seizure_only setting)."
        )

    # Object array because segments have different lengths.
    X = np.empty(len(X_segments), dtype=object)
    for i, seg in enumerate(X_segments):
        X[i] = seg
    y = np.asarray(y_labels, dtype=np.int8)

    return X, y