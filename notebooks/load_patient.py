import json
import numpy as np
import mne
from pathlib import Path

BASE_DIR  = Path("/home/gmarihuan/chbmit")
JSON_PATH = BASE_DIR / "chbmit_summary.json"


def _load_summary() -> dict:
    with open(JSON_PATH) as f:
        return json.load(f)


def get_patient_files(patient_name: str, base_dir: Path = BASE_DIR):
    """
    Classify a patient's EDF files into seizure and normal groups.

    The presence of an *.edf.seizures file is used as the seizure marker —
    the seizures file itself is NOT read (it is not EDF format).
    Instead the matching base *.edf file is returned as the seizure recording.

    Rules
    -----
    - If *.edf.seizures exists  →  the base *.edf is a seizure recording  (label 1)
    - If no *.edf.seizures      →  the *.edf is a normal recording         (label 0)

    Parameters
    ----------
    patient_name : str   e.g. "chb16"
    base_dir     : Path  root directory containing patient folders

    Returns
    -------
    seizure_edfs : list[Path]   base .edf files that have a .seizures marker
    normal_edfs  : list[Path]   base .edf files with no .seizures marker
    """
    patient_dir = Path(base_dir) / patient_name
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    # Names of .edf files that have a seizure marker file
    seizure_edf_names = {
        f.name.removesuffix(".seizures")
        for f in patient_dir.glob("*.edf.seizures")
    }

    all_edfs     = sorted(patient_dir.glob("*.edf"))
    seizure_edfs = [f for f in all_edfs if f.name in seizure_edf_names]
    normal_edfs  = [f for f in all_edfs if f.name not in seizure_edf_names]

    return seizure_edfs, normal_edfs


def _channel_names(filepath: Path) -> list[str]:
    """Return channel names from an EDF file without loading signal data."""
    raw = mne.io.read_raw_edf(str(filepath), preload=False, verbose=False)
    return raw.ch_names


def _common_channels(file_list: list[Path]) -> list[str]:
    """
    Return channels present in every file, preserving the order of the first file.
    Handles patients whose montage changes across sessions.
    """
    if not file_list:
        return []
    first   = _channel_names(file_list[0])
    common  = set(first)
    for f in file_list[1:]:
        common &= set(_channel_names(f))
    return [ch for ch in first if ch in common]


def _label_samples(n_times: int, sfreq: float, seizure_info: dict | None) -> np.ndarray:
    """
    Build a per-sample label array for one EDF file.

    Parameters
    ----------
    n_times      : total number of time samples in the file
    sfreq        : sampling frequency (Hz)
    seizure_info : entry from chbmit_summary.json for this file, or None

    Returns
    -------
    y : np.ndarray shape (n_times,) dtype int8   1=seizure  0=normal
    """
    y = np.zeros(n_times, dtype=np.int8)

    if seizure_info:
        for sz in seizure_info["seizures"].values():
            start = int(sz["start_sec"] * sfreq)
            end   = int(sz["end_sec"]   * sfreq)
            # clamp to valid range
            start = max(0, min(start, n_times))
            end   = max(0, min(end,   n_times))
            y[start:end] = 1

    return y


def load_patient_data(patient_name: str, base_dir: str = str(BASE_DIR)):
    """
    Load and label all EEG data for one patient using MNE.

    Seizure files: reads the base .edf and applies sample-accurate seizure
    labels from chbmit_summary.json (only the annotated windows are label 1;
    the rest of the recording remains label 0).

    Normal files: reads the .edf and labels every sample 0.

    Parameters
    ----------
    patient_name : str   e.g. "chb16"
    base_dir     : str   root directory containing patient folders

    Returns
    -------
    X : np.ndarray, shape (n_channels, n_total_samples)
        All recordings concatenated along the time axis.
    y : np.ndarray, shape (n_total_samples,), dtype int8
        Per-sample label — 1 = seizure, 0 = normal.
    """
    summary         = _load_summary()
    patient_summary = summary.get(patient_name, {})

    seizure_edfs, normal_edfs = get_patient_files(patient_name, Path(base_dir))

    all_edfs    = seizure_edfs + normal_edfs
    common_chs  = _common_channels(all_edfs)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for filepath in seizure_edfs:
        raw   = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
        raw.pick_channels(common_chs)
        data  = raw.get_data()                        # (n_channels, n_times)
        sfreq = raw.info["sfreq"]
        seizure_info = patient_summary.get(filepath.name)
        y = _label_samples(data.shape[1], sfreq, seizure_info)
        X_parts.append(data)
        y_parts.append(y)

    for filepath in normal_edfs:
        raw  = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
        raw.pick_channels(common_chs)
        data = raw.get_data()
        X_parts.append(data)
        y_parts.append(np.zeros(data.shape[1], dtype=np.int8))

    if not X_parts:
        raise ValueError(f"No EDF files found for patient '{patient_name}'.")

    X = np.concatenate(X_parts, axis=1)   # (n_channels, total_samples)
    y = np.concatenate(y_parts)            # (total_samples,)

    return X, y


if __name__ == "__main__":
    patient = "chb16"
    print(f"Loading data for {patient} ...")

    seizure_edfs, normal_edfs = get_patient_files(patient)
    print(f"  Seizure EDF files : {len(seizure_edfs)}")
    for f in seizure_edfs:
        print(f"    {f.name}")
    print(f"  Normal  EDF files : {len(normal_edfs)}")

    X, y = load_patient_data(patient)
    print(f"\nX shape : {X.shape}   (channels × samples)")
    print(f"y shape : {y.shape}")
    print(f"Seizure samples : {y.sum():,}  ({100 * y.mean():.2f} %)")
    print(f"Normal  samples : {(y == 0).sum():,}  ({100 * (1 - y.mean()):.2f} %)")
