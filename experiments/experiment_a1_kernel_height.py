"""
Experiment A1: best kernel height for SpatialNet(spatial="height").
See experiment_a1_kernel_height.md.

Recording CV per patient (5 folds, leave-one-recording-out if fewer recordings),
train/val 80/20 stratified on the training recordings, focal loss.
Every (patient, window, kernel height, fold) row is checkpointed, so a rerun resumes.

    cd /home/gmarihuan/metapaper && python -m experiments.experiment_a1_kernel_height
    python -m experiments.experiment_a1_kernel_height --heights 16 21 --windows 1 3 5 8 10 \
        --out a1_kernel_height_h16_h21     # subfolder of experiments/results/
    python -m experiments.experiment_a1_kernel_height --heights 16 21 --windows 1 3 5 8 10 \
        --patients all --out a1_kernel_height_h16_h21_all
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from datasets import recording_cv as rcv
from experiments.training import (TrainConfig, set_seed, train_model, predict_proba,
                                  confusion_counts, add_metrics)
from models.spatial_net import SpatialNet

PATIENTS = ["chb01", "chb18", "chb06"]
WINDOWS = [1, 3, 5, 10]
KERNEL_HEIGHTS = [1, 4, 8, 16, 21]
N_SPLITS = 5
VAL_FRAC = 0.2
SEED = 0
CFG = TrainConfig()
EXCLUDE = {10: ["chb16"]}          # window_sec -> patients left with < 2 recordings

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
RESULTS_DIR = RESULTS_ROOT / "a1_kernel_height"


def result_paths(results_dir):
    """(partial checkpoint CSV, final CSV, report .md) inside results_dir."""
    return (results_dir / f"{results_dir.name}_folds_partial.csv",
            results_dir / f"{results_dir.name}_folds.csv",
            results_dir / "results.md")


def f1_table(mean, var, title, cols=KERNEL_HEIGHTS, label="h={}", bold_best=True,
             row_header="Window", row_label="{} s"):
    """Markdown table of mean ± variance: one row per index entry of `mean`
    (a window by default), one column per entry of `cols` (headed
    label.format(col)), best mean per row in bold."""
    lines = [f"**{title}**", "",
             f"| {row_header} | " + " | ".join(label.format(c) for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for w in mean.index:
        best = mean.loc[w].max()
        cells = []
        for c in cols:
            v = mean.loc[w, c]
            if pd.isna(v):                       # e.g. a patient excluded at this window
                cells.append("—")
                continue
            m =f"**{v:.3f}**" if bold_best and v == best else f"{v:.3f}"
            cells.append(f"{m} ± {var.loc[w, c]:.3f}")
        lines.append(f"| {row_label.format(w)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(df, results_dir=RESULTS_DIR):
    _, final_csv, report_md = result_paths(results_dir)
    heights = sorted(df["kernel_height"].unique())
    patients = list(dict.fromkeys(df["patient"]))         # order of first appearance
    by_patient = df.groupby(["window_sec", "kernel_height", "patient"])["f1"]
    fold_mean, fold_var = by_patient.mean(), by_patient.var()   # var over folds (ddof=1)
    by_window = fold_mean.groupby(["window_sec", "kernel_height"])
    parts = [
        "# Experiment A1: Kernel height results\n",
        f"`SpatialNet(spatial=\"height\", kernel_height=h)`, recording CV ({N_SPLITS} folds), "
        f"patients {', '.join(patients)}. See `experiments/experiment_a1_kernel_height.md`.\n",
        "Cells are F1 mean ± variance. Best mean per window in bold. "
        f"Per-fold rows with all metrics: `{final_csv.name}`.\n",
        "## Mean over patients\n",
        f1_table(by_window.mean().unstack(), by_window.var().unstack(),
                 "All patients (mean of the per-patient fold means ± variance across patients)",
                 cols=heights),
        "## Per patient\n",
    ]
    for p in patients:
        parts.append(f1_table(fold_mean.xs(p, level="patient").unstack(),
                              fold_var.xs(p, level="patient").unstack(),
                              f"{p} (mean ± variance over folds)", cols=heights))
    report_md.write_text("\n".join(parts))


def main(heights=KERNEL_HEIGHTS, windows=WINDOWS, results_dir=RESULTS_DIR, patients=PATIENTS):
    partial_csv, final_csv, _ = result_paths(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(partial_csv).to_dict("records") if partial_csv.exists() else []
    done = {(r["patient"], r["window_sec"], r["kernel_height"], r["fold"]) for r in rows}

    for window_sec in windows:
        for patient in patients:
            if patient in EXCLUDE.get(window_sec, []):
                continue
            X, y, rec = rcv.load_patient(patient, window_sec)
            folds = list(rcv.recording_folds(y, rec, N_SPLITS, seed=SEED))
            for fold, (tr, te) in enumerate(folds):
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X[tr], y[tr], test_size=VAL_FRAC, stratify=y[tr], random_state=SEED + fold)
                for h in heights:
                    if (patient, window_sec, h, fold) in done:
                        continue
                    t0 = time.time()
                    set_seed(SEED + fold)
                    model, best_epoch, _ = train_model(
                        SpatialNet(spatial="height", kernel_height=h),
                        X_tr, y_tr, X_val, y_val, CFG)
                    prob = predict_proba(model, X[te])
                    rows.append({
                        "patient": patient, "window_sec": window_sec, "kernel_height": h,
                        "fold": fold, "n_folds": len(folds),
                        "test_recordings": ";".join(sorted(set(rec[te]))),
                        "n_train": len(y_tr), "n_test": len(te), "best_epoch": best_epoch,
                        **confusion_counts(y[te], (prob >= 0.5).astype(int)),
                        "auroc": roc_auc_score(y[te], prob) if len(set(y[te])) == 2 else np.nan,
                    })
                    pd.DataFrame(rows).to_csv(partial_csv, index=False)   # checkpoint
                    print(f"{window_sec:>2}s {patient} h={h:<2} fold {fold}: "
                          f"F1={add_metrics(pd.DataFrame(rows[-1:]))['f1'].iloc[0]:.3f} "
                          f"({time.time() - t0:.0f}s)", flush=True)

    df = add_metrics(pd.DataFrame(rows))
    df.to_csv(final_csv, index=False)
    write_report(df, results_dir)

    # Mean F1 over folds, per patient, then mean over patients
    per_patient = df.groupby(["window_sec", "kernel_height", "patient"])["f1"].mean()
    print("\nMean F1 (folds averaged per patient, then across patients):")
    print(per_patient.groupby(["window_sec", "kernel_height"]).mean()
          .unstack("kernel_height").round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heights", type=int, nargs="+", default=KERNEL_HEIGHTS)
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    parser.add_argument("--patients", nargs="+", default=PATIENTS,
                        help="patient names, or 'all' for every patient with data")
    parser.add_argument("--out", default=RESULTS_DIR.name,
                        help="results subfolder name inside experiments/results/")
    args = parser.parse_args()
    patients = (rcv.available_patients(min(args.windows)) if args.patients == ["all"]
                else args.patients)
    main(args.heights, args.windows, RESULTS_ROOT / args.out, patients)
