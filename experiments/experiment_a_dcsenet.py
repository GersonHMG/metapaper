"""
DCSENet baseline (Aboyeji et al. 2025) on the recording-CV data, compared with
the SpatialNet runs of Experiment A. See experiment_b1_dcsenet.md for the model
choices (one spectrogram image per channel, mean of the 21 channel
probabilities per window; Adam 1e-3, batch 32, 20 epochs; focal loss, best
epoch on validation loss).

Protocol identical to experiment_a1_kernel_height.py: recording_folds (5 folds,
leave-one-recording-out if fewer recordings), 80/20 stratified train/val split
of the training recordings, test keeps its <= 1:5 ratio. One output per window.

    cd /home/gmarihuan/metapaper && python -m experiments.experiment_a_dcsenet --patients chb01
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from datasets import recording_cv as rcv
from experiments.experiment_a1_kernel_height import (N_SPLITS, VAL_FRAC, SEED, EXCLUDE,
                                                     RESULTS_ROOT, result_paths)
from experiments.experiment_b1_dcsenet import train_dcsenet, image_proba
from experiments.training import DEVICE, confusion_counts, add_metrics

WINDOWS = [1, 3, 5, 8, 10]
RESULTS_DIR = RESULTS_ROOT / "a_dcsenet"
# SpatialNet runs to compare with: (results folder, column naming the model, prefix)
SPATIALNET_RUNS = [
    ("a_grouped_all", None, "grouped"),
    ("a1_kernel_height_h1_all", "kernel_height", "h="),
    ("a1_kernel_height", "kernel_height", "h="),
    ("a1_kernel_height_h16_h21_all", "kernel_height", "h="),
]


def load_spatialnet(patients, windows):
    """SpatialNet fold rows for the same patients/windows, one `model` column.
    Earlier runs that repeat a (model, patient, window, fold) are dropped."""
    frames = []
    for folder, col, prefix in SPATIALNET_RUNS:
        csv = RESULTS_ROOT / folder / f"{folder}_folds.csv"
        if not csv.exists():
            continue
        d = pd.read_csv(csv)
        d["model"] = prefix if col is None else prefix + d[col].astype(str)
        frames.append(d)
    d = pd.concat(frames)
    d = d[d["patient"].isin(patients) & d["window_sec"].isin(windows)]
    d = d.drop_duplicates(["model", "patient", "window_sec", "fold"])
    return add_metrics(d)


def write_report(df, results_dir=RESULTS_DIR):
    _, final_csv, report_md = result_paths(results_dir)
    patients = list(dict.fromkeys(df["patient"]))
    windows = sorted(df["window_sec"].unique())
    sn = load_spatialnet(patients, windows)

    # Same folds? compare test recordings fold by fold
    key = ["patient", "window_sec", "fold"]
    ref = df.set_index(key)["test_recordings"]
    other = sn.set_index(key)["test_recordings"]
    same = other.groupby(level=key).first().reindex(ref.index)
    mismatch = int((same.notna() & (same != ref)).sum())

    allm = pd.concat([df.assign(model="DCSENet"), sn])
    order = ["DCSENet", "grouped"] + sorted(
        (m for m in allm["model"].unique() if m.startswith("h=")), key=lambda m: int(m[2:]))
    parts = [
        "# DCSENet vs. SpatialNet on recording CV\n",
        f"Recording CV ({N_SPLITS} folds), patients {', '.join(patients)}. DCSENet: one spectrogram "
        "image per channel, window probability = mean over the 21 channels "
        "(see `experiments/experiment_b1_dcsenet.md`). SpatialNet rows come from "
        + ", ".join(f"`{f}`" for f, _, _ in SPATIALNET_RUNS) + ".\n",
        f"Folds with different test recordings than DCSENet: **{mismatch}** "
        "(0 = identical splits).\n",
    ]
    for metric in ("f1", "auroc", "sensitivity", "specificity"):
        for p in patients:
            g = allm[allm["patient"] == p].groupby(["model", "window_sec"])[metric]
            mean, std = g.mean().unstack(), g.std().unstack()
            rows = [m for m in order if m in mean.index]
            lines = [f"**{p}: {metric} (mean ± std over folds). Best per window in bold.**", "",
                     "| Model | " + " | ".join(f"{w} s" for w in windows) + " |",
                     "|---|" + "---|" * len(windows)]
            best = mean.loc[rows].max()
            for m in rows:
                cells = []
                for w in windows:
                    v = mean.loc[m].get(w, np.nan)
                    if pd.isna(v):
                        cells.append("—")
                        continue
                    s = f"**{v:.3f}**" if v == best[w] else f"{v:.3f}"
                    cells.append(f"{s} ± {std.loc[m].get(w, np.nan):.3f}")
                lines.append(f"| {m} | " + " | ".join(cells) + " |")
            parts.append("\n".join(lines) + "\n")

    img = df.groupby("window_sec")[["img_TP", "img_FP", "img_TN", "img_FN"]].sum()
    img.columns = ["TP", "FP", "TN", "FN"]
    img = add_metrics(img)
    parts += ["## DCSENet per image (paper style, pooled over folds)\n",
              "| Window | F1 | Sens. | Spec. | Acc. |", "|---|---|---|---|---|"]
    parts += [f"| {w} s | {r.f1:.3f} | {r.sensitivity:.3f} | {r.specificity:.3f} | {r.accuracy:.3f} |"
              for w, r in img.iterrows()]
    parts += ["", f"Training time, all folds: {df['train_sec'].sum() / 60:.1f} min. "
              f"Per-fold rows: `{final_csv.name}`."]
    report_md.write_text("\n".join(parts) + "\n")


def main(results_dir=RESULTS_DIR, patients=("chb01",), windows=WINDOWS):
    partial_csv, final_csv, report_md = result_paths(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(partial_csv).to_dict("records") if partial_csv.exists() else []
    done = {(r["patient"], r["window_sec"], r["fold"]) for r in rows}

    for window_sec in windows:
        for patient in patients:
            if patient in EXCLUDE.get(window_sec, []):
                continue
            X, y, rec = rcv.load_patient(patient, window_sec)
            folds = list(rcv.recording_folds(y, rec, N_SPLITS, seed=SEED))
            for fold, (tr, te) in enumerate(folds):
                if (patient, window_sec, fold) in done:
                    continue
                t0 = time.time()
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X[tr], y[tr], test_size=VAL_FRAC, stratify=y[tr], random_state=SEED + fold)
                model, spec, best_epoch, history = train_dcsenet(
                    X_tr, y_tr[:, None], X_val, y_val[:, None], SEED + fold)
                t_train = time.time() - t0
                p_img = image_proba(model, spec, torch.from_numpy(X[te]).to(DEVICE))[..., 0]  # (n, 21)
                prob = p_img.mean(axis=1)
                y_img = np.broadcast_to(y[te][:, None], p_img.shape)
                rows.append({
                    "patient": patient, "window_sec": window_sec, "fold": fold,
                    "n_folds": len(folds), "test_recordings": ";".join(sorted(set(rec[te]))),
                    "n_train": len(y_tr), "n_test": len(te), "best_epoch": best_epoch,
                    "train_loss": history[best_epoch][0], "val_loss": history[best_epoch][1],
                    **confusion_counts(y[te], (prob >= 0.5).astype(int)),
                    **{f"img_{k}": v for k, v in
                       confusion_counts(y_img.ravel(), (p_img.ravel() >= 0.5).astype(int)).items()},
                    "auroc": roc_auc_score(y[te], prob) if len(set(y[te])) == 2 else np.nan,
                    "train_sec": t_train,
                })
                pd.DataFrame(rows).to_csv(partial_csv, index=False)   # checkpoint
                del model
                torch.cuda.empty_cache()
                r = add_metrics(pd.DataFrame(rows[-1:])).iloc[0]
                print(f"{window_sec:>2}s {patient} fold {fold}: F1={r['f1']:.3f} "
                      f"auroc={r['auroc']:.3f} best_epoch={best_epoch} "
                      f"({len(y_tr)} train win, {t_train:.0f}s)", flush=True)

    df = add_metrics(pd.DataFrame(rows))
    df.to_csv(final_csv, index=False)
    write_report(df, results_dir)
    print("\n" + report_md.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", nargs="+", default=["chb01"])
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    parser.add_argument("--out", default=RESULTS_DIR.name,
                        help="results subfolder name inside experiments/results/")
    args = parser.parse_args()
    main(RESULTS_ROOT / args.out, args.patients, args.windows)
