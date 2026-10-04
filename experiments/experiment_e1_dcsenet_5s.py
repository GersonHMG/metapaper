"""
Experiment E1: DCSENet baseline at 5 s windows vs. the grouped spatial block
(Table 3.4). See experiment_e1_dcsenet_5s.md.

DCSENet as in a_dcsenet (one spectrogram image per channel, window probability
= mean of the 21 channel probabilities), with the paper's settings for 5 s:
FIR band-pass 0.5-30 Hz (Hamming), Gaussian taper sigma = 0.1, STFT window
WL = 5 s (one frame per 5 s window). Training: TrainConfig defaults (focal loss,
Adam 1e-3, weight decay 1e-4, batch 64 images, <= 100 epochs, early stopping
patience 15 on validation loss).
Protocol identical to Experiment A / b_tcn at 5 s: recording_folds (5 folds, or
leave-one-recording-out), 80/20 stratified train/val split, test keeps <= 1:5.
Every (patient, fold) row is checkpointed.

    cd /home/gmarihuan/metapaper && python -m experiments.experiment_e1_dcsenet_5s --patients chb01
    python -m experiments.experiment_e1_dcsenet_5s --patients all
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from datasets import recording_cv as rcv
from experiments.experiment_a1_kernel_height import (N_SPLITS, VAL_FRAC, SEED, CFG, EXCLUDE,
                                                     RESULTS_ROOT, result_paths)
from experiments.experiment_b import RESULTS_DIR as B_TCN_DIR
from experiments.experiment_b1_dcsenet import train_dcsenet, image_proba
from experiments.training import DEVICE, confusion_counts, add_metrics

WINDOW_SEC = 5
SPEC = dict(window_length_s=5.0, step_s=5.0, taper="gaussian", sigma=0.1, bandpass=(0.5, 30.0))
TRAIN = dict(epochs=CFG.epochs, batch_size=CFG.batch_size, lr=CFG.lr,
             weight_decay=CFG.weight_decay, patience=CFG.patience,
             focal_alpha=CFG.focal_alpha, focal_gamma=CFG.focal_gamma, cache=True)
RESULTS_DIR = RESULTS_ROOT / "e1_dcsenet_5s"
METRICS = ["accuracy", "sensitivity", "specificity", "f1", "auroc"]
NAMES = {"accuracy": "Acc", "sensitivity": "Sen", "specificity": "Spec", "f1": "F1", "auroc": "AUROC"}


def spatial_block(patients):
    """b_tcn, 0 TCN layers, 1 s segments, 5 s windows: per-fold rows."""
    b = add_metrics(pd.read_csv(result_paths(B_TCN_DIR)[1]))
    return b[(b.tcn_layers == 0) & (b.segment_sec == 1) & (b.window_sec == WINDOW_SEC)
             & b.patient.isin(patients)]


def patient_means(df, metric):
    return df.groupby("patient")[metric].mean()


def write_report(df, results_dir=RESULTS_DIR):
    _, final_csv, report_md = result_paths(results_dir)
    patients = list(dict.fromkeys(df["patient"]))
    sp = spatial_block(patients)
    same = (df.merge(sp, on=["patient", "fold"], suffixes=("", "_sp"))
            .eval("test_recordings == test_recordings_sp"))

    fmt = lambda s: f"{100 * s.mean():.1f} ± {100 * s.std():.1f}"
    summary = ["| Model | " + " | ".join(NAMES[m] for m in METRICS) + " |",
               "|---|" + "---|" * len(METRICS)]
    for name, d in (("DCSENet", df), ("Bloque espacial (`b_tcn`, 0 layers)", sp)):
        summary.append(f"| {name} | " + " | ".join(fmt(patient_means(d, m)) for m in METRICS) + " |")

    paired = ["| Metric | Spatial − DCSENet | Spatial wins | Ties | p (Wilcoxon) |",
              "|---|---|---|---|---|"]
    for m in METRICS:
        a, b = patient_means(sp, m), patient_means(df, m)
        pair = pd.concat([a, b], axis=1, keys=["sp", "dc"]).dropna()
        d = pair["sp"] - pair["dc"]
        p = wilcoxon(pair["sp"], pair["dc"]).pvalue if (d != 0).any() else np.nan
        paired.append(f"| {NAMES[m]} | {100 * d.mean():+.1f} | {(d > 0).sum()} / {len(d)} | "
                      f"{(d == 0).sum()} | " + ("<0.001" if p < 0.001 else f"{p:.3f}") + " |")

    img = df[["img_TP", "img_FP", "img_TN", "img_FN"]].copy()
    img.columns = ["TP", "FP", "TN", "FN"]
    img = add_metrics(img.assign(patient=df["patient"].values))
    img_row = " | ".join(fmt(patient_means(img, m)) for m in METRICS[:4])

    per_patient = ["| Patient | Folds | DCSENet F1 | Spatial F1 | DCSENet AUROC | Spatial AUROC |",
                   "|---|---|---|---|---|---|"]
    for p in patients:
        per_patient.append(
            f"| {p} | {(df.patient == p).sum()} | {patient_means(df, 'f1')[p]:.3f} | "
            f"{patient_means(sp, 'f1').get(p, np.nan):.3f} | {patient_means(df, 'auroc')[p]:.3f} | "
            f"{patient_means(sp, 'auroc').get(p, np.nan):.3f} |")

    collapsed = df[(df.TP + df.FP) == 0]
    parts = [
        "# Experiment E1: DCSENet baseline at 5 s windows\n",
        f"Recording CV at {WINDOW_SEC} s ({N_SPLITS} folds, leave-one-recording-out when a patient "
        f"has fewer seizure recordings), {len(patients)} patients, same folds and test sets as "
        "`a_grouped_all` and `b_tcn`. See `experiments/experiment_e1_dcsenet_5s.md`.\n",
        "## DCSENet settings\n",
        "- **Preprocessing:** FIR band-pass 0.5–30 Hz, Hamming window (1,691 taps, zero phase), "
        "applied to each 5 s channel signal with reflect padding.",
        "- **Spectrogram (Algorithm 1):** **Gaussian taper, σ = 0.1, STFT window WL = 5 s**, the "
        "paper's best setting for WL = 5 s (Table 3, Fig. 6). On a 5 s window this is a single "
        "STFT frame: the image is one log-frequency spectrum (224 bins, 0.5–40 Hz, dB relative to "
        "its mean) stretched to 224 × 224 × 3.",
        "- **Network:** DCSENet as in the paper (5,763,361 parameters), one output.",
        "- **Channels:** one image per channel (21 channels in our montage), window probability = "
        "mean of the 21 image probabilities.",
        f"- **Training:** `TrainConfig` defaults: focal loss (α {CFG.focal_alpha}, γ "
        f"{CFG.focal_gamma}), Adam lr {CFG.lr}, weight decay {CFG.weight_decay}, batch "
        f"{CFG.batch_size} images, ≤ {CFG.epochs} epochs, early stopping (patience {CFG.patience}) "
        "on validation loss.\n",
        f"Identical test recordings to `b_tcn` in every fold: **{bool(same.all())}** "
        f"({len(same)} folds compared).\n",
        "## Table 3.4, 5 s (window level)\n",
        "Mean of the per-patient fold means ± standard deviation across patients, in %.\n",
        "\n".join(summary) + "\n",
        "## Paired test per patient\n",
        "Per-patient fold means; difference in percentage points (positive = spatial block "
        "better); two-sided Wilcoxon signed-rank test over patients.\n",
        "\n".join(paired) + "\n",
        "## DCSENet per image (paper style)\n",
        "Every channel image scored on its own; mean of per-patient fold means ± std, in %.\n",
        "| Acc | Sen | Spec | F1 |", "|---|---|---|---|", f"| {img_row} |\n",
        f"Folds where DCSENet predicts no seizure at all: **{len(collapsed)}** of {len(df)}"
        + (" (" + ", ".join(f"{r.patient} f{r.fold}" for r in collapsed.itertuples()) + ")"
           if len(collapsed) else "") + ".\n",
        "## Per patient\n",
        "\n".join(per_patient) + "\n",
        f"Training time, all folds: {df['train_sec'].sum() / 60:.1f} min. "
        f"Per-fold rows: `{final_csv.name}`.",
    ]
    report_md.write_text("\n".join(parts) + "\n")


def main(results_dir=RESULTS_DIR, patients=None):
    partial_csv, final_csv, report_md = result_paths(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(partial_csv).to_dict("records") if partial_csv.exists() else []
    done = {(r["patient"], r["fold"]) for r in rows}

    for patient in patients or rcv.available_patients(WINDOW_SEC):
        if patient in EXCLUDE.get(WINDOW_SEC, []):
            continue
        X, y, rec = rcv.load_patient(patient, WINDOW_SEC)
        folds = list(rcv.recording_folds(y, rec, N_SPLITS, seed=SEED))
        for fold, (tr, te) in enumerate(folds):
            if (patient, fold) in done:
                continue
            t0 = time.time()
            X_tr, X_val, y_tr, y_val = train_test_split(
                X[tr], y[tr], test_size=VAL_FRAC, stratify=y[tr], random_state=SEED + fold)
            model, spec, best_epoch, history = train_dcsenet(
                X_tr, y_tr[:, None], X_val, y_val[:, None], SEED + fold, spec_kwargs=SPEC, **TRAIN)
            t_train = time.time() - t0
            p_img = image_proba(model, spec, torch.from_numpy(X[te]).to(DEVICE))[..., 0]  # (n, 21)
            prob = p_img.mean(axis=1)
            y_img = np.broadcast_to(y[te][:, None], p_img.shape)
            rows.append({
                "patient": patient, "window_sec": WINDOW_SEC, "fold": fold, "n_folds": len(folds),
                "test_recordings": ";".join(sorted(set(rec[te]))),
                "n_train": len(y_tr), "n_test": len(te), "best_epoch": best_epoch,
                "n_epochs": len(history),
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
            print(f"{patient} fold {fold}: F1={r['f1']:.3f} auroc={r['auroc']:.3f} "
                  f"best_epoch={best_epoch}/{len(history)} ({len(y_tr)} train win, {t_train:.0f}s)",
                  flush=True)

    df = add_metrics(pd.DataFrame(rows))
    df.to_csv(final_csv, index=False)
    write_report(df, results_dir)
    print("\n" + report_md.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", nargs="+", default=["all"],
                        help="patient names, or 'all' for every patient with data")
    parser.add_argument("--out", default=RESULTS_DIR.name,
                        help="results subfolder name inside experiments/results/")
    args = parser.parse_args()
    main(RESULTS_ROOT / args.out, None if args.patients == ["all"] else args.patients)
