"""
Experiment A2: factorized spatial module (temporal h=1 blocks, then learned
pooling over electrodes) against the h=1 and h=21 baselines.

Same patients, windows, folds, train/val split and TrainConfig as Experiment A1
(imported from experiment_a1_kernel_height), so the results are comparable.

Besides F1 on the test fold, every model is also scored on the test fold with
the electrode order shuffled (one fixed permutation per fold): `f1_shuffled`.
The drop f1 - f1_shuffled measures how much the model relies on spatial
layout. h=1 is permutation invariant, so its drop is 0 up to GPU noise.
Every (patient, window, model, fold) row is checkpointed, so a rerun resumes.

    cd /home/gmarihuan/metapaper && python -m experiments.experiment_a2_factorized \
        --patients chb01 chb02 --windows 1 3 5 8 10

Grouped vs kernel height as the spatial step after the temporal blocks:

    python -m experiments.experiment_a2_factorized --patients chb01 chb02 \
        --models h1 grouped fact_grouped fact_h4 fact_h8 fact_h16 fact_hcmax fact_mix \
        --out a3_fact_grouped_vs_height_chb01_chb02
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from datasets import recording_cv as rcv
from experiments.experiment_a1_kernel_height import (PATIENTS, WINDOWS, N_SPLITS, VAL_FRAC,
                                                     SEED, CFG, EXCLUDE, RESULTS_ROOT,
                                                     result_paths, f1_table)
from experiments.training import (set_seed, train_model, predict_proba,
                                  confusion_counts, add_metrics)
from models.spatial_net import SpatialNet

RESULTS_DIR = RESULTS_ROOT / "a2_factorized"

MODELS = {
    "h1":        dict(spatial="height", kernel_height=1),
    "h21":       dict(spatial="height", kernel_height=21),
    "fact_mix":  dict(spatial="factorized", pool="mix"),
    "fact_attn": dict(spatial="factorized", pool="attn"),
    "fact_attn_cdrop": dict(spatial="factorized", pool="attn", channel_dropout=0.2),
    # grouped vs kernel height as the spatial step after the h=1 temporal blocks
    "grouped":      dict(spatial="grouped"),          # raw-signal grouped (Experiment A)
    "fact_grouped": dict(spatial="factorized", pool="grouped"),
    "fact_h4":      dict(spatial="factorized", pool="height", kernel_height=4),
    "fact_h8":      dict(spatial="factorized", pool="height", kernel_height=8),
    "fact_h16":     dict(spatial="factorized", pool="height", kernel_height=16),
    "fact_hcmax":   dict(spatial="factorized", pool="height"),   # kernel height Cmax = 21
}


def f1_of(y_true, prob):
    c = confusion_counts(y_true, (prob >= 0.5).astype(int))
    return 2 * c["TP"] / max(2 * c["TP"] + c["FP"] + c["FN"], 1)


def write_report(df, results_dir=RESULTS_DIR):
    _, final_csv, report_md = result_paths(results_dir)
    models = [m for m in MODELS if m in set(df["model"])]
    patients = list(dict.fromkeys(df["patient"]))
    df = df.assign(shuffle_drop=df["f1"] - df["f1_shuffled"])
    parts = [
        "# Experiment A2: Factorized spatial module results\n",
        f"`SpatialNet`, recording CV ({N_SPLITS} folds), patients {', '.join(patients)}. "
        "Models: " + ", ".join(f"`{m}` = `{MODELS[m]}`" for m in models) + ".\n",
        "Cells are mean ± variance. Best mean per window in bold. "
        f"Per-fold rows with all metrics: `{final_csv.name}`.\n",
    ]
    for metric, title in (("f1", "F1"),
                          ("shuffle_drop", "F1 drop with electrodes shuffled (spatial reliance)")):
        by_patient = df.groupby(["window_sec", "model", "patient"])[metric]
        fold_mean, fold_var = by_patient.mean(), by_patient.var()
        by_window = fold_mean.groupby(["window_sec", "model"])
        bold = metric == "f1"
        parts += [f"## {title}\n",
                  f1_table(by_window.mean().unstack()[models], by_window.var().unstack()[models],
                           "Mean over patients", cols=models, label="{}", bold_best=bold)]
        for p in patients:
            parts.append(f1_table(fold_mean.xs(p, level="patient").unstack()[models],
                                  fold_var.xs(p, level="patient").unstack()[models],
                                  f"{p} (mean ± variance over folds)", cols=models,
                                  label="{}", bold_best=bold))
    report_md.write_text("\n".join(parts))


def main(models=tuple(MODELS), windows=WINDOWS, results_dir=RESULTS_DIR, patients=PATIENTS):
    partial_csv, final_csv, _ = result_paths(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(partial_csv).to_dict("records") if partial_csv.exists() else []
    done = {(r["patient"], r["window_sec"], r["model"], r["fold"]) for r in rows}

    for window_sec in windows:
        for patient in patients:
            if patient in EXCLUDE.get(window_sec, []):
                continue
            X, y, rec = rcv.load_patient(patient, window_sec)
            folds = list(rcv.recording_folds(y, rec, N_SPLITS, seed=SEED))
            for fold, (tr, te) in enumerate(folds):
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X[tr], y[tr], test_size=VAL_FRAC, stratify=y[tr], random_state=SEED + fold)
                perm = np.random.default_rng(SEED + fold).permutation(X.shape[1])
                X_te_shuffled = np.ascontiguousarray(X[te][:, perm])
                for name in models:
                    if (patient, window_sec, name, fold) in done:
                        continue
                    t0 = time.time()
                    set_seed(SEED + fold)
                    model, best_epoch, _ = train_model(
                        SpatialNet(**MODELS[name]), X_tr, y_tr, X_val, y_val, CFG)
                    prob = predict_proba(model, X[te])
                    rows.append({
                        "patient": patient, "window_sec": window_sec, "model": name,
                        "fold": fold, "n_folds": len(folds),
                        "test_recordings": ";".join(sorted(set(rec[te]))),
                        "n_train": len(y_tr), "n_test": len(te), "best_epoch": best_epoch,
                        **confusion_counts(y[te], (prob >= 0.5).astype(int)),
                        "auroc": roc_auc_score(y[te], prob) if len(set(y[te])) == 2 else np.nan,
                        "f1_shuffled": f1_of(y[te], predict_proba(model, X_te_shuffled)),
                    })
                    pd.DataFrame(rows).to_csv(partial_csv, index=False)   # checkpoint
                    r = add_metrics(pd.DataFrame(rows[-1:])).iloc[0]
                    print(f"{window_sec:>2}s {patient} {name:<16} fold {fold}: "
                          f"F1={r['f1']:.3f} shuffled={r['f1_shuffled']:.3f} "
                          f"({time.time() - t0:.0f}s)", flush=True)

    df = add_metrics(pd.DataFrame(rows))
    df.to_csv(final_csv, index=False)
    write_report(df, results_dir)

    per_patient = df.groupby(["window_sec", "model", "patient"])[["f1", "f1_shuffled"]].mean()
    summary = per_patient.groupby(["window_sec", "model"]).mean()
    print("\nMean F1 (folds averaged per patient, then across patients):")
    print(summary["f1"].unstack("model")[list(models)].round(3))
    print("\nMean F1 with electrodes shuffled:")
    print(summary["f1_shuffled"].unstack("model")[list(models)].round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    parser.add_argument("--patients", nargs="+", default=PATIENTS,
                        help="patient ids, or 'all' for chb01-chb24")
    parser.add_argument("--out", default=RESULTS_DIR.name,
                        help="results subfolder of experiments/results/")
    args = parser.parse_args()
    patients = ([f"chb{i:02d}" for i in range(1, 25)] if args.patients == ["all"]
                else args.patients)
    main(args.models, args.windows, RESULTS_ROOT / args.out, patients)
