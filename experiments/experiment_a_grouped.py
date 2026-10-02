"""
Experiment A: grouped kernel model, SpatialNet(spatial="grouped").
See experiment_a.md.

Same patients, windows, folds, train/val split and TrainConfig as Experiment A1
(imported from experiment_a1_kernel_height), so the results are comparable.
Every (patient, window, fold) row is checkpointed, so a rerun resumes.

    cd /home/gmarihuan/metapaper && python -m experiments.experiment_a_grouped
    python -m experiments.experiment_a_grouped --windows 1 3 5 8 10 --patients all \
        --out a_grouped_all     # subfolder of experiments/results/
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

RESULTS_DIR = RESULTS_ROOT / "a_grouped"


def write_report(df, results_dir=RESULTS_DIR):
    _, final_csv, report_md = result_paths(results_dir)
    patients = list(dict.fromkeys(df["patient"]))         # order of first appearance
    by_patient = df.groupby(["window_sec", "patient"])["f1"]
    fold_mean, fold_var = by_patient.mean(), by_patient.var()   # var over folds (ddof=1)
    by_window = fold_mean.groupby("window_sec")
    overall = pd.DataFrame({"grouped": by_window.mean()})
    overall_var = pd.DataFrame({"grouped": by_window.var()})
    parts = [
        "# Experiment A: Grouped kernel results\n",
        f"`SpatialNet(spatial=\"grouped\")`, recording CV ({N_SPLITS} folds), "
        f"patients {', '.join(patients)}. See `experiments/experiment_a.md`.\n",
        "Cells are F1 mean ± variance. "
        f"Per-fold rows with all metrics: `{final_csv.name}`.\n",
        "## Mean over patients\n",
        f1_table(overall, overall_var,
                 "All patients (mean of the per-patient fold means ± variance across patients)",
                 cols=["grouped"], label="{}", bold_best=False),
        "## Per patient\n",
        f1_table(fold_mean.unstack("window_sec").loc[patients],
                 fold_var.unstack("window_sec").loc[patients], "Mean ± variance over folds",
                 cols=sorted(df["window_sec"].unique()), label="{} s", bold_best=False,
                 row_header="Patient", row_label="{}"),
    ]
    report_md.write_text("\n".join(parts))


def main(windows=WINDOWS, results_dir=RESULTS_DIR, patients=PATIENTS):
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
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X[tr], y[tr], test_size=VAL_FRAC, stratify=y[tr], random_state=SEED + fold)
                t0 = time.time()
                set_seed(SEED + fold)
                model, best_epoch, _ = train_model(
                    SpatialNet(spatial="grouped"), X_tr, y_tr, X_val, y_val, CFG)
                prob = predict_proba(model, X[te])
                rows.append({
                    "patient": patient, "window_sec": window_sec,
                    "fold": fold, "n_folds": len(folds),
                    "test_recordings": ";".join(sorted(set(rec[te]))),
                    "n_train": len(y_tr), "n_test": len(te), "best_epoch": best_epoch,
                    **confusion_counts(y[te], (prob >= 0.5).astype(int)),
                    "auroc": roc_auc_score(y[te], prob) if len(set(y[te])) == 2 else np.nan,
                })
                pd.DataFrame(rows).to_csv(partial_csv, index=False)   # checkpoint
                print(f"{window_sec:>2}s {patient} fold {fold}: "
                      f"F1={add_metrics(pd.DataFrame(rows[-1:]))['f1'].iloc[0]:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    df = add_metrics(pd.DataFrame(rows))
    df.to_csv(final_csv, index=False)
    write_report(df, results_dir)
    print("\n" + report_md.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    parser.add_argument("--patients", nargs="+", default=PATIENTS,
                        help="patient names, or 'all' for every patient with data")
    parser.add_argument("--out", default=RESULTS_DIR.name,
                        help="results subfolder name inside experiments/results/")
    args = parser.parse_args()
    patients = (rcv.available_patients(min(args.windows)) if args.patients == ["all"]
                else args.patients)
    main(args.windows, RESULTS_ROOT / args.out, patients)
