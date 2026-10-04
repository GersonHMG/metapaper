"""
Experiment A, filtered: does band-pass filtering change grouped vs. h=1?

Siena is unfiltered (strong < 0.5 Hz drift, DC offsets, > 45 Hz noise) while
CHB-MIT looks band-limited by its hardware, and grouped collapses on Siena. Here
every window is filtered at load time (datasets.preprocessing.filter_windows:
0.5-45 Hz band-pass + mains notch, 60 Hz CHB-MIT / 50 Hz Siena) and grouped and
h=1 are retrained on a few patients. Folds, splits, seeds and TrainConfig are the
same as the unfiltered runs, which serve as the baseline:
a_grouped_all + a1_kernel_height_h1_all (CHB-MIT) and a_siena_5p (Siena).

    cd /home/gmarihuan/metapaper && python -m experiments.experiment_a_filtered
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from datasets import recording_cv as chb_rcv
from datasets.preprocessing import filter_windows
from experiments.experiment_a1_kernel_height import (N_SPLITS, VAL_FRAC, SEED, CFG,
                                                     RESULTS_ROOT, result_paths)
from experiments.experiment_a_siena import model_specs, rcv as siena_rcv
from experiments.training import (set_seed, train_model, predict_proba,
                                  confusion_counts, add_metrics)
from models.spatial_net import SpatialNet

# (dataset, montage) -> patients
RUNS = {("chbmit", "bipolar"): ["chb01", "chb18"],
        ("siena", "bipolar"): ["PN00", "PN14"],
        ("siena", "referential"): ["PN00", "PN14"]}
NOTCH = {"chbmit": 60, "siena": 50}
WINDOWS = [3, 10]
BAND = (0.5, 45.0)
RESULTS_DIR = RESULTS_ROOT / "a_filtered"


def load(dataset, montage, patient, window_sec):
    if dataset == "chbmit":
        return chb_rcv.load_patient(patient, window_sec), chb_rcv.recording_folds
    return siena_rcv.load_patient(patient, window_sec, montage), siena_rcv.recording_folds


def specs(dataset, montage):
    if dataset == "siena":
        return model_specs(montage, ["1"])
    return [("grouped", dict(spatial="grouped")),
            ("h=1", dict(spatial="height", kernel_height=1))]


def unfiltered_baseline():
    """Fold rows of the unfiltered runs, in the same schema (dataset, montage, model)."""
    g = pd.read_csv(result_paths(RESULTS_ROOT / "a_grouped_all")[1]).assign(model="grouped")
    h = pd.read_csv(result_paths(RESULTS_ROOT / "a1_kernel_height_h1_all")[1])
    h = h[h["kernel_height"] == 1].assign(model="h=1").drop(columns="kernel_height")
    chb = pd.concat([g, h]).assign(dataset="chbmit", montage="bipolar")
    sie = pd.read_csv(result_paths(RESULTS_ROOT / "a_siena_5p")[1]).assign(dataset="siena")
    return pd.concat([chb, sie[sie["model"].isin(["grouped", "h=1"])]], ignore_index=True)


def write_report(df, results_dir):
    _, final_csv, report_md = result_paths(results_dir)
    key = ["dataset", "montage", "window_sec", "patient", "model"]
    filt = df.groupby(key)["f1"].mean().rename("filtered")
    base = unfiltered_baseline()
    base = base.merge(df[key[:-1] + ["fold"]].drop_duplicates(), on=key[:-1] + ["fold"])
    raw = base.groupby(key)["f1"].mean().rename("unfiltered")
    t = pd.concat([raw, filt], axis=1).unstack("model")
    lines = ["| Dataset | Montage | Window | Patient | grouped raw | grouped filt | h=1 raw | h=1 filt "
             "| grouped − h=1 raw | grouped − h=1 filt |", "|" + "---|" * 10]
    for idx, r in t.iterrows():
        d, m, w, p = idx
        g0, g1 = r[("unfiltered", "grouped")], r[("filtered", "grouped")]
        h0, h1 = r[("unfiltered", "h=1")], r[("filtered", "h=1")]
        lines.append(f"| {d} | {m} | {w} s | {p} | {g0:.3f} | {g1:.3f} | {h0:.3f} | {h1:.3f} "
                     f"| {g0 - h0:+.3f} | {g1 - h1:+.3f} |")
    mean = t.groupby(level=["dataset", "montage", "window_sec"]).mean()
    mlines = ["| Dataset | Montage | Window | grouped raw | grouped filt | h=1 raw | h=1 filt "
              "| grouped − h=1 raw | grouped − h=1 filt |", "|" + "---|" * 9]
    for (d, m, w), r in mean.iterrows():
        g0, g1 = r[("unfiltered", "grouped")], r[("filtered", "grouped")]
        h0, h1 = r[("unfiltered", "h=1")], r[("filtered", "h=1")]
        mlines.append(f"| {d} | {m} | {w} s | {g0:.3f} | {g1:.3f} | {h0:.3f} | {h1:.3f} "
                      f"| {g0 - h0:+.3f} | {g1 - h1:+.3f} |")
    parts = [
        "# Experiment A, filtered: grouped vs. h=1 with band-pass filtering\n",
        f"Windows filtered at load time: {BAND[0]}–{BAND[1]} Hz zero-phase Butterworth band-pass "
        "(order 4) after removing the window mean, plus a notch at the mains frequency "
        "(60 Hz CHB-MIT, 50 Hz Siena), `datasets.preprocessing.filter_windows`. "
        "Same folds, splits, seeds and TrainConfig as the unfiltered runs "
        "(`a_grouped_all`, `a1_kernel_height_h1_all`, `a_siena_5p`), which give the *raw* columns. "
        "GPU training varies by about ±0.02 F1 between identical runs.\n",
        f"Per-fold rows: `{final_csv.name}`. Generated by `experiments/experiment_a_filtered.py`.\n",
        "## Mean over the patients (F1, fold means averaged)\n", "\n".join(mlines), "",
        "## Per patient (F1, mean over folds)\n", "\n".join(lines), "",
    ]
    report_md.write_text("\n".join(parts))


def main(windows=WINDOWS, results_dir=RESULTS_DIR):
    partial_csv, final_csv, report_md = result_paths(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(partial_csv).to_dict("records") if partial_csv.exists() else []
    done = {(r["dataset"], r["montage"], r["patient"], r["window_sec"], r["model"], r["fold"])
            for r in rows}

    for (dataset, montage), patients in RUNS.items():
        for window_sec in windows:
            for patient in patients:
                (X, y, rec), recording_folds = load(dataset, montage, patient, window_sec)
                X = filter_windows(X, band=BAND, notch=NOTCH[dataset])
                folds = list(recording_folds(y, rec, N_SPLITS, seed=SEED))
                for fold, (tr, te) in enumerate(folds):
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X[tr], y[tr], test_size=VAL_FRAC, stratify=y[tr],
                        random_state=SEED + fold)
                    for name, kwargs in specs(dataset, montage):
                        if (dataset, montage, patient, window_sec, name, fold) in done:
                            continue
                        t0 = time.time()
                        set_seed(SEED + fold)
                        model, best_epoch, _ = train_model(
                            SpatialNet(**kwargs), X_tr, y_tr, X_val, y_val, CFG)
                        prob = predict_proba(model, X[te])
                        rows.append({
                            "dataset": dataset, "montage": montage, "patient": patient,
                            "window_sec": window_sec, "model": name,
                            "fold": fold, "n_folds": len(folds),
                            "test_recordings": ";".join(sorted(set(rec[te]))),
                            "n_train": len(y_tr), "n_test": len(te), "best_epoch": best_epoch,
                            **confusion_counts(y[te], (prob >= 0.5).astype(int)),
                            "auroc": (roc_auc_score(y[te], prob) if len(set(y[te])) == 2
                                      else np.nan),
                        })
                        pd.DataFrame(rows).to_csv(partial_csv, index=False)   # checkpoint
                        print(f"{dataset} {montage[:3]} {window_sec:>2}s {patient} {name:<7} "
                              f"fold {fold}: "
                              f"F1={add_metrics(pd.DataFrame(rows[-1:]))['f1'].iloc[0]:.3f} "
                              f"({time.time() - t0:.0f}s)", flush=True)

    df = add_metrics(pd.DataFrame(rows))
    df.to_csv(final_csv, index=False)
    write_report(df, results_dir)
    print("\n" + report_md.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    parser.add_argument("--out", default=RESULTS_DIR.name,
                        help="results subfolder name inside experiments/results/")
    args = parser.parse_args()
    main(args.windows, RESULTS_ROOT / args.out)
