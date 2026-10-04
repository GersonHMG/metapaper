"""
Experiment A4: grouped kernel at 1 s windows (fills Table 3.1, "Kernel por
grupo", 1 s). See experiment_a4_grouped_1s.md.

The model is the 0-layer model of b_tcn: AsymSETNetGroupedSegments with 1 s
segments, no TCN, Linear-ReLU-Linear head per segment, window probability = mean
of the segment probabilities. At 1 s the window is a single segment.
Same folds, train/val split, seeds and TrainConfig as Experiment A and b_tcn
(experiment_b.py). No shuffle test: one segment cannot be reordered.
Sanity check: --windows 3 --patients chb01 chb18 must match b_tcn 0 layers.
Every (patient, window, fold) row is checkpointed.

    cd /home/gmarihuan/metapaper && python -m experiments.experiment_a4_grouped_1s --patients all
    python -m experiments.experiment_a4_grouped_1s --windows 3 --patients chb01 chb18
"""

import argparse
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import recording_cv as rcv
from experiments.experiment_a1_kernel_height import (N_SPLITS, VAL_FRAC, SEED, CFG, EXCLUDE,
                                                     RESULTS_ROOT, result_paths)
from experiments.experiment_b import window_proba, scores, RESULTS_DIR as B_TCN_DIR, REFERENCE_DIR
from experiments.training import set_seed, train_model, add_metrics
from models.asymsetnet_grouped_segments import AsymSETNetGroupedSegments

SEGMENT_SEC = 1
RESULTS_DIR = RESULTS_ROOT / "a4_grouped_1s"
METRICS = ["accuracy", "sensitivity", "specificity", "f1", "auroc"]


def per_patient_summary(df, metric):
    """Mean of the per-patient fold means ± std across patients, per window."""
    pm = df.groupby(["window_sec", "patient"])[metric].mean()
    return pm.groupby("window_sec").mean(), pm.groupby("window_sec").std(), pm


def write_report(df, results_dir=RESULTS_DIR):
    _, final_csv, report_md = result_paths(results_dir)
    windows = sorted(df["window_sec"].unique())
    n_pat = df.groupby("window_sec")["patient"].nunique()

    main = ["| Window | Patients | " + " | ".join(m.capitalize() if m != "auroc" else "AUROC"
                                                for m in METRICS) + " |",
            "|---|---|" + "---|" * len(METRICS)]
    for w in windows:
        cells = []
        for m in METRICS:
            mean, std, _ = per_patient_summary(df, m)
            cells.append(f"{mean[w]:.3f} ± {std[w]:.3f}")
        main.append(f"| {w} s | {n_pat[w]} | " + " | ".join(cells) + " |")

    # References on the same patients: b_tcn 0 layers (1 s segments) and SpatialNet(grouped)
    ref = add_metrics(pd.read_csv(result_paths(B_TCN_DIR)[1]))
    ref = ref[(ref.tcn_layers == 0) & (ref.segment_sec == SEGMENT_SEC)]
    spn = pd.read_csv(result_paths(REFERENCE_DIR)[1])
    comp = ["| Window | Patients | A4 (this run) | `b_tcn` 0 layers | `SpatialNet(grouped)` (`a_grouped_all`) |",
            "|---|---|---|---|---|"]
    for w in windows:
        pats = df.loc[df.window_sec == w, "patient"].unique()
        _, _, pm = per_patient_summary(df[df.window_sec == w], "f1")
        cell = lambda d: (lambda s: "—" if s.empty else f"{s.mean():.3f} ± {s.std():.3f}")(
            d[(d.window_sec == w) & d.patient.isin(pats)].groupby("patient")["f1"].mean())
        comp.append(f"| {w} s | {len(pats)} | {pm.mean():.3f} ± {pm.std():.3f} | "
                    f"{cell(ref)} | {cell(spn)} |")

    # Sanity: per patient vs b_tcn wherever both exist (windows other than 1 s)
    sanity = []
    for w in windows:
        r = ref[ref.window_sec == w]
        if r.empty:
            continue
        a = df[df.window_sec == w].groupby("patient")["f1"].mean()
        b = r.groupby("patient")["f1"].mean().reindex(a.index)
        sanity += [f"**{w} s window: A4 vs. `b_tcn` 0 layers, F1 per patient (fold mean)**", "",
                   "| Patient | A4 | `b_tcn` | Difference |", "|---|---|---|---|"]
        sanity += [f"| {p} | {a[p]:.3f} | {b[p]:.3f} | {a[p] - b[p]:+.3f} |" for p in a.index]
        same = (df[df.window_sec == w].merge(r, on=["patient", "fold"], suffixes=("", "_b"))
                .eval("test_recordings == test_recordings_b").all())
        sanity += ["", f"Identical test recordings in every fold: **{same}**.", ""]

    per_patient = ["| Patient | " + " | ".join(f"{w} s" for w in windows) + " |",
                   "|---|" + "---|" * len(windows)]
    pm_all = df.groupby(["patient", "window_sec"])["f1"].mean().unstack()
    for p in pm_all.index:
        per_patient.append(f"| {p} | " + " | ".join(
            "—" if pd.isna(pm_all.loc[p, w]) else f"{pm_all.loc[p, w]:.3f}" for w in windows) + " |")

    parts = [
        "# Experiment A4: Grouped kernel at 1 s windows\n",
        "`AsymSETNetGroupedSegments`, 1 s segments, 0 TCN layers, Linear-ReLU-Linear head; window "
        "probability = mean of the segment probabilities (a single segment at 1 s). Same model, "
        f"folds, seeds and `TrainConfig` as `b_tcn` 0 layers. Recording CV ({N_SPLITS} folds, "
        "leave-one-recording-out when a patient has fewer recordings). "
        "See `experiments/experiment_a4_grouped_1s.md`.\n",
        "Cells are the mean of the per-patient fold means ± **standard deviation** across patients. "
        f"Per-fold rows: `{final_csv.name}`.\n",
        "## Table 3.1 / 3.2 values\n",
        "\n".join(main) + "\n",
        "## Comparison with the existing grouped cells (F1, same patients)\n",
        "\n".join(comp) + "\n",
    ]
    if sanity:
        parts += ["## Sanity check\n",
                  "GPU training is not bit-reproducible: expect about ±0.02 F1.\n",
                  "\n".join(sanity)]
    parts += ["## Per patient (F1, fold mean)\n", "\n".join(per_patient) + "\n"]
    report_md.write_text("\n".join(parts))


def main(results_dir=RESULTS_DIR, patients=None, windows=(1,)):
    partial_csv, final_csv, report_md = result_paths(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(partial_csv).to_dict("records") if partial_csv.exists() else []
    done = {(r["patient"], r["window_sec"], r["fold"]) for r in rows}

    for window_sec in windows:
        n_seg = window_sec // SEGMENT_SEC
        for patient in patients or rcv.available_patients(window_sec):
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
                    AsymSETNetGroupedSegments(n_segments=n_seg, tcn_channels=()),
                    X_tr, y_tr, X_val, y_val, CFG)
                prob = window_proba(model, X[te])
                rows.append({
                    "patient": patient, "window_sec": window_sec, "segment_sec": SEGMENT_SEC,
                    "n_segments": n_seg, "tcn_layers": 0, "fold": fold, "n_folds": len(folds),
                    "test_recordings": ";".join(sorted(set(rec[te]))),
                    "n_train": len(y_tr), "n_test": len(te), "best_epoch": best_epoch,
                    **scores(y[te], prob), "train_sec": time.time() - t0,
                })
                pd.DataFrame(rows).to_csv(partial_csv, index=False)   # checkpoint
                last = add_metrics(pd.DataFrame(rows[-1:])).iloc[0]
                print(f"{window_sec:>2}s {patient} fold {fold}: F1={last['f1']:.3f} "
                      f"auroc={last['auroc']:.3f} best_epoch={best_epoch} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    df = add_metrics(pd.DataFrame(rows))
    df.to_csv(final_csv, index=False)
    write_report(df, results_dir)
    print("\n" + report_md.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", nargs="+", default=["all"],
                        help="patient names, or 'all' for every patient with data")
    parser.add_argument("--windows", type=int, nargs="+", default=[1])
    parser.add_argument("--out", default=RESULTS_DIR.name,
                        help="results subfolder name inside experiments/results/")
    args = parser.parse_args()
    patients = None if args.patients == ["all"] else args.patients
    main(RESULTS_ROOT / args.out, patients, args.windows)
