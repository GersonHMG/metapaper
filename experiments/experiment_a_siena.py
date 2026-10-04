"""
Experiment A on Siena Scalp: grouped kernel vs. kernel height, bipolar and referential.
See experiment_a_siena.md.

Same protocol as Experiment A / A1 on CHB-MIT (folds, train/val split, seeds and
TrainConfig imported from experiment_a1_kernel_height), on the Siena recording-CV
data in /home/gmarihuan/SIENNA_SCALP_PROCESSED. Every model of a fold shares the
same train/val split. Every (montage, patient, window, model, fold) row is
checkpointed, so a rerun resumes.

    cd /home/gmarihuan/metapaper && python -m experiments.experiment_a_siena
    python -m experiments.experiment_a_siena --montages referential --windows 3 5 \
        --patients all --out a_siena_all     # subfolder of experiments/results/
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SIENA_DIR = "/home/gmarihuan/SIENNA_SCALP_PROCESSED"
if SIENA_DIR not in sys.path:
    sys.path.append(SIENA_DIR)
from siena import recording_cv as rcv                                    # noqa: E402

from experiments.experiment_a1_kernel_height import (N_SPLITS, VAL_FRAC, SEED, CFG,   # noqa: E402
                                                     RESULTS_ROOT, result_paths, f1_table)
from experiments.training import (set_seed, train_model, predict_proba,  # noqa: E402
                                  confusion_counts, add_metrics)
from models.asymsetnet_grouped import (ELECTRODE_GROUPS, FLIP_CHANNELS,   # noqa: E402
                                       REFERENTIAL_GROUPS)
from models.spatial_net import SpatialNet                                # noqa: E402

MONTAGES = ["bipolar", "referential"]
N_CHANNELS = {"bipolar": 21, "referential": 19}
GROUPS = {"bipolar": (ELECTRODE_GROUPS, FLIP_CHANNELS),
          "referential": (REFERENTIAL_GROUPS, [])}
PATIENTS = ["PN00", "PN06", "PN10", "PN12", "PN14"]   # the most recordings
WINDOWS = [1, 3, 5, 8, 10]
KERNEL_HEIGHTS = ["1", "4", "8", "16", "full"]         # "full" = all channels
RESULTS_DIR = RESULTS_ROOT / "a_siena_5p"


def model_specs(montage, heights):
    """[(model name, SpatialNet kwargs)]: grouped first, then one per kernel height."""
    n = N_CHANNELS[montage]
    groups, flip = GROUPS[montage]
    specs = [("grouped", dict(spatial="grouped", groups=groups, flip_channels=flip))]
    for h in heights:
        h = n if h == "full" else int(h)
        specs.append((f"h={h}", dict(spatial="height", kernel_height=h)))
    return [(name, dict(n_channels=n, **kw)) for name, kw in specs]


def model_order(models):
    """grouped first, then kernel heights in increasing order."""
    return sorted(models, key=lambda m: -1 if m == "grouped" else int(m[2:]))


def paired_table(pm, windows, best_h):
    """Grouped vs. the best kernel height of each window: mean F1 difference,
    wins and Wilcoxon p over patients."""
    lines = ["| Window | Best h | grouped − best h | grouped wins | p |",
             "|---|---|---|---|---|"]
    for w in windows:
        pair = pm.loc[w, ["grouped", best_h[w]]].dropna()
        d = pair["grouped"] - pair[best_h[w]]
        try:
            p = wilcoxon(pair["grouped"], pair[best_h[w]]).pvalue
            p = "<0.001" if p < 0.001 else f"{p:.3f}"
        except ValueError:                       # e.g. all differences zero
            p = "—"
        lines.append(f"| {w} s | {best_h[w]} | {d.mean():+.3f} | {(d > 0).sum()} / {len(d)} | {p} |")
    return "\n".join(lines) + "\n"


def per_patient_table(pm, windows, models):
    """Rows patient × model, best model per patient and window in bold."""
    lines = ["| Patient | Model | " + " | ".join(f"{w} s" for w in windows) + " |",
             "|---|---|" + "---|" * len(windows)]
    for p in pm.index.get_level_values("patient").unique():
        for m in models:
            cells = []
            for w in windows:
                if (w, p) not in pm.index:
                    cells.append("—")
                    continue
                v = pm.loc[(w, p), m]
                cells.append(f"**{v:.3f}**" if v == pm.loc[(w, p)].max() else f"{v:.3f}")
            lines.append(f"| {p} | {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def montage_section(df, montage):
    models = model_order(df["model"].unique())
    windows = sorted(df["window_sec"].unique())
    fold_mean = df.groupby(["window_sec", "patient", "model"])["f1"].mean()
    pm = fold_mean.unstack("model")[models]                 # (window, patient) x model
    by_window = pm.groupby("window_sec")
    mean = by_window.mean()
    heights = [m for m in models if m != "grouped"]
    best_h = mean[heights].idxmax(axis=1)
    auroc = (df.groupby(["window_sec", "patient", "model"])["auroc"].mean()
             .groupby(["window_sec", "model"]).mean().unstack()[models])
    table = lambda t: ["| Window | " + " | ".join(models) + " |",
                       "|---|" + "---|" * len(models),
                       *[f"| {w} s | " + " | ".join(f"{v:.3f}" for v in t.loc[w]) + " |"
                         for w in windows], ""]
    groups = GROUPS[montage][0]
    return [
        f"## {montage.capitalize()} ({N_CHANNELS[montage]} channels)\n",
        "Grouped chains: " + "; ".join(f"{k} {v}" for k, v in groups.items()) + ".\n",
        "### F1, mean over patients\n",
        f1_table(mean, by_window.var(),
                 "Mean of the per-patient fold means ± variance across patients. "
                 "Best mean per window in bold.", cols=models, label="{}"),
        "### Median F1 over patients\n", *table(by_window.median()),
        "### AUROC, mean over patients\n", *table(auroc),
        "### Grouped vs. best kernel height\n",
        "Best h = highest mean F1 per window, picked on the test folds (optimistic for h). "
        "Difference in mean F1 (positive = grouped better), patients where grouped is better, "
        "two-sided Wilcoxon signed-rank p over patients (little power with few patients).\n",
        paired_table(pm, windows, best_h),
        "### Per patient\n",
        "F1 averaged over folds. Best model per patient and window in bold.\n",
        per_patient_table(pm, windows, models),
    ]


def write_report(df, results_dir=RESULTS_DIR):
    _, final_csv, report_md = result_paths(results_dir)
    patients = list(dict.fromkeys(df["patient"]))
    parts = [
        "# Experiment A on Siena Scalp: grouped kernel vs. kernel height\n",
        f"`SpatialNet`, recording CV ({N_SPLITS} folds, leave-one-recording-out when a patient "
        f"has fewer recordings), patients {', '.join(patients)}, same folds, splits and seeds "
        "for every model. See `experiments/experiment_a_siena.md`.\n",
        f"Per-fold rows with all metrics: `{final_csv.name}`. "
        "Generated by `experiments/experiment_a_siena.py`.\n",
    ]
    for montage in [m for m in MONTAGES if m in set(df["montage"])]:
        parts += montage_section(df[df["montage"] == montage], montage)
    report_md.write_text("\n".join(parts))


def main(montages=MONTAGES, windows=WINDOWS, heights=KERNEL_HEIGHTS,
         results_dir=RESULTS_DIR, patients=PATIENTS):
    partial_csv, final_csv, report_md = result_paths(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(partial_csv).to_dict("records") if partial_csv.exists() else []
    done = {(r["montage"], r["patient"], r["window_sec"], r["model"], r["fold"]) for r in rows}

    for montage in montages:
        specs = model_specs(montage, heights)
        for window_sec in windows:
            for patient in patients:
                X, y, rec = rcv.load_patient(patient, window_sec, montage)
                folds = list(rcv.recording_folds(y, rec, N_SPLITS, seed=SEED))
                for fold, (tr, te) in enumerate(folds):
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X[tr], y[tr], test_size=VAL_FRAC, stratify=y[tr],
                        random_state=SEED + fold)
                    for name, kwargs in specs:
                        if (montage, patient, window_sec, name, fold) in done:
                            continue
                        t0 = time.time()
                        set_seed(SEED + fold)
                        model, best_epoch, _ = train_model(
                            SpatialNet(**kwargs), X_tr, y_tr, X_val, y_val, CFG)
                        prob = predict_proba(model, X[te])
                        rows.append({
                            "montage": montage, "patient": patient, "window_sec": window_sec,
                            "model": name, "fold": fold, "n_folds": len(folds),
                            "test_recordings": ";".join(sorted(set(rec[te]))),
                            "n_train": len(y_tr), "n_test": len(te), "best_epoch": best_epoch,
                            **confusion_counts(y[te], (prob >= 0.5).astype(int)),
                            "auroc": (roc_auc_score(y[te], prob) if len(set(y[te])) == 2
                                      else np.nan),
                        })
                        pd.DataFrame(rows).to_csv(partial_csv, index=False)   # checkpoint
                        print(f"{montage[:3]} {window_sec:>2}s {patient} {name:<7} fold {fold}: "
                              f"F1={add_metrics(pd.DataFrame(rows[-1:]))['f1'].iloc[0]:.3f} "
                              f"({time.time() - t0:.0f}s)", flush=True)

    df = add_metrics(pd.DataFrame(rows))
    df.to_csv(final_csv, index=False)
    write_report(df, results_dir)
    print("\n" + report_md.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--montages", nargs="+", default=MONTAGES, choices=MONTAGES)
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    parser.add_argument("--heights", nargs="+", default=KERNEL_HEIGHTS,
                        help="kernel heights; 'full' = all channels of the montage")
    parser.add_argument("--patients", nargs="+", default=PATIENTS,
                        help="patient names, or 'all' for every patient with >= 2 recordings")
    parser.add_argument("--out", default=RESULTS_DIR.name,
                        help="results subfolder name inside experiments/results/")
    args = parser.parse_args()
    patients = ([p for p in rcv.available_patients(min(args.windows))
                 if p not in rcv.SINGLE_RECORDING_PATIENTS]
                if args.patients == ["all"] else args.patients)
    main(args.montages, args.windows, args.heights, RESULTS_ROOT / args.out, patients)
