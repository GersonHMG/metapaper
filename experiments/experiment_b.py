"""
Experiment B: is the TCN worth it? See experiment_b.md.

AsymSETNetGroupedSegments (grouped spatial module per segment, non-causal TCN with
0-3 layers, Linear-ReLU-Linear head per segment). Window probability = mean of the
segment probabilities. Same patients, folds, train/val split, seeds and TrainConfig
as Experiment A. Every trained model is also evaluated with its segments shuffled
(shuffle test). Every (patient, window, segment, layers, fold) row is checkpointed,
so a rerun resumes.

    cd /home/gmarihuan/metapaper && python -m experiments.experiment_b --patients all
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
from experiments.experiment_a1_kernel_height import (PATIENTS, N_SPLITS, VAL_FRAC, SEED, CFG,
                                                     EXCLUDE, RESULTS_ROOT, result_paths,
                                                     f1_table)
from experiments.training import DEVICE, set_seed, train_model, confusion_counts, add_metrics
from models.asymsetnet_grouped_segments import AsymSETNetGroupedSegments

SFREQ = 256
# (window_sec, segment_sec): 1 s windows are left out (a single segment)
COMBOS = [(3, 1), (5, 1), (8, 1), (10, 1), (8, 2), (10, 2)]
TCN_LAYERS = [0, 1, 2, 3]
TCN_WIDTH = 32
RESULTS_DIR = RESULTS_ROOT / "b_tcn"
REFERENCE_DIR = RESULTS_ROOT / "a_grouped_all"            # SpatialNet(grouped), Experiment A


@torch.no_grad()
def window_proba(model, X, batch_size=256):
    """Window probabilities (mean of segment probabilities), shape (n,)."""
    model.eval()
    return torch.cat([model.predict_proba(torch.from_numpy(X[i:i + batch_size]).to(DEVICE)).cpu()
                      for i in range(0, len(X), batch_size)]).numpy()


def shuffle_segments(X, n_segments, seed):
    """Copy of X (n, C, T) with the segments of every window in a random order,
    never the original one."""
    rng = np.random.default_rng(seed)
    n, c, t = X.shape
    seg = X.reshape(n, c, n_segments, t // n_segments)
    identity = np.arange(n_segments)
    perms = np.empty((n, n_segments), dtype=int)
    for i in range(n):
        p = rng.permutation(n_segments)
        while np.array_equal(p, identity):
            p = rng.permutation(n_segments)
        perms[i] = p
    out = np.take_along_axis(seg, perms[:, None, :, None], axis=2)
    return out.reshape(n, c, t)


def scores(y, prob, prefix=""):
    counts = confusion_counts(y, (prob >= 0.5).astype(int))
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    out = {f"{prefix}{k}": v for k, v in counts.items()} if prefix else counts
    if prefix:
        out[f"{prefix}f1"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    out[f"{prefix}auroc"] = roc_auc_score(y, prob) if len(set(y)) == 2 else np.nan
    return out


def combo_label(w, s):
    return f"{w} s / {s} s seg"


def write_report(df, results_dir=RESULTS_DIR):
    _, final_csv, report_md = result_paths(results_dir)
    df = df.assign(combo=[combo_label(w, s) for w, s in zip(df.window_sec, df.segment_sec)])
    combos = [combo_label(w, s) for w, s in COMBOS if combo_label(w, s) in set(df.combo)]
    layers = sorted(df["tcn_layers"].unique())
    patients = list(dict.fromkeys(df["patient"]))

    pm = df.groupby(["combo", "patient", "tcn_layers"])["f1"].mean().unstack("tcn_layers")
    pm_shuf = df.groupby(["combo", "patient", "tcn_layers"])["shuf_f1"].mean().unstack("tcn_layers")
    mean = pm.groupby("combo").mean().loc[combos]
    var = pm.groupby("combo").var().loc[combos]

    # Experiment A reference: SpatialNet(grouped) per window, same patients
    ref_lines = []
    ref_csv = result_paths(REFERENCE_DIR)[1]
    if ref_csv.exists():
        ref = pd.read_csv(ref_csv)
        ref = ref[ref.patient.isin(patients)].groupby(["window_sec", "patient"])["f1"].mean()
        ref_mean, ref_var = ref.groupby("window_sec").mean(), ref.groupby("window_sec").var()
        ref_lines = ["**Reference: `SpatialNet(spatial=\"grouped\")`, Experiment A "
                     f"(`{REFERENCE_DIR.name}/`), whole window, no segments**", "",
                     "| Window | F1 |", "|---|---|"]
        ref_lines += [f"| {w} s | {ref_mean[w]:.3f} ± {ref_var[w]:.3f} |"
                      for w in sorted({w for w, _ in COMBOS}) if w in ref_mean.index]
        ref_lines.append("")

    paired = ["| Window / segment | " + " | ".join(f"{l} layers − 0 | wins | p" for l in layers[1:])
              + " |", "|---|" + "---|---|---|" * (len(layers) - 1)]
    for c in combos:
        cells = []
        for l in layers[1:]:
            pair = pm.loc[c, [0, l]].dropna()
            d = pair[l] - pair[0]
            p = wilcoxon(pair[l], pair[0]).pvalue if (d != 0).any() else np.nan
            cells += [f"{d.mean():+.3f}", f"{(d > 0).sum()} / {len(d)}",
                      "<0.001" if p < 0.001 else f"{p:.3f}"]
        paired.append(f"| {c} | " + " | ".join(cells) + " |")

    shuffle = ["| Window / segment | " + " | ".join(f"{l} layers: as is → shuffled (drop)"
                                                    for l in layers) + " |",
               "|---|" + "---|" * len(layers)]
    for c in combos:
        cells = []
        for l in layers:
            a, s = pm.loc[c, l].mean(), pm_shuf.loc[c, l].mean()
            cells.append(f"{a:.3f} → {s:.3f} ({a - s:+.3f})")
        shuffle.append(f"| {c} | " + " | ".join(cells) + " |")

    per_patient = ["| Patient | Window / segment | " + " | ".join(f"{l} layers" for l in layers)
                   + " |", "|---|---|" + "---|" * len(layers)]
    for p in patients:
        for c in combos:
            if (c, p) not in pm.index:
                continue
            row = pm.loc[(c, p)]
            per_patient.append(f"| {p} | {c} | " + " | ".join(
                f"**{v:.3f}**" if v == row.max() else f"{v:.3f}" for v in row[layers]) + " |")

    parts = [
        "# Experiment B: TCN layers results\n",
        "`AsymSETNetGroupedSegments`: grouped spatial module per segment, non-causal TCN "
        f"(width {TCN_WIDTH}), Linear-ReLU-Linear head per segment; window probability = mean of "
        f"the segment probabilities. Recording CV ({N_SPLITS} folds, leave-one-recording-out when "
        f"a patient has fewer recordings), {len(patients)} patients, chb16 excluded at 10 s. "
        "See `experiments/experiment_b.md`.\n",
        f"Per-fold rows with all metrics: `{final_csv.name}`. "
        "Cells are F1 mean ± variance across patients unless stated otherwise.\n",
        "## F1 by number of TCN layers\n",
        f1_table(mean, var, "Mean of the per-patient fold means ± variance across patients. "
                 "Best per row in bold.", cols=layers, label="{} layers",
                 row_header="Window / segment", row_label="{}"),
        *ref_lines,
        "## Paired comparison with 0 layers\n",
        "F1 difference (positive = TCN better), patients where the TCN is better, and the "
        "two-sided Wilcoxon signed-rank p-value over patients. No correction for multiple "
        "comparisons.\n",
        "\n".join(paired) + "\n",
        "## Shuffle test\n",
        "Mean F1 over patients with the segments of every test window in their original order "
        "and in a random order. 0 layers must not change (control).\n",
        "\n".join(shuffle) + "\n",
        "## Per patient\n",
        "F1 averaged over folds. Best number of layers per row in bold.\n",
        "\n".join(per_patient) + "\n",
    ]
    report_md.write_text("\n".join(parts))


def main(results_dir=RESULTS_DIR, patients=PATIENTS, combos=COMBOS, layers=TCN_LAYERS):
    partial_csv, final_csv, report_md = result_paths(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(partial_csv).to_dict("records") if partial_csv.exists() else []
    done = {(r["patient"], r["window_sec"], r["segment_sec"], r["tcn_layers"], r["fold"])
            for r in rows}

    for window_sec in sorted({w for w, _ in combos}):
        seg_secs = [s for w, s in combos if w == window_sec]
        for patient in patients:
            if patient in EXCLUDE.get(window_sec, []):
                continue
            X, y, rec = rcv.load_patient(patient, window_sec)
            folds = list(rcv.recording_folds(y, rec, N_SPLITS, seed=SEED))
            for fold, (tr, te) in enumerate(folds):
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X[tr], y[tr], test_size=VAL_FRAC, stratify=y[tr], random_state=SEED + fold)
                for seg_sec in seg_secs:
                    n_seg = window_sec // seg_sec
                    X_te_shuf = None
                    for n_layers in layers:
                        if (patient, window_sec, seg_sec, n_layers, fold) in done:
                            continue
                        if X_te_shuf is None:
                            X_te_shuf = shuffle_segments(X[te], n_seg, seed=SEED + fold)
                        t0 = time.time()
                        set_seed(SEED + fold)
                        model, best_epoch, _ = train_model(
                            AsymSETNetGroupedSegments(n_segments=n_seg,
                                                      tcn_channels=(TCN_WIDTH,) * n_layers),
                            X_tr, y_tr, X_val, y_val, CFG)
                        prob = window_proba(model, X[te])
                        prob_shuf = window_proba(model, X_te_shuf)
                        rows.append({
                            "patient": patient, "window_sec": window_sec,
                            "segment_sec": seg_sec, "n_segments": n_seg, "tcn_layers": n_layers,
                            "fold": fold, "n_folds": len(folds),
                            "test_recordings": ";".join(sorted(set(rec[te]))),
                            "n_train": len(y_tr), "n_test": len(te), "best_epoch": best_epoch,
                            **scores(y[te], prob), **scores(y[te], prob_shuf, prefix="shuf_"),
                        })
                        pd.DataFrame(rows).to_csv(partial_csv, index=False)   # checkpoint
                        last = add_metrics(pd.DataFrame(rows[-1:])).iloc[0]
                        print(f"{window_sec:>2}s/{seg_sec}s {patient} L={n_layers} fold {fold}: "
                              f"F1={last['f1']:.3f} shuffled={last['shuf_f1']:.3f} "
                              f"({time.time() - t0:.0f}s)", flush=True)

    df = add_metrics(pd.DataFrame(rows))
    df.to_csv(final_csv, index=False)
    write_report(df, results_dir)
    print("\n" + report_md.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", nargs="+", default=PATIENTS,
                        help="patient names, or 'all' for every patient with data")
    parser.add_argument("--windows", type=int, nargs="+", default=None,
                        help="restrict to these window lengths (default: all combos)")
    parser.add_argument("--out", default=RESULTS_DIR.name,
                        help="results subfolder name inside experiments/results/")
    args = parser.parse_args()
    patients = (rcv.available_patients(1) if args.patients == ["all"] else args.patients)
    combos = [c for c in COMBOS if args.windows is None or c[0] in args.windows]
    main(RESULTS_ROOT / args.out, patients, combos)
