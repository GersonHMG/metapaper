"""
Experiment B2: TCN + head trained on a frozen, pretrained spatial module.
See experiment_b2_frozen_spatial.md.

Same data, windows/segments, patients, folds, train/val split and seeds as
Experiment B (experiment_b.py). Inside every fold and window/segment combination:
  1. pretrain B's 0-layer model (AsymSETNetGroupedSegments, no TCN) -> row tcn_layers=0
  2. freeze its spatial module, drop its head
  3. for 1-3 layers: new non-causal TCN + new Linear-ReLU-Linear head, trained alone
Every model also gets the shuffle test. The report adds a comparison with
Experiment B (end-to-end training, results/b_tcn/) at the same number of layers.
Rows are checkpointed per (patient, window, segment, fold), so a rerun resumes.

    cd /home/gmarihuan/metapaper && python -m experiments.experiment_b2_frozen_spatial --patients all
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

from datasets import recording_cv as rcv
from experiments import experiment_b as b
from experiments.experiment_a1_kernel_height import (PATIENTS, N_SPLITS, VAL_FRAC, SEED, CFG,
                                                     EXCLUDE, RESULTS_ROOT, result_paths)
from experiments.training import set_seed, train_model, add_metrics
from models.asymsetnet_grouped_segments import AsymSETNetGroupedSegments

RESULTS_DIR = RESULTS_ROOT / "b2_frozen_spatial"
END_TO_END_DIR = RESULTS_ROOT / "b_tcn"                 # Experiment B


def frozen_vs_end_to_end(df):
    """Markdown: B2 (frozen) vs. B (end to end) at the same number of TCN layers."""
    b_csv = result_paths(END_TO_END_DIR)[1]
    if not b_csv.exists():
        return ""
    e2e = pd.read_csv(b_csv)
    keys = ["window_sec", "segment_sec", "patient", "tcn_layers"]
    fz = df.groupby(keys)["f1"].mean().rename("frozen")
    ee = e2e[e2e.patient.isin(set(df.patient))].groupby(keys)["f1"].mean().rename("end_to_end")
    m = pd.concat([fz, ee], axis=1).dropna().reset_index()
    layers = sorted(l for l in m.tcn_layers.unique() if l > 0)
    lines = ["| Window / segment | " + " | ".join(f"{l} layers: frozen − end-to-end | wins | p"
                                                   for l in layers) + " |",
             "|---|" + "---|---|---|" * len(layers)]
    for w, s in b.COMBOS:
        cells = []
        for l in layers:
            g = m[(m.window_sec == w) & (m.segment_sec == s) & (m.tcn_layers == l)]
            if g.empty:
                cells += ["—"] * 3
                continue
            d = g.frozen - g.end_to_end
            p = wilcoxon(g.frozen, g.end_to_end).pvalue if (d != 0).any() and len(d) >= 5 else np.nan
            cells += [f"{d.mean():+.3f}", f"{(d > 0).sum()} / {len(d)}",
                      "—" if np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")]
        lines.append(f"| {b.combo_label(w, s)} | " + " | ".join(cells) + " |")
    zero = m[m.tcn_layers == 0]
    return ("## Frozen (B2) vs. end-to-end (B)\n\n"
            "Same folds and seeds. Per-patient F1 difference (positive = freezing the spatial "
            "module is better), patients where frozen is better, Wilcoxon p over patients. "
            f"0 layers is the same model in both (mean |difference| {(zero.frozen - zero.end_to_end).abs().mean():.3f}, "
            "GPU run-to-run noise).\n\n" + "\n".join(lines) + "\n")


def write_report(df, results_dir=RESULTS_DIR):
    b.write_report(df, results_dir)                       # same tables as Experiment B
    report_md = result_paths(results_dir)[2]
    text = report_md.read_text()
    text = text.replace(
        "# Experiment B: TCN layers results\n",
        "# Experiment B2: TCN on a frozen spatial module\n\n"
        "**0 layers** = the pretrained model (B's 0-layer model). **1–3 layers** = its spatial "
        "module frozen (eval mode, no gradients), new TCN + head trained on top. "
        "See `experiments/experiment_b2_frozen_spatial.md`.\n", 1)
    text = text.replace("## Per patient\n", frozen_vs_end_to_end(df) + "\n## Per patient\n", 1)
    report_md.write_text(text)


def main(results_dir=RESULTS_DIR, patients=PATIENTS, combos=b.COMBOS):
    partial_csv, final_csv, report_md = result_paths(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(partial_csv).to_dict("records") if partial_csv.exists() else []
    done = {(r["patient"], r["window_sec"], r["segment_sec"], r["fold"]) for r in rows}

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
                    if (patient, window_sec, seg_sec, fold) in done:
                        continue
                    n_seg = window_sec // seg_sec
                    X_te_shuf = b.shuffle_segments(X[te], n_seg, seed=SEED + fold)
                    fold_rows = []
                    pretrained = None
                    for n_layers in b.TCN_LAYERS:
                        t0 = time.time()
                        set_seed(SEED + fold)
                        if n_layers == 0:
                            model = AsymSETNetGroupedSegments(n_segments=n_seg)
                        else:
                            model = AsymSETNetGroupedSegments.from_pretrained_spatial(
                                pretrained, tcn_channels=(b.TCN_WIDTH,) * n_layers)
                        model, best_epoch, _ = train_model(model, X_tr, y_tr, X_val, y_val, CFG)
                        if n_layers == 0:
                            pretrained = model
                        prob = b.window_proba(model, X[te])
                        prob_shuf = b.window_proba(model, X_te_shuf)
                        fold_rows.append({
                            "patient": patient, "window_sec": window_sec,
                            "segment_sec": seg_sec, "n_segments": n_seg, "tcn_layers": n_layers,
                            "fold": fold, "n_folds": len(folds),
                            "test_recordings": ";".join(sorted(set(rec[te]))),
                            "n_train": len(y_tr), "n_test": len(te), "best_epoch": best_epoch,
                            **b.scores(y[te], prob), **b.scores(y[te], prob_shuf, prefix="shuf_"),
                        })
                        last = add_metrics(pd.DataFrame(fold_rows[-1:])).iloc[0]
                        print(f"{window_sec:>2}s/{seg_sec}s {patient} L={n_layers} fold {fold}: "
                              f"F1={last['f1']:.3f} shuffled={last['shuf_f1']:.3f} "
                              f"({time.time() - t0:.0f}s)", flush=True)
                    rows += fold_rows
                    pd.DataFrame(rows).to_csv(partial_csv, index=False)   # checkpoint

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
    patients = rcv.available_patients(1) if args.patients == ["all"] else args.patients
    combos = [c for c in b.COMBOS if args.windows is None or c[0] in args.windows]
    main(RESULTS_ROOT / args.out, patients, combos)
