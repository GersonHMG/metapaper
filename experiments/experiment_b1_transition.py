"""
Experiment B1: does a TCN help label 1 s segments in windows that cross seizure
onsets and offsets? See experiment_b1_transition.md.

Data and protocol: datasets/transition.py -> fold_windows (same as
transition_segment_experiment.ipynb): folds by recording, stride-1 train/val
windows with validation from ~60 s time blocks, non-overlapping test tiles
(tiles touching a boundary always kept), all-normal windows capped at 5x.
Model: AsymSETNetGroupedSegments (as Experiment B), 1 s segments, non-causal TCN
with 0-3 layers, focal loss on every segment.
Every model is also evaluated with the segments of each test window shuffled
(shuffle test); predictions are mapped back so every segment is scored against
its own label. Every (patient, window, layers, fold) row is checkpointed.

    cd /home/gmarihuan/metapaper && python -m experiments.experiment_b1_transition
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from datasets import transition as tr
from experiments.experiment_a1_kernel_height import RESULTS_ROOT, result_paths, f1_table
from experiments.training import DEVICE, TrainConfig, set_seed, train_model, confusion_counts
from models.asymsetnet_grouped_segments import AsymSETNetGroupedSegments

PATIENTS = ["chb01", "chb18", "chb06"]
WINDOWS = [3, 5, 8, 10]              # 1 s segments -> n_segments = window_sec
TCN_LAYERS = [0, 1, 2, 3]
TCN_WIDTH = 32
N_SPLITS = 5
STRIDE = 1                           # train/val window stride (s)
BLOCK_SEC = 60                       # validation time-block length (s)
VAL_FRAC = 0.2
MAX_RATIO = 5                        # all-normal windows <= 5 x windows with seizure
BOUNDARY_K = 2                       # boundary segments: within +-2 s of an onset/offset
THRESHOLD = 0.5
SEED = 0
CFG = TrainConfig()
RESULTS_DIR = RESULTS_ROOT / "b1_transition"
COUNTS = ["TP", "FP", "TN", "FN"]


@torch.no_grad()
def segment_proba(model, X, batch_size=256):
    """Per-segment probabilities, shape (n, n_segments)."""
    model.eval()
    return torch.cat([torch.sigmoid(model(torch.from_numpy(X[i:i + batch_size]).to(DEVICE))).cpu()
                      for i in range(0, len(X), batch_size)]).numpy()


def random_perms(n, n_segments, seed):
    """One random order per window, never the original one."""
    rng = np.random.default_rng(seed)
    identity = np.arange(n_segments)
    perms = np.empty((n, n_segments), dtype=int)
    for i in range(n):
        p = rng.permutation(n_segments)
        while np.array_equal(p, identity):
            p = rng.permutation(n_segments)
        perms[i] = p
    return perms


def shuffled_proba(model, X, perms):
    """Probabilities of every segment when the window's segments are fed in the
    order `perms`, mapped back to the original segment positions."""
    n, c, t = X.shape
    k = perms.shape[1]
    Xs = np.take_along_axis(X.reshape(n, c, k, t // k), perms[:, None, :, None], axis=2)
    p_shuf = segment_proba(model, np.ascontiguousarray(Xs.reshape(n, c, t)))
    p = np.empty_like(p_shuf)
    np.put_along_axis(p, perms, p_shuf, axis=1)     # position perms[i, j] got p_shuf[i, j]
    return p


def counts(Y, prob, mask=None, prefix=""):
    y, p = (Y, prob) if mask is None else (Y[mask], prob[mask])
    return {f"{prefix}{k}": v for k, v in
            confusion_counts(y.ravel(), (p.ravel() >= THRESHOLD).astype(int)).items()}


def f1(tp, fp, fn):
    """F1 from counts; works on scalars and on pandas Series (NaN where undefined)."""
    denom = 2 * tp + fp + fn
    if np.ndim(denom) == 0:
        return 2 * tp / denom if denom else np.nan
    return 2 * tp / denom.where(denom > 0)


def write_report(df, results_dir=RESULTS_DIR):
    _, final_csv, report_md = result_paths(results_dir)
    patients = list(dict.fromkeys(df["patient"]))
    layers = sorted(df["tcn_layers"].unique())
    keys = ["window_sec", "tcn_layers"]

    def pooled(prefix):
        s = df.groupby(keys)[[prefix + c for c in COUNTS]].sum()
        return f1(s[prefix + "TP"], s[prefix + "FP"], s[prefix + "FN"]).unstack("tcn_layers")

    def per_patient(prefix):
        s = df.groupby(keys + ["patient"])[[prefix + c for c in COUNTS]].sum()
        return f1(s[prefix + "TP"], s[prefix + "FP"], s[prefix + "FN"])

    def plain_table(t, title):
        lines = [f"**{title}**", "", "| Window | " + " | ".join(f"{l} layers" for l in layers)
                 + " |", "|---|" + "---|" * len(layers)]
        for w in t.index:
            best = t.loc[w].max()
            lines.append(f"| {w} s | " + " | ".join(
                f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in t.loc[w, layers]) + " |")
        return "\n".join(lines) + "\n"

    def shuffle_table(prefix, title):
        a, s = pooled(prefix), pooled("shuf_" + prefix)
        lines = [f"**{title}**", "", "| Window | " + " | ".join(f"{l} layers" for l in layers)
                 + " |", "|---|" + "---|" * len(layers)]
        for w in a.index:
            lines.append(f"| {w} s | " + " | ".join(
                f"{a.loc[w, l]:.3f} → {s.loc[w, l]:.3f} ({a.loc[w, l] - s.loc[w, l]:+.3f})"
                for l in layers) + " |")
        return "\n".join(lines) + "\n"

    def paired_table(prefix, title):
        """Each TCN depth vs. 0 layers on per-patient (folds pooled) F1."""
        pp = per_patient(prefix).unstack("tcn_layers")
        lines = [f"**{title}**", "",
                 "| Window | " + " | ".join(f"{l} layers − 0 | wins | p" for l in layers[1:]) + " |",
                 "|---|" + "---|---|---|" * (len(layers) - 1)]
        for w in pp.index.get_level_values("window_sec").unique():
            cells = []
            for l in layers[1:]:
                pair = pp.loc[w, [0, l]].dropna()
                d = pair[l] - pair[0]
                p = wilcoxon(pair[l], pair[0]).pvalue if len(d) >= 5 and (d != 0).any() else np.nan
                cells += [f"{d.mean():+.3f}", f"{(d > 0).sum()} / {len(d)}",
                          "—" if np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")]
            lines.append(f"| {w} s | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    pp_b = per_patient("b")
    mean_b = pp_b.groupby(keys).mean().unstack("tcn_layers")
    var_b = pp_b.groupby(keys).var().unstack("tcn_layers")
    n_bnd = df.groupby(["patient"])["n_boundary_seg"].sum() / df.groupby("patient")["tcn_layers"].nunique() \
        / df.groupby("patient")["window_sec"].nunique()

    parts = [
        "# Experiment B1: Transition windows results\n",
        "`AsymSETNetGroupedSegments`, 1 s segments, non-causal TCN (width "
        f"{TCN_WIDTH}), one prediction per segment. Transition dataset (`fold_windows`), "
        f"{N_SPLITS} folds by recording, patients {', '.join(patients)}. Boundary segments: "
        f"within ±{BOUNDARY_K} s of an onset or offset. See `experiments/experiment_b1_transition.md`.\n",
        "**Pooled F1** is computed from TP/FP/FN summed over every patient and fold. It is the "
        "most reliable number for boundary segments, which are few per patient. "
        f"Per-fold counts: `{final_csv.name}`.\n",
        "Boundary segments per patient (all test folds, one window length): "
        + ", ".join(f"{p} {int(n_bnd[p])}" for p in patients) + ".\n",
        "## Boundary segments\n",
        plain_table(pooled("b"), "Pooled F1, boundary segments. Best per row in bold."),
        f1_table(mean_b, var_b, "Boundary F1, mean of per-patient F1 ± variance across patients",
                 cols=layers, label="{} layers"),
        "## All segments\n",
        plain_table(pooled(""), "Pooled F1, all test segments. Best per row in bold."),
        "## Paired comparison with 0 layers\n",
        "Per-patient F1 (folds pooled): mean difference (positive = TCN better), patients where "
        "the TCN is better, and the two-sided Wilcoxon signed-rank p-value over patients (— with "
        "fewer than 5 patients). No correction for multiple comparisons.\n",
        paired_table("b", "Boundary segments"),
        paired_table("", "All segments"),
        "## Shuffle test\n",
        "Pooled F1 with the segments of every test window in their original order → in a "
        "random order (drop). Each segment is scored against its own label. 0 layers must not "
        "change (control).\n",
        shuffle_table("b", "Boundary segments"),
        shuffle_table("", "All segments"),
        "## Per patient\n",
    ]
    for p in patients:
        parts.append(plain_table(pp_b.xs(p, level="patient").unstack("tcn_layers"),
                                 f"{p}: boundary F1 (folds pooled)"))
    report_md.write_text("\n".join(parts))


def main(results_dir=RESULTS_DIR, patients=PATIENTS, windows=WINDOWS, layers=TCN_LAYERS):
    partial_csv, final_csv, report_md = result_paths(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(partial_csv).to_dict("records") if partial_csv.exists() else []
    done = {(r["patient"], r["window_sec"], r["tcn_layers"], r["fold"]) for r in rows}

    for patient in patients:
        d = tr.load_patient(patient)
        folds = list(tr.split_folds(d, N_SPLITS, seed=SEED))
        for window_sec in windows:
            for fold, (train_recs, test_recs) in enumerate(folds):
                todo = [l for l in layers if (patient, window_sec, l, fold) not in done]
                if not todo:
                    continue
                w = tr.fold_windows(d, train_recs, test_recs, window_sec, stride=STRIDE,
                                    max_ratio=MAX_RATIO, block_sec=BLOCK_SEC, val_frac=VAL_FRAC,
                                    boundary_k=BOUNDARY_K, seed=SEED + fold)
                X_te, Y_te, B_te = w["test"]["X"], w["test"]["Y"], w["test"]["boundary"]
                perms = random_perms(len(X_te), window_sec, seed=SEED + fold)
                for n_layers in todo:
                    t0 = time.time()
                    set_seed(SEED + fold)
                    model, best_epoch, _ = train_model(
                        AsymSETNetGroupedSegments(n_segments=window_sec,
                                                  tcn_channels=(TCN_WIDTH,) * n_layers),
                        w["train"]["X"], w["train"]["Y"], w["val"]["X"], w["val"]["Y"], CFG)
                    prob = segment_proba(model, X_te)
                    prob_shuf = shuffled_proba(model, X_te, perms)
                    rows.append({
                        "patient": patient, "window_sec": window_sec, "tcn_layers": n_layers,
                        "fold": fold,
                        "test_recordings": ";".join(str(d["rec_names"][r]) for r in test_recs),
                        "n_train_win": len(w["train"]["Y"]), "n_val_win": len(w["val"]["Y"]),
                        "n_test_seg": Y_te.size, "n_boundary_seg": int(B_te.sum()),
                        "best_epoch": best_epoch,
                        **counts(Y_te, prob), **counts(Y_te, prob, B_te, "b"),
                        **counts(Y_te, prob_shuf, prefix="shuf_"),
                        **counts(Y_te, prob_shuf, B_te, "shuf_b"),
                        "auroc": roc_auc_score(Y_te.ravel(), prob.ravel())
                        if len(np.unique(Y_te)) == 2 else np.nan,
                    })
                    pd.DataFrame(rows).to_csv(partial_csv, index=False)   # checkpoint
                    r = rows[-1]
                    print(f"{patient} {window_sec:>2}s L={n_layers} fold {fold}: "
                          f"F1 all={f1(r['TP'], r['FP'], r['FN']):.3f} "
                          f"boundary={f1(r['bTP'], r['bFP'], r['bFN']):.3f} "
                          f"(shuffled {f1(r['shuf_bTP'], r['shuf_bFP'], r['shuf_bFN']):.3f}) "
                          f"({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(final_csv, index=False)
    write_report(df, results_dir)
    print("\n" + report_md.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", nargs="+", default=PATIENTS,
                        help="patient names, or 'all' for every patient with data")
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    parser.add_argument("--out", default=RESULTS_DIR.name,
                        help="results subfolder name inside experiments/results/")
    args = parser.parse_args()
    patients = tr.available_patients() if args.patients == ["all"] else args.patients
    main(RESULTS_ROOT / args.out, patients, args.windows)
