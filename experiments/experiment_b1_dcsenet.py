"""
Experiment B1 baseline: DCSENet (Aboyeji et al. 2025) on the transition data,
one prediction per 1 s segment. See experiment_b1_dcsenet.md.

Data and protocol: identical to experiment_b1_transition.py (fold_windows: folds
by recording, stride-1 train/val windows with validation from ~60 s time blocks,
non-overlapping test tiles, all-normal windows capped at 5x).
Model as in the paper: every channel's spectrogram (Algorithm 1, Hann, WL 1 s)
is an independent grayscale image (3x224x224); DCSENet's last FC outputs one
logit per segment. Training samples are (window, channel) images: Adam 1e-3,
batch 32, 20 epochs (paper), focal loss and best epoch on validation loss (B1).
Test: per-image probabilities, averaged over the 21 channels per segment.
Every (patient, window, fold) row is checkpointed.

    cd /home/gmarihuan/metapaper && python -m experiments.experiment_b1_dcsenet
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
from experiments.experiment_b1_transition import (
    PATIENTS, WINDOWS, N_SPLITS, STRIDE, BLOCK_SEC, VAL_FRAC, MAX_RATIO, BOUNDARY_K, SEED,
    COUNTS, counts, f1)
from experiments.training import DEVICE, FocalLoss, set_seed
from models.dcsenet import DCSENet, SpectrogramImage, to_rgb

EPOCHS = 20                          # paper: Adam, lr 1e-3, batch 32, 20 epochs
BATCH_SIZE = 32                      # images (window, channel) per step
LR = 1e-3
FOCAL_ALPHA, FOCAL_GAMMA = 0.75, 2.0
EVAL_BATCH = 1024                    # images per inference batch
RESULTS_DIR = RESULTS_ROOT / "b1_dcsenet"
B1_FOLDS_CSV = RESULTS_ROOT / "b1_transition_all" / "b1_transition_all_folds.csv"


def image_logits(model, spec, X, idx):
    """Logits (len(idx), W) for images idx = (window, channel) pairs. X on DEVICE is
    either raw EEG (n, 21, T) float, converted with `spec`, or cached images
    (n, 21, H, W) uint8 from cache_images."""
    x = X[idx[:, 0], idx[:, 1]]
    return model(to_rgb(x) if x.dtype == torch.uint8 else spec(x))


@torch.no_grad()
def cache_images(spec, X, batch=EVAL_BATCH):
    """Spectrogram grey levels of every (window, channel), (n, 21, H, W) uint8 on
    DEVICE: computed once instead of every epoch (identical pixels)."""
    flat = X.reshape(-1, X.shape[-1])
    out = torch.cat([spec.levels(flat[i:i + batch]) for i in range(0, len(flat), batch)])
    return out.reshape(X.shape[0], X.shape[1], *out.shape[1:])


def all_pairs(n_windows, n_channels):
    w, c = np.meshgrid(np.arange(n_windows), np.arange(n_channels), indexing="ij")
    return torch.from_numpy(np.stack([w.ravel(), c.ravel()], axis=1)).to(DEVICE)


@torch.no_grad()
def image_proba(model, spec, X):
    """Per-image segment probabilities, shape (n_windows, n_channels, W)."""
    model.eval()
    pairs = all_pairs(X.shape[0], X.shape[1])
    p = torch.cat([torch.sigmoid(image_logits(model, spec, X, pairs[i:i + EVAL_BATCH]))
                   for i in range(0, len(pairs), EVAL_BATCH)])
    return p.reshape(X.shape[0], X.shape[1], -1).cpu().numpy()


@torch.no_grad()
def eval_loss(model, spec, X, Y, loss_fn):
    model.eval()
    pairs = all_pairs(X.shape[0], X.shape[1])
    total = 0.0
    for i in range(0, len(pairs), EVAL_BATCH):
        b = pairs[i:i + EVAL_BATCH]
        total += loss_fn(image_logits(model, spec, X, b), Y[b[:, 0]]).item() * len(b)
    return total / len(pairs)


def train_dcsenet(X_tr, Y_tr, X_va, Y_va, seed, spec_kwargs=None, epochs=EPOCHS,
                  batch_size=BATCH_SIZE, lr=LR, weight_decay=0.0, patience=None,
                  focal_alpha=FOCAL_ALPHA, focal_gamma=FOCAL_GAMMA, cache=False):
    """Train on every (window, channel) image of X_tr (n, 21, T) with targets
    Y_tr (n, n_outputs); keep the epoch with the lowest validation loss
    (early stopping after `patience` epochs without improvement, if given).
    `spec_kwargs` configure SpectrogramImage. cache=True builds every train/val
    image once (uint8 on the GPU, 21 x 50 KB per window) instead of every epoch. Returns (model, spec, best_epoch, history)."""
    set_seed(seed)
    spec = SpectrogramImage(**(spec_kwargs or {})).to(DEVICE)
    model = DCSENet(n_outputs=Y_tr.shape[1]).to(DEVICE)
    loss_fn = FocalLoss(focal_alpha, focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_tr = torch.from_numpy(X_tr).to(DEVICE)
    Y_tr = torch.from_numpy(Y_tr.astype(np.float32)).to(DEVICE)
    X_va = torch.from_numpy(X_va).to(DEVICE)
    Y_va = torch.from_numpy(Y_va.astype(np.float32)).to(DEVICE)
    if cache:
        X_tr, X_va = cache_images(spec, X_tr), cache_images(spec, X_va)
    pairs = all_pairs(X_tr.shape[0], X_tr.shape[1])
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    best_loss, best_state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        order = pairs[torch.randperm(len(pairs), generator=gen, device=DEVICE)]
        train_loss = torch.zeros((), device=DEVICE)       # summed on the GPU: no per-step sync
        for i in range(0, len(order), batch_size):
            b = order[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(image_logits(model, spec, X_tr, b), Y_tr[b[:, 0]])
            loss.backward()
            optimizer.step()
            train_loss += loss.detach() * len(b)
        val_loss = eval_loss(model, spec, X_va, Y_va, loss_fn)
        history.append((train_loss.item() / len(order), val_loss))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif patience is not None and epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, spec, best_epoch, history


def write_report(df, results_dir=RESULTS_DIR):
    _, final_csv, report_md = result_paths(results_dir)
    patients = list(dict.fromkeys(df["patient"]))
    windows = sorted(df["window_sec"].unique())

    def pooled(d, prefix, keys=("window_sec",)):
        s = d.groupby(list(keys))[[prefix + c for c in COUNTS]].sum()
        return f1(s[prefix + "TP"], s[prefix + "FP"], s[prefix + "FN"])

    def rate(d, prefix):
        s = d.groupby("window_sec")[[prefix + c for c in COUNTS]].sum()
        return s[prefix + "TP"] / (s[prefix + "TP"] + s[prefix + "FN"]), \
            s[prefix + "TN"] / (s[prefix + "TN"] + s[prefix + "FP"])

    # B1 AsymSETNetGroupedSegments on the same patients and windows
    b1 = pd.read_csv(B1_FOLDS_CSV)
    b1 = b1[b1["patient"].isin(patients) & b1["window_sec"].isin(windows)]
    b1_pooled = {p: pooled(b1, p, ("window_sec", "tcn_layers")).unstack("tcn_layers")
                 for p in ("b", "")}
    best_layers = b1_pooled["b"].idxmax(axis=1)            # best TCN depth per window (boundary)

    lines = ["| Window | DCSENet boundary | AsymSET 0 layers | AsymSET best TCN (layers) | "
             "DCSENet all | AsymSET 0 layers | AsymSET best TCN |", "|---|---|---|---|---|---|---|"]
    dc = {p: pooled(df, p) for p in ("b", "")}
    for w in windows:
        L = best_layers.get(w)
        cell = lambda p, l: "—" if L is None else f"{b1_pooled[p].loc[w, l]:.3f}"
        lines.append(f"| {w} s | **{dc['b'][w]:.3f}** | {cell('b', 0)} | {cell('b', L)} ({L}) | "
                     f"**{dc[''][w]:.3f}** | {cell('', 0)} | {cell('', L)} |")
    comparison = "\n".join(lines) + "\n"

    def paired(prefix):
        """Per-patient F1 (folds pooled): DCSENet - AsymSETNet 0 layers."""
        a = pooled(df, prefix, ("window_sec", "patient"))
        b = pooled(b1[b1["tcn_layers"] == 0], prefix, ("window_sec", "patient"))
        out = ["| Window | DCSENet − AsymSET 0 layers | DCSENet wins | p |", "|---|---|---|---|"]
        for w in windows:
            pair = pd.concat([a.loc[w], b.loc[w]], axis=1, keys=["dc", "b1"]).dropna()
            d = pair["dc"] - pair["b1"]
            p = wilcoxon(pair["dc"], pair["b1"]).pvalue if len(d) >= 5 and (d != 0).any() else np.nan
            out.append(f"| {w} s | {d.mean():+.3f} | {(d > 0).sum()} / {len(d)} | "
                       + ("—" if np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")) + " |")
        return "\n".join(out) + "\n"

    pp_b = pooled(df, "b", ("window_sec", "patient"))
    pp_all = pooled(df, "", ("window_sec", "patient"))
    mean_var = pd.DataFrame({"boundary": pp_b.groupby("window_sec").mean(),
                             "all": pp_all.groupby("window_sec").mean()})
    var = pd.DataFrame({"boundary": pp_b.groupby("window_sec").var(),
                        "all": pp_all.groupby("window_sec").var()}).fillna(0)
    img_lines = ["| Window | F1 all | Sens. | Spec. | F1 boundary |", "|---|---|---|---|---|"]
    sens, spec = rate(df, "img_")
    img_f1, img_bf1 = pooled(df, "img_"), pooled(df, "img_b")
    for w in windows:
        img_lines.append(f"| {w} s | {img_f1[w]:.3f} | {sens[w]:.3f} | {spec[w]:.3f} | {img_bf1[w]:.3f} |")
    secs = df.groupby("window_sec")["train_sec"].sum()

    parts = [
        "# Experiment B1 baseline: DCSENet on transition windows\n",
        "DCSENet (Aboyeji et al. 2025), one grayscale spectrogram image per channel (Algorithm 1, "
        "Hann, WL 1 s, step 0.125 s), last FC = one logit per 1 s segment. Prediction per segment = "
        f"mean of the 21 channel probabilities. Transition dataset (`fold_windows`), {N_SPLITS} folds "
        f"by recording, patients {', '.join(patients)}. Boundary segments: within ±{BOUNDARY_K} s of "
        "an onset or offset. See `experiments/experiment_b1_dcsenet.md`.\n",
        f"Per-fold counts: `{final_csv.name}`. AsymSETNet numbers: `{B1_FOLDS_CSV.parent.name}` "
        "restricted to the same patients and windows.\n",
        "## Pooled F1 vs. AsymSETNetGroupedSegments (B1)\n",
        "Best TCN depth chosen per window on pooled boundary F1.\n",
        comparison,
        "## Paired comparison with AsymSETNet 0 layers\n",
        "Per-patient F1 (folds pooled), two-sided Wilcoxon over patients (— with fewer than 5).\n",
        "**Boundary segments**\n", paired("b"), "**All segments**\n", paired(""),
        f1_table(mean_var, var, "DCSENet, mean of per-patient F1 ± variance across patients",
                 cols=["boundary", "all"], label="{}", bold_best=False),
        "## Per image (paper style)\n",
        "Every (window, channel) image scored on its own, pooled over patients and folds.\n",
        "\n".join(img_lines) + "\n",
        "## Per patient\n",
        "| Patient | " + " | ".join(f"{w} s boundary | {w} s all" for w in windows) + " |",
        "|---|" + "---|---|" * len(windows),
    ]
    for p in patients:
        parts.append(f"| {p} | " + " | ".join(
            f"{pp_b.get((w, p), np.nan):.3f} | {pp_all.get((w, p), np.nan):.3f}" for w in windows) + " |")
    parts += ["", "## Training time\n", "| Window | train time (min, all folds) |", "|---|---|"]
    parts += [f"| {w} s | {secs[w] / 60:.1f} |" for w in windows]
    report_md.write_text("\n".join(parts) + "\n")


def main(results_dir=RESULTS_DIR, patients=PATIENTS, windows=WINDOWS):
    partial_csv, final_csv, report_md = result_paths(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(partial_csv).to_dict("records") if partial_csv.exists() else []
    done = {(r["patient"], r["window_sec"], r["fold"]) for r in rows}

    for patient in patients:
        d = tr.load_patient(patient)
        folds = list(tr.split_folds(d, N_SPLITS, seed=SEED))
        for window_sec in windows:
            for fold, (train_recs, test_recs) in enumerate(folds):
                if (patient, window_sec, fold) in done:
                    continue
                t0 = time.time()
                w = tr.fold_windows(d, train_recs, test_recs, window_sec, stride=STRIDE,
                                    max_ratio=MAX_RATIO, block_sec=BLOCK_SEC, val_frac=VAL_FRAC,
                                    boundary_k=BOUNDARY_K, seed=SEED + fold)
                model, spec, best_epoch, history = train_dcsenet(
                    w["train"]["X"], w["train"]["Y"], w["val"]["X"], w["val"]["Y"], SEED + fold)
                t_train = time.time() - t0
                X_te = torch.from_numpy(w["test"]["X"]).to(DEVICE)
                Y_te, B_te = w["test"]["Y"], w["test"]["boundary"]
                p_img = image_proba(model, spec, X_te)                   # (n, 21, W)
                prob = p_img.mean(axis=1)                                # (n, W)
                Y_img = np.broadcast_to(Y_te[:, None], p_img.shape)
                B_img = np.broadcast_to(B_te[:, None], p_img.shape)
                rows.append({
                    "patient": patient, "window_sec": window_sec, "fold": fold,
                    "test_recordings": ";".join(str(d["rec_names"][r]) for r in test_recs),
                    "n_train_win": len(w["train"]["Y"]), "n_val_win": len(w["val"]["Y"]),
                    "n_test_seg": Y_te.size, "n_boundary_seg": int(B_te.sum()),
                    "best_epoch": best_epoch,
                    "train_loss": history[best_epoch][0], "val_loss": history[best_epoch][1],
                    **counts(Y_te, prob), **counts(Y_te, prob, B_te, "b"),
                    **counts(Y_img, p_img, prefix="img_"), **counts(Y_img, p_img, B_img, "img_b"),
                    "auroc": roc_auc_score(Y_te.ravel(), prob.ravel())
                    if len(np.unique(Y_te)) == 2 else np.nan,
                    "train_sec": t_train,
                })
                pd.DataFrame(rows).to_csv(partial_csv, index=False)   # checkpoint
                del model, X_te
                torch.cuda.empty_cache()
                r = rows[-1]
                print(f"{patient} {window_sec:>2}s fold {fold}: "
                      f"F1 all={f1(r['TP'], r['FP'], r['FN']):.3f} "
                      f"boundary={f1(r['bTP'], r['bFP'], r['bFN']):.3f} "
                      f"auroc={r['auroc']:.3f} best_epoch={best_epoch} "
                      f"losses={' '.join(f'{a:.3f}/{b:.3f}' for a, b in history)} "
                      f"({r['n_train_win']} train win, {t_train:.0f}s)", flush=True)

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
