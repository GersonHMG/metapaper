"""
Experiment B3: long temporal context and event-level metrics on continuous EEG.
See experiment_b3_long_context.md.

Per patient and fold (datasets/continuous.py: seizure recordings split as in B1,
seizure-free recordings dealt round-robin to the folds):
  1. Encoder: AsymSETNetGroupedSegments(n_segments=1) trained on 1 s segments of
     the training recordings (all seizure seconds + 5x normal seconds), validation
     from ~300 s time blocks. Its per-second probability is the no-context baseline.
  2. Freeze it; embed every second of every recording (32 features per second).
  3. TemporalHead (TCN over embeddings, receptive field 13/29/61/125 s), causal and
     non-causal, trained on 240 s sequences of training seconds (all sequences with
     seizure + 5x normal ones), per-second focal loss, early stopping on validation.
  4. Every model runs over whole recordings. Threshold = best per-second F1 on the
     validation seconds. Test: per-second counts (all, boundary within +-2 s) and
     events (positive runs, gaps < 10 s merged): seizures detected, false alarms,
     latency.
Rows are checkpointed per (patient, fold), so a rerun resumes.

    cd /home/gmarihuan/metapaper && python -m experiments.experiment_b3_long_context --patients all
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from datasets import continuous as ct
from datasets.transition import time_block_split, boundary_mask
from experiments.experiment_a1_kernel_height import RESULTS_ROOT, result_paths
from experiments.training import DEVICE, TrainConfig, set_seed, train_model, confusion_counts
from models.asymsetnet_grouped_segments import AsymSETNetGroupedSegments
from models.temporal_head import TemporalHead, receptive_field

PATIENTS = ["chb01", "chb18", "chb06"]
TCN_LAYERS = [2, 3, 4, 5]            # receptive field 13, 29, 61, 125 s
CAUSAL = [True, False]
N_SPLITS = 5
MAX_RATIO = 5                        # normal samples <= 5 x seizure samples
BLOCK_SEC = 300                      # validation time-block length (s)
VAL_FRAC = 0.2
SEQ_LEN = 240                        # stage-2 training sequence length (s), < BLOCK_SEC
SEQ_STRIDE = 8                       # stage-2 sequence stride (s)
BOUNDARY_K = 2
MERGE_GAP = 10                       # events closer than this (s) are merged
SEED = 0
CFG = TrainConfig()
RESULTS_DIR = RESULTS_ROOT / "b3_long_context"
COUNTS = ["TP", "FP", "TN", "FN"]
_saved_probs = None                  # dict model -> test-second probabilities, when saving
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------
def seconds_to_X(X, secs):
    """(n,) second indices -> (n, 21, 256) float32 from the (21, S*256) memmap."""
    secs = np.asarray(secs)
    order = np.argsort(secs)                     # sorted reads are faster on a memmap
    X3 = X.reshape(X.shape[0], -1, ct.SFREQ)
    out = np.empty((len(secs), X.shape[0], ct.SFREQ), dtype=np.float32)
    for i in range(0, len(secs), 4096):
        chunk = order[i:i + 4096]
        out[chunk] = np.asarray(X3[:, secs[chunk], :], dtype=np.float32).transpose(1, 0, 2)
    return out


def sample_seconds(sec_label, mask, rng):
    """All seizure seconds in `mask` + up to MAX_RATIO x as many normal seconds."""
    pos = np.flatnonzero(mask & (sec_label == 1))
    neg = np.flatnonzero(mask & (sec_label == 0))
    neg = rng.choice(neg, size=min(len(neg), MAX_RATIO * max(len(pos), 1)), replace=False)
    return np.sort(np.concatenate([pos, neg]))


def sequence_starts(rec_bounds, recs, mask, sec_label, rng):
    """Starts of SEQ_LEN sequences inside one recording with every second in
    `mask`: all with a seizure second + up to MAX_RATIO x as many without."""
    starts = []
    for r in recs:
        s, e = rec_bounds[r]
        starts.append(np.arange(s, e - SEQ_LEN + 1, SEQ_STRIDE))
    starts = np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)
    if len(starts) == 0:
        return starts
    idx = starts[:, None] + np.arange(SEQ_LEN)
    starts = starts[mask[idx].all(axis=1)]
    has_sz = sec_label[starts[:, None] + np.arange(SEQ_LEN)].any(axis=1)
    pos, neg = starts[has_sz], starts[~has_sz]
    neg = rng.choice(neg, size=min(len(neg), MAX_RATIO * max(len(pos), 1)), replace=False)
    return np.sort(np.concatenate([pos, neg]))


@torch.no_grad()
def embed_patient(encoder, X, n_sec, batch=4096):
    """Frozen-encoder embeddings (n_sec, D) and no-context probabilities (n_sec,)."""
    encoder.eval()
    feats, probs = [], []
    for a in range(0, n_sec, batch):
        b = min(n_sec, a + batch)
        x = torch.from_numpy(np.asarray(X[:, a * ct.SFREQ:b * ct.SFREQ], dtype=np.float32))
        x = x.view(X.shape[0], b - a, ct.SFREQ).transpose(0, 1).to(DEVICE)   # (n, 21, 256)
        f = encoder.spatial_module(x.unsqueeze(1))                             # (n, D)
        feats.append(f.cpu())
        probs.append(torch.sigmoid(encoder.classifier(f).squeeze(-1)).cpu())
    return torch.cat(feats).numpy(), torch.cat(probs).numpy()


def moving_average(p, rec_bounds, width, causal):
    """Non-learned control: mean of p over the last `width` seconds (causal) or
    the centred `width` seconds, inside each recording (shorter at the edges)."""
    out = np.empty_like(p)
    for s, e in rec_bounds:
        c = np.concatenate([[0.0], np.cumsum(p[s:e], dtype=np.float64)])
        t = np.arange(e - s)
        lo = t - width + 1 if causal else t - width // 2
        hi = t + 1 if causal else t + width // 2 + 1
        lo, hi = np.clip(lo, 0, e - s), np.clip(hi, 0, e - s)
        out[s:e] = (c[hi] - c[lo]) / (hi - lo)
    return out


@torch.no_grad()
def run_over_recordings(model, E, rec_bounds):
    """Per-second probabilities of a TemporalHead run over each whole recording."""
    model.eval()
    p = np.empty(len(E), dtype=np.float32)
    for s, e in rec_bounds:
        x = torch.from_numpy(E[s:e].T[None]).to(DEVICE)                       # (1, D, L)
        p[s:e] = torch.sigmoid(model(x))[0].cpu().numpy()
    return p


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def best_threshold(y, p):
    if y.sum() == 0:
        return 0.5
    f1s = [(2 * ((p >= t) & (y == 1)).sum()) / max(1, (p >= t).sum() + y.sum()) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(f1s))])


def runs(binary):
    """[start, end) of runs of 1s."""
    d = np.diff(np.concatenate([[0], binary.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def event_metrics(y, pred):
    """One recording: (n_seizures, n_detected, latencies, n_false_alarms)."""
    events = []
    for s, e in runs(pred):                       # merge events closer than MERGE_GAP
        if events and s - events[-1][1] < MERGE_GAP:
            events[-1] = (events[-1][0], e)
        else:
            events.append((s, e))
    seizures = runs(y)
    detected, latencies = 0, []
    for s, e in seizures:
        hits = [ev for ev in events if ev[0] < e and ev[1] > s]
        if hits:
            detected += 1
            first_pos = min(max(ev[0], s) for ev in hits)    # first detected second in the seizure
            latencies.append(int(first_pos - s))
    false_alarms = sum(1 for ev in events if not any(ev[0] < e and ev[1] > s for s, e in seizures))
    return len(seizures), detected, latencies, false_alarms


def evaluate(name, p, d, test_recs, val_mask, bmask):
    """One row of test metrics for model `name` (probabilities p over all seconds)."""
    y = d["sec_label"]
    thr = best_threshold(y[val_mask], p[val_mask])
    test = np.isin(d["sec_rec"], test_recs)
    if _saved_probs is not None:
        _saved_probs[name] = p[test].astype(np.float16)
    pred = (p >= thr).astype(np.int8)
    row = {"model": name, "threshold": thr,
           **confusion_counts(y[test], pred[test]),
           **{f"b{k}": v for k, v in confusion_counts(y[test & bmask], pred[test & bmask]).items()},
           "auroc": roc_auc_score(y[test], p[test]) if len(np.unique(y[test])) == 2 else np.nan}
    n_sz = n_det = n_fa = 0
    lat = []
    for r in test_recs:
        s, e = d["rec_bounds"][r]
        a, b, l, f = event_metrics(y[s:e], pred[s:e])
        n_sz, n_det, n_fa = n_sz + a, n_det + b, n_fa + f
        lat += l
    hours = sum(d["rec_bounds"][r, 1] - d["rec_bounds"][r, 0] for r in test_recs) / 3600
    row.update({"n_seizures": n_sz, "n_detected": n_det, "n_false_alarms": n_fa,
                "test_hours": hours, "latencies": ";".join(map(str, lat))})
    return row


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
MODEL_ORDER = (["none"]
               + [f"{receptive_field(n)}s {'causal' if c else 'non-causal'}{s}"
                  for s in (" avg", "") for n in TCN_LAYERS for c in CAUSAL])


def f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    if np.ndim(denom) == 0:
        return 2 * tp / denom if denom else np.nan
    return 2 * tp / denom.where(denom > 0)


def write_report(df, results_dir=RESULTS_DIR):
    _, final_csv, report_md = result_paths(results_dir)
    patients = list(dict.fromkeys(df["patient"]))
    models = [m for m in MODEL_ORDER if m in set(df["model"])]

    def lat_all(g):
        v = [int(x) for s in g.dropna().astype(str) for x in s.split(";") if x not in ("", "nan")]
        return v

    s = df.groupby("model")[COUNTS + ["bTP", "bFP", "bTN", "bFN", "n_seizures", "n_detected",
                                      "n_false_alarms", "test_hours"]].sum()
    lat = df.groupby("model")["latencies"].apply(lat_all)
    pooled = pd.DataFrame({
        "F1 (all s)": f1(s.TP, s.FP, s.FN),
        "F1 (boundary s)": f1(s.bTP, s.bFP, s.bFN),
        "Seizures detected": s.n_detected / s.n_seizures,
        "False alarms / h": s.n_false_alarms / s.test_hours,
        "Median latency (s)": lat.apply(lambda v: np.median(v) if v else np.nan),
    }).loc[models]

    pp = df.groupby(["patient", "model"])[COUNTS + ["n_false_alarms", "test_hours",
                                                    "n_detected", "n_seizures"]].sum()
    per_patient = pd.DataFrame({"f1": f1(pp.TP, pp.FP, pp.FN),
                                "fa_h": pp.n_false_alarms / pp.test_hours,
                                "sens": pp.n_detected / pp.n_seizures}).unstack("model")

    def fmt(c, v):
        if pd.isna(v):
            return "—"
        if c == "Seizures detected":
            return f"{100 * v:.1f} %"
        if c == "Median latency (s)":
            return f"{v:.0f}"
        return f"{v:.3f}" if "F1" in c else f"{v:.2f}"

    lines = ["| Model | " + " | ".join(pooled.columns) + " |", "|---|" + "---|" * len(pooled.columns)]
    for m in models:
        lines.append(f"| {m} | " + " | ".join(fmt(c, pooled.loc[m, c]) for c in pooled.columns) + " |")

    paired = ["| Model | F1 − none | wins | p | FA/h − none | fewer FA | p |",
              "|---|---|---|---|---|---|---|"]
    for m in models[1:]:
        cells = []
        for metric, better in (("f1", 1), ("fa_h", -1)):
            pair = per_patient[metric][["none", m]].dropna()
            dlt = pair[m] - pair["none"]
            p = wilcoxon(pair[m], pair["none"]).pvalue if len(dlt) >= 5 and (dlt != 0).any() else np.nan
            cells += [f"{dlt.mean():+.3f}" if metric == "f1" else f"{dlt.mean():+.2f}",
                      f"{(better * dlt > 0).sum()} / {len(dlt)}",
                      "—" if np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")]
        paired.append(f"| {m} | " + " | ".join(cells) + " |")

    vs_avg = ["| TCN | F1 − avg | wins | p | FA/h − avg | fewer FA | p |",
              "|---|---|---|---|---|---|---|"]
    for m in models:
        if m == "none" or m.endswith(" avg") or f"{m} avg" not in models:
            continue
        cells = []
        for metric, better in (("f1", 1), ("fa_h", -1)):
            pair = per_patient[metric][[f"{m} avg", m]].dropna()
            dlt = pair[m] - pair[f"{m} avg"]
            p = wilcoxon(pair[m], pair[f"{m} avg"]).pvalue if len(dlt) >= 5 and (dlt != 0).any() else np.nan
            cells += [f"{dlt.mean():+.3f}" if metric == "f1" else f"{dlt.mean():+.2f}",
                      f"{(better * dlt > 0).sum()} / {len(dlt)}",
                      "—" if np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")]
        vs_avg.append(f"| {m} | " + " | ".join(cells) + " |")

    pt = ["| Patient | Hours | Seizures | " + " | ".join(f"F1 {m}" for m in models) + " |",
          "|---|---|---|" + "---|" * len(models)]
    for p in patients:
        row = per_patient["f1"].loc[p, models]
        pt.append(f"| {p} | {pp.loc[(p, 'none'), 'test_hours']:.1f} | "
                  f"{int(pp.loc[(p, 'none'), 'n_seizures'])} | " + " | ".join(
                      f"**{v:.3f}**" if v == row.max() else f"{v:.3f}" for v in row) + " |")
    pfa = ["| Patient | " + " | ".join(f"FA/h {m}" for m in models) + " |",
           "|---|" + "---|" * len(models)]
    for p in patients:
        row = per_patient["fa_h"].loc[p, models]
        pfa.append(f"| {p} | " + " | ".join(
            f"**{v:.2f}**" if v == row.min() else f"{v:.2f}" for v in row) + " |")

    total_h = df[df.model == "none"]["test_hours"].sum()
    parts = [
        "# Experiment B3: Long temporal context, continuous EEG\n",
        "Frozen grouped spatial encoder (per-second classifier, its own output = **none**, no "
        "context). **avg** rows: non-learned moving average of the `none` probabilities over the "
        "same window (control). TCN rows: `TemporalHead` TCN over the per-second embeddings with receptive field "
        "13/29/61/125 s, causal (past only) and non-causal (both sides). Evaluated on whole test "
        f"recordings, seizure-free ones included: {len(patients)} patients, {total_h:.0f} test "
        "hours. Threshold per model and fold = best per-second F1 on validation seconds. "
        "See `experiments/experiment_b3_long_context.md`.\n",
        f"Events: positive seconds, gaps < {MERGE_GAP} s merged. A seizure is detected if an "
        "event overlaps it; latency = first detected second − onset; an event overlapping no "
        f"seizure is a false alarm. Boundary seconds: within ±{BOUNDARY_K} s of an onset or "
        f"offset. Per-fold rows: `{final_csv.name}`.\n",
        "## Pooled over patients and folds\n",
        "\n".join(lines) + "\n",
        "## Paired comparison with no context\n",
        "Per-patient F1 (all seconds) and false alarms per hour: mean difference, patients "
        "improved, two-sided Wilcoxon signed-rank p-value (— with fewer than 5 patients). No "
        "correction for multiple comparisons.\n",
        "\n".join(paired) + "\n",
        "## TCN vs. moving average with the same context\n",
        "Same comparison, each TCN against the non-learned moving average of the no-context "
        "probabilities over the same window (`avg` rows). Positive F1 / fewer FA = the learned "
        "temporal layer does more than averaging.\n",
        "\n".join(vs_avg) + "\n",
        "## Per patient\n",
        "F1 over all test seconds, best per row in bold.\n",
        "\n".join(pt) + "\n",
        "False alarms per hour, lowest per row in bold.\n",
        "\n".join(pfa) + "\n",
    ]
    report_md.write_text("\n".join(parts))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run_fold(d, fold, train_recs, test_recs):
    y = d["sec_label"]
    rng = np.random.default_rng(SEED + fold)
    split = time_block_split(y, d["rec_bounds"], train_recs, BLOCK_SEC, VAL_FRAC, seed=SEED + fold)
    tr_mask, val_mask = split == 0, split == 1
    bmask = boundary_mask(y, d["rec_bounds"], BOUNDARY_K)
    rows, t0 = [], time.time()

    # 1. Encoder (= no-context model)
    tr_s, val_s = sample_seconds(y, tr_mask, rng), sample_seconds(y, val_mask, rng)
    set_seed(SEED + fold)
    encoder, best_epoch, _ = train_model(
        AsymSETNetGroupedSegments(n_segments=1),
        seconds_to_X(d["X"], tr_s), y[tr_s], seconds_to_X(d["X"], val_s), y[val_s], CFG)
    encoder.freeze_spatial()

    # 2. Embeddings of every second
    E, p_none = embed_patient(encoder, d["X"], len(y))
    rows.append({**evaluate("none", p_none, d, test_recs, val_mask, bmask), "best_epoch": best_epoch})
    for n_layers in TCN_LAYERS:                  # non-learned control, same context widths
        for causal in CAUSAL:
            rf = receptive_field(n_layers)
            rows.append(evaluate(f"{rf}s {'causal' if causal else 'non-causal'} avg",
                                 moving_average(p_none, d["rec_bounds"], rf, causal),
                                 d, test_recs, val_mask, bmask))

    # 3. Temporal heads
    tr_starts = sequence_starts(d["rec_bounds"], train_recs, tr_mask, y, rng)
    val_starts = sequence_starts(d["rec_bounds"], train_recs, val_mask, y, rng)
    if len(val_starts) == 0:                     # no validation block long enough
        val_starts = rng.choice(tr_starts, size=max(1, len(tr_starts) // 5), replace=False)
    seq = lambda st: (np.ascontiguousarray(E[st[:, None] + np.arange(SEQ_LEN)].transpose(0, 2, 1)),
                      y[st[:, None] + np.arange(SEQ_LEN)])
    (E_tr, Y_tr), (E_val, Y_val) = seq(tr_starts), seq(val_starts)
    for n_layers in TCN_LAYERS:
        for causal in CAUSAL:
            set_seed(SEED + fold)
            head, best_epoch, _ = train_model(
                TemporalHead(in_dim=E.shape[1], n_layers=n_layers, causal=causal),
                E_tr, Y_tr, E_val, Y_val, CFG)
            name = f"{receptive_field(n_layers)}s {'causal' if causal else 'non-causal'}"
            p = run_over_recordings(head, E, d["rec_bounds"])
            rows.append({**evaluate(name, p, d, test_recs, val_mask, bmask), "best_epoch": best_epoch})
    for r in rows:
        r.update({"fold": fold, "n_train_seconds": len(tr_s), "n_train_seq": len(tr_starts),
                  "test_recordings": ";".join(map(str, d["rec_names"][test_recs]))})
    return rows, time.time() - t0


def main(results_dir=RESULTS_DIR, patients=PATIENTS, save_probs=False):
    """save_probs: also write probs/chbXX_foldK.npz with every model's per-second
    probabilities on the test seconds (for operating-point analysis)."""
    global _saved_probs
    partial_csv, final_csv, report_md = result_paths(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if save_probs:
        (results_dir / "probs").mkdir(exist_ok=True)
    rows = pd.read_csv(partial_csv).to_dict("records") if partial_csv.exists() else []
    done = {(r["patient"], r["fold"]) for r in rows}

    for patient in patients:
        d = ct.load_patient(patient)
        folds = list(ct.split_folds(d, N_SPLITS, seed=SEED))
        for fold, (train_recs, test_recs) in enumerate(folds):
            if (patient, fold) in done:
                continue
            _saved_probs = {} if save_probs else None
            fold_rows, dt = run_fold(d, fold, train_recs, test_recs)
            if save_probs:
                test = np.isin(d["sec_rec"], test_recs)
                np.savez_compressed(results_dir / "probs" / f"{patient}_fold{fold}.npz",
                                    y=d["sec_label"][test], rec=d["sec_rec"][test],
                                    rec_names=d["rec_names"], models=np.array(list(_saved_probs)),
                                    p=np.stack(list(_saved_probs.values())))
            rows += [{"patient": patient, **r} for r in fold_rows]
            pd.DataFrame(rows).to_csv(partial_csv, index=False)        # checkpoint
            summary = "  ".join(f"{r['model'].replace(' non-causal', 'nc').replace(' causal', 'c')}"
                                f"={f1(r['TP'], r['FP'], r['FN']):.2f}/{r['n_false_alarms']}"
                                for r in fold_rows)
            print(f"{patient} fold {fold} ({dt:.0f}s) F1/FA: {summary}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(final_csv, index=False)
    write_report(df, results_dir)
    print("\n" + report_md.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", nargs="+", default=PATIENTS,
                        help="patient names, or 'all' for every patient with data")
    parser.add_argument("--out", default=RESULTS_DIR.name,
                        help="results subfolder name inside experiments/results/")
    parser.add_argument("--save-probs", action="store_true",
                        help="save test-second probabilities of every model (probs/)")
    args = parser.parse_args()
    patients = ct.available_patients() if args.patients == ["all"] else args.patients
    main(RESULTS_ROOT / args.out, patients, args.save_probs)
