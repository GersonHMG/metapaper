"""
Experiment B3 operating points: compare models at the SAME seizure sensitivity.

Reads the test-second probabilities saved by
    python -m experiments.experiment_b3_long_context --patients all \
        --out b3_long_context_probs --save-probs
and, for every model, sweeps one global threshold over all patients and folds:
pooled seizure sensitivity (events, as in B3) vs. false alarms per hour. Reports
FA/h and mean latency at the highest threshold that still detects >= 80/90/95 %
of seizures, and a paired per-patient test (FA/h at those thresholds) of each TCN
against the moving average with the same context and against no context.
Writes operating_points.md and operating_curves.csv next to the probabilities.

    cd /home/gmarihuan/metapaper && python -m experiments.b3_operating_points
"""

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiments.experiment_a1_kernel_height import RESULTS_ROOT
from experiments.experiment_b3_long_context import MODEL_ORDER, event_metrics

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)
TARGETS = [0.80, 0.90, 0.95]
RESULTS_DIR = RESULTS_ROOT / "b3_long_context_probs"


def sweep_file(path):
    """Rows (patient, model, threshold, n_seizures, n_detected, n_fa, hours, latency_sum)."""
    f = np.load(path, allow_pickle=True)
    patient = path.stem.split("_fold")[0]
    y, rec, models, P = f["y"], f["rec"], f["models"], f["p"].astype(np.float32)
    cuts = np.flatnonzero(np.diff(rec)) + 1
    rec_slices = list(zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [len(rec)]])))
    hours = len(y) / 3600
    rows = []
    for m, p in zip(models, P):
        for t in THRESHOLDS:
            pred = (p >= t).astype(np.int8)
            n_sz = n_det = n_fa = lat = 0
            for a, b in rec_slices:
                s, dt, l, fa = event_metrics(y[a:b], pred[a:b])
                n_sz, n_det, n_fa, lat = n_sz + s, n_det + dt, n_fa + fa, lat + sum(l)
            rows.append((patient, str(m), t, n_sz, n_det, n_fa, hours, lat))
    return rows


def operating_point(curve, target):
    """Highest threshold with pooled sensitivity >= target (fewest false alarms)."""
    ok = curve[curve.sens >= target]
    return None if ok.empty else ok.loc[ok.threshold.idxmax()]


def main(results_dir=RESULTS_DIR, jobs=8):
    files = sorted((results_dir / "probs").glob("chb*_fold*.npz"))
    with Pool(jobs) as pool:
        rows = [r for part in pool.imap_unordered(sweep_file, files) for r in part]
    df = pd.DataFrame(rows, columns=["patient", "model", "threshold", "n_seizures",
                                     "n_detected", "n_fa", "hours", "latency_sum"])
    per_patient = df.groupby(["patient", "model", "threshold"]).sum().reset_index()
    pooled = per_patient.groupby(["model", "threshold"])[
        ["n_seizures", "n_detected", "n_fa", "hours", "latency_sum"]].sum().reset_index()
    pooled["sens"] = pooled.n_detected / pooled.n_seizures
    pooled["fa_h"] = pooled.n_fa / pooled.hours
    pooled["mean_latency"] = pooled.latency_sum / pooled.n_detected.where(pooled.n_detected > 0)
    pooled.to_csv(results_dir / "operating_curves.csv", index=False)

    models = [m for m in MODEL_ORDER if m in set(pooled.model)]
    points = {(m, s): operating_point(pooled[pooled.model == m], s) for m in models for s in TARGETS}

    # Table: FA/h (and latency) at each target sensitivity
    head = "| Model | " + " | ".join(f"FA/h at ≥{int(100 * s)} % | latency (s)" for s in TARGETS) + " |"
    lines = [head, "|---|" + "---|---|" * len(TARGETS)]
    for m in models:
        cells = []
        for s in TARGETS:
            pt = points[(m, s)]
            cells += (["not reached", "—"] if pt is None else
                      [f"{pt.fa_h:.2f}", f"{pt.mean_latency:.1f}"])
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    # Paired per-patient FA/h at each model's pooled operating point
    def patient_fa(m, s):
        pt = points[(m, s)]
        if pt is None:
            return None
        g = per_patient[(per_patient.model == m) & (per_patient.threshold == pt.threshold)]
        return (g.set_index("patient").n_fa / g.set_index("patient").hours)

    def paired(a, b, s):
        fa, fb = patient_fa(a, s), patient_fa(b, s)
        if fa is None or fb is None:
            return ["—"] * 3
        pair = pd.concat([fa.rename("a"), fb.rename("b")], axis=1).dropna()
        d = pair.a - pair.b
        p = wilcoxon(pair.a, pair.b).pvalue if (d != 0).any() else np.nan
        return [f"{d.mean():+.2f}", f"{(d < 0).sum()} / {len(d)}",
                "—" if np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")]

    tcn = [m for m in models if m != "none" and not m.endswith(" avg")]
    comp = []
    for ref_name, ref in (("moving average, same context", lambda m: f"{m} avg"),
                          ("no context", lambda m: "none")):
        comp += [f"**TCN vs. {ref_name}** (negative = TCN has fewer false alarms)", "",
                 "| TCN | " + " | ".join(f"ΔFA/h at ≥{int(100 * s)} % | fewer FA | p" for s in TARGETS)
                 + " |", "|---|" + "---|---|---|" * len(TARGETS)]
        for m in tcn:
            if ref(m) not in models:
                continue
            comp.append(f"| {m} | " + " | ".join(c for s in TARGETS for c in paired(m, ref(m), s)) + " |")
        comp.append("")

    n_patients, total_h = per_patient.patient.nunique(), \
        per_patient[(per_patient.model == models[0]) & (per_patient.threshold == THRESHOLDS[0])].hours.sum()
    n_sz = int(pooled[(pooled.model == models[0]) & (pooled.threshold == THRESHOLDS[0])].n_seizures.iloc[0])
    parts = [
        "# Experiment B3: Models compared at the same sensitivity\n",
        f"{n_patients} patients, {total_h:.0f} test hours, {n_sz} seizures. Each model's "
        "probabilities are thresholded with **one global threshold** (swept 0.01–0.99) over all "
        "patients and folds. For each target, the highest threshold that still detects at least "
        "that share of seizures (pooled) is used, i.e. the fewest false alarms at that "
        "sensitivity. Events as in B3: gaps < 10 s merged, a seizure is detected if an event "
        "overlaps it, latency = first detected second − onset (mean over detected seizures).\n",
        "The threshold is read from the test curve, so this is a threshold-free comparison of "
        "models (like comparing ROC curves), not a deployable operating point. Curves: "
        "`operating_curves.csv`. Source run: `experiment_b3_long_context.py --save-probs` "
        f"(`{results_dir.name}/`).\n",
        "## False alarms per hour at a fixed seizure sensitivity\n",
        "\n".join(lines) + "\n",
        "## Paired per patient\n",
        "Per-patient FA/h, each model at its own pooled operating point: mean difference, patients "
        "where the TCN has fewer false alarms, two-sided Wilcoxon signed-rank p-value. No "
        "correction for multiple comparisons.\n",
        "\n".join(comp),
    ]
    (results_dir / "operating_points.md").write_text("\n".join(parts))
    print((results_dir / "operating_points.md").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=RESULTS_DIR.name,
                        help="results subfolder (inside experiments/results/) with probs/")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    main(RESULTS_ROOT / args.dir, args.jobs)
