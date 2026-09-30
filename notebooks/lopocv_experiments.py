import csv
import re
from pathlib import Path

import numpy as np
import torch

from lopocv.run_lopocv_fold import run_fold
from models.asymsetnet import AsymSETNet
from models.asymsetnet_grouped import AsymSETNetGrouped


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_PATH = "/home/gmarihuan/processed/chbmit_windows_all.npz"
RESULTS_ROOT = Path(".")      # CSVs go to RESULTS_ROOT/<model>_lopocv/run_<x>.csv
N_SEGMENTS = 3
N_RUNS = 30                   # full LOPOCV repetitions, each with its own seed
BASE_SEED = 0                 # run r uses seed BASE_SEED + r

# name -> function that builds a FRESH model for each fold
MODELS = {
    "asymsetnet": lambda n_ch: AsymSETNet(n_channels=n_ch,
                                          n_segments=N_SEGMENTS),
    "asymsetnet_grouped": lambda n_ch: AsymSETNetGrouped(n_channels=n_ch,
                                                         n_segments=N_SEGMENTS),
}

METRICS = ("acc", "sen", "spec", "f1")
COUNTS = ("tp", "tn", "fp", "fn")
FIELDS = ["patient", "seed", "n_windows", "n_seizure", *COUNTS, *METRICS]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_dataset(path=DATA_PATH):
    """Return the whole dataset as X (float32), y, patient."""
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"]
    patient = d["patient_id"]
    return X, y, patient


def next_run_id(model_names, root=RESULTS_ROOT):
    """Next run number not used by any model folder, so all models
    of the same run share the same run_<x>."""
    used = [0]
    for name in model_names:
        folder = root / f"{name}_lopocv"
        if folder.exists():
            for f in folder.glob("run_*.csv"):
                m = re.fullmatch(r"run_(\d+)\.csv", f.name)
                if m:
                    used.append(int(m.group(1)))
    return max(used) + 1


def summarize(rows, key):
    vals = np.array([r[key] for r in rows], dtype=float)
    return np.nanmean(vals), np.nanstd(vals)


def metrics_from_counts(tp, tn, fp, fn):
    """acc, sen, spec, f1 from confusion-matrix counts."""
    n = tp + tn + fp + fn
    return {
        "acc": (tp + tn) / n if n > 0 else float("nan"),
        "sen": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "spec": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0
              else float("nan"),
    }


def run_lopocv(model_name, build_model, X, y, patient, out_path, seed, device):
    """One full LOPOCV for one model. Writes out_path.

    Returns (means, pooled):
        means  - per-patient metrics averaged over patients
        pooled - metrics computed from the summed TP/TN/FP/FN of all patients
    """
    n_channels = X.shape[1]
    rows = []

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()

        for pid in np.unique(patient):
            test_mask = patient == pid
            train_mask = ~test_mask

            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_model(n_channels)           # fresh model per fold

            res = run_fold(
                model,
                X[train_mask], y[train_mask],
                X[test_mask], y[test_mask],
                device=device,
            )

            row = {
                "patient": pid, "seed": seed,
                "n_windows": int(test_mask.sum()),
                "n_seizure": int(y[test_mask].sum()),
                **res,
            }
            rows.append(row)
            writer.writerow(row)
            fh.flush()                                # keep partial results

            print(f"  Patient {pid}: Acc={res['acc']:.4f}  "
                  f"Sen={res['sen']:.4f}  Spec={res['spec']:.4f}  "
                  f"F1={res['f1']:.4f}  (TP={res['tp']} FN={res['fn']} "
                  f"FP={res['fp']} TN={res['tn']})")

        # Summary rows: mean / std over patients, then pooled totals
        means = {k: summarize(rows, k)[0] for k in METRICS}
        stds = {k: summarize(rows, k)[1] for k in METRICS}
        totals = {k: sum(r[k] for r in rows) for k in COUNTS}
        pooled = metrics_from_counts(**totals)

        writer.writerow({"patient": "mean", "seed": seed, **means})
        writer.writerow({"patient": "std", "seed": seed, **stds})
        writer.writerow({
            "patient": "total", "seed": seed,
            "n_windows": sum(r["n_windows"] for r in rows),
            "n_seizure": sum(r["n_seizure"] for r in rows),
            **totals, **pooled,
        })

    print(f"  --- {model_name} summary (seed {seed}) ---")
    for k in METRICS:
        print(f"  {k.capitalize():<4}: mean {means[k]:.4f} +/- {stds[k]:.4f}"
              f"   | pooled {pooled[k]:.4f}")
    return means, pooled


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(n_runs=N_RUNS):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, patient = load_dataset()
    print(f"Device: {device} | {len(np.unique(patient))} patients | "
          f"{n_runs} runs\n")

    # model_name -> list of per-run results
    all_means = {name: [] for name in MODELS}
    all_pooled = {name: [] for name in MODELS}

    for r in range(n_runs):
        seed = BASE_SEED + r
        run_id = next_run_id(MODELS.keys())
        print(f"########## Run {r + 1}/{n_runs}  (run_{run_id}, seed {seed}) "
              f"##########")

        for model_name, build_model in MODELS.items():
            out_dir = RESULTS_ROOT / f"{model_name}_lopocv"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"run_{run_id}.csv"
            print(f"\n===== {model_name} -> {out_path} =====")

            means, pooled = run_lopocv(model_name, build_model, X, y, patient,
                                       out_path, seed, device)
            all_means[model_name].append(means)
            all_pooled[model_name].append(pooled)
        print()

    # Summary across runs: mean +/- std of the per-run values
    print("========== Summary across runs ==========")
    for model_name in MODELS:
        print(f"{model_name} ({len(all_means[model_name])} runs)")
        for k in METRICS:
            m = np.array([d[k] for d in all_means[model_name]], dtype=float)
            p = np.array([d[k] for d in all_pooled[model_name]], dtype=float)
            print(f"  {k.capitalize():<4}: patient-mean "
                  f"{np.nanmean(m):.4f} +/- {np.nanstd(m):.4f}   | pooled "
                  f"{np.nanmean(p):.4f} +/- {np.nanstd(p):.4f}")


if __name__ == "__main__":
    main()