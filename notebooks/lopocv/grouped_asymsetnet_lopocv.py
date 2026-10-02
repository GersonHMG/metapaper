import numpy as np
from pathlib import Path
from lopocv.run_lopocv_fold import run_fold
from models.asymsetnet import AsymSETNet
from models.asymsetnet_grouped import AsymSETNetGrouped

def load_dataset(path="/home/gmarihuan/metapaper/datasets/data/balanced_windows/chbmit_windows_all.npz"):
    """Return the whole dataset as X (float32), y, patient."""
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"]
    patient = d["patient_id"]
    return X, y, patient


X, y, patient = load_dataset()

device = "cuda" if torch.cuda.is_available() else "cpu"
X, y, patient = load_dataset()
results = {}
for pid in np.unique(patient):
    test_mask = patient == pid
    train_mask = ~test_mask
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    acc, sen, spec = run_fold(X_train, y_train, X_test, y_test, device=device, n_segments=3)
    results[pid] = (acc, sen, spec)
    print(f"Patient {pid}: Acc={acc:.4f}  Sen={sen:.4f}  Spec={spec:.4f}")
accs = [v[0] for v in results.values()]
sens = [v[1] for v in results.values()]
specs = [v[2] for v in results.values()]

print("\n=== LOPOCV summary ===")
print(f"Mean Acc:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
print(f"Mean Sen:  {np.mean(sens):.4f} +/- {np.std(sens):.4f}")
print(f"Mean Spec: {np.mean(specs):.4f} +/- {np.std(specs):.4f}")