import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def run_fold(model, X_train, y_train, X_test, y_test,
             epochs=80, patience=20, batch_size=64, lr=1e-3,
             weight_decay=1e-4, threshold=0.5, device="cpu"):
    """Train `model` on one fold and evaluate on the held-out fold.

    Pass a FRESH (untrained) model for every fold, e.g.
        run_fold(AsymSETNetGrouped(n_segments=3), X_tr, y_tr, X_te, y_te)

    Returns: acc, sen, spec, f1
    """
    train_ds = TensorDataset(torch.from_numpy(X_train).float(),
                             torch.from_numpy(y_train).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(),
                            torch.from_numpy(y_test).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)

    # Materialize lazy layers (e.g. LazyLinear in the classifier head)
    # before the optimizer collects the parameters.
    model.eval()
    with torch.no_grad():
        model(train_ds.tensors[0][:1].to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_loss, best_state, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validation loss (held-out fold, used for early stopping)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(loss_fn(model(xb), yb).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_loss:
            # deepcopy: state_dict() returns references that keep changing
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            probs = torch.sigmoid(model(xb.to(device))).cpu().numpy()
            all_preds.append((probs >= threshold).astype(int))
            all_true.append(yb.numpy().astype(int))
    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_true)

    tp = int(np.sum((preds == 1) & (trues == 1)))
    tn = int(np.sum((preds == 0) & (trues == 0)))
    fp = int(np.sum((preds == 1) & (trues == 0)))
    fn = int(np.sum((preds == 0) & (trues == 1)))

    acc = (tp + tn) / len(trues)
    sen = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "acc": acc, "sen": sen, "spec": spec, "f1": f1}