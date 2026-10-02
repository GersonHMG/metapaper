"""
Shared training / evaluation helpers for the experiments notebooks.

    from experiments.training import (FocalLoss, TrainConfig, set_seed, train_model,
                                      predict_proba, confusion_counts, add_metrics)
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    epochs: int = 100
    patience: int = 15              # early stopping on validation loss
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    focal_alpha: float = 0.75       # weight of the positive (seizure) class
    focal_gamma: float = 2.0


class FocalLoss(nn.Module):
    """Binary focal loss on logits (Lin et al., 2017):
    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)                                   # prob. of the true class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_proba(model, X, device=DEVICE):
    model.eval()
    probs = [torch.sigmoid(model(xb.to(device))).cpu()
             for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=256)]
    return torch.cat(probs).numpy()


def train_model(model, X_tr, y_tr, X_val, y_val, cfg=TrainConfig(), device=DEVICE):
    """Train with focal loss; early-stop on validation loss and restore the best
    weights. Returns (model, best_epoch, best_val_loss)."""
    model = model.to(device)
    loss_fn = FocalLoss(cfg.focal_alpha, cfg.focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_loader = make_loader(X_tr, y_tr, cfg.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, cfg.batch_size, shuffle=False)

    best_loss, best_state, best_epoch, no_improve = float("inf"), None, 0, 0
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = np.mean([loss_fn(model(xb.to(device)), yb.to(device)).item()
                                for xb, yb in val_loader])

        if val_loss < best_loss:
            best_loss, best_epoch, no_improve = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def confusion_counts(y_true, y_pred):
    return {
        "TP": int(((y_pred == 1) & (y_true == 1)).sum()),
        "FP": int(((y_pred == 1) & (y_true == 0)).sum()),
        "TN": int(((y_pred == 0) & (y_true == 0)).sum()),
        "FN": int(((y_pred == 0) & (y_true == 1)).sum()),
    }


def add_metrics(df):
    """Add sensitivity, specificity, precision, F1 and accuracy from TP/FP/TN/FN columns."""
    tp, fp, tn, fn = (df[c].astype(float) for c in ("TP", "FP", "TN", "FN"))
    df = df.copy()
    df["sensitivity"] = tp / (tp + fn)
    df["specificity"] = tn / (tn + fp)
    df["precision"] = tp / (tp + fp)
    df["f1"] = 2 * tp / (2 * tp + fp + fn)
    df["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    return df
