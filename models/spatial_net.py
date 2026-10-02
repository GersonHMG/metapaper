"""
SpatialNet: AsymSETNet without the temporal module and without segmentation.

The whole window goes through one spatial module, then global average pooling
and a linear head. Any window length works (no n_segments to divide T).

    Raw EEG (B, C, T) -> Spatial Module -> GAP -> FC -> ReLU -> FC -> logit (B,)

spatial="height":  SpatialModule from asymsetnet.py, first kernel (kernel_height, 4).
spatial="grouped": GroupedSpatialModule from asymsetnet_grouped.py, one block
                   per bipolar electrode chain.
"""

import torch
import torch.nn as nn

from models.asymsetnet import SpatialModule
from models.asymsetnet_grouped import GroupedSpatialModule


class SpatialNet(nn.Module):
    def __init__(self, spatial="height", n_channels=21, kernel_height=None,
                 spatial_f1=16, spatial_f2=32, se_reduction=16, hidden=64):
        super().__init__()
        if spatial == "height":
            self.spatial_module = SpatialModule(
                n_channels=n_channels, f1=spatial_f1, f2=spatial_f2,
                se_reduction=se_reduction, kernel_height=kernel_height,
            )
        elif spatial == "grouped":
            assert kernel_height is None, "grouped uses one kernel height per group"
            self.spatial_module = GroupedSpatialModule(
                f1=spatial_f1, f2=spatial_f2, se_reduction=se_reduction,
            )
        else:
            raise ValueError(f"spatial must be 'height' or 'grouped', got {spatial!r}")
        self.classifier = nn.Sequential(
            nn.Linear(spatial_f2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        """x: (B, C, T) -> logits (B,)"""
        feats = self.spatial_module(x.unsqueeze(1))    # (B, f2)
        return self.classifier(feats).squeeze(-1)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


if __name__ == "__main__":
    for window_sec in (1, 3, 5, 10):
        x = torch.randn(4, 21, window_sec * 256)
        for kwargs in ({"spatial": "grouped"},
                       *({"spatial": "height", "kernel_height": h} for h in (1, 4, 8, 16, 21))):
            out = SpatialNet(**kwargs)(x)
            assert out.shape == (4,), out.shape
        print(f"{window_sec:>2} s: ok")
