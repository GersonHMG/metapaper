"""
AsymSETNetGroupedSegments: grouped spatial module per segment, optional TCN
over the segments, and one logit per segment (Experiment B).

    Raw EEG (B, C, T) -> N segments
        -> GroupedSpatialModule per segment     (B, f2, N)
        -> TemporalConvNet (or none)            (B, C_tcn, N)
        -> head Linear -> ReLU -> Linear, shared by every segment
        -> logits (B, N)

The head is the same as SpatialNet's. With tcn_channels=() each segment is
classified from its own spatial features only.
A window-level probability is the mean of the segment probabilities.
"""

import torch
import torch.nn as nn

from models.asymsetnet_grouped import (ELECTRODE_GROUPS, FLIP_CHANNELS,
                                       GroupedSpatialModule, TemporalConvNet)


class AsymSETNetGroupedSegments(nn.Module):
    def __init__(self, n_segments, tcn_channels=(), causal=False, groups=ELECTRODE_GROUPS,
                 flip_channels=FLIP_CHANNELS, spatial_f1=16, spatial_f2=32,
                 se_reduction=16, dropout=0.2, hidden=64):
        super().__init__()
        self.n_segments = n_segments
        self.spatial_module = GroupedSpatialModule(
            groups=groups, flip_channels=flip_channels,
            f1=spatial_f1, f2=spatial_f2, se_reduction=se_reduction,
        )
        self.temporal_module = TemporalConvNet(
            num_inputs=spatial_f2, num_channels=list(tcn_channels),
            kernel_size=3, dropout=dropout, causal=causal,
        )
        out_dim = tcn_channels[-1] if tcn_channels else spatial_f2
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.spatial_frozen = False

    def freeze_spatial(self):
        """Stop training the spatial module: no gradients, and kept in eval mode
        (BatchNorm running stats fixed) even when the model is set to train()."""
        for p in self.spatial_module.parameters():
            p.requires_grad = False
        self.spatial_frozen = True
        self.spatial_module.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.spatial_frozen:
            self.spatial_module.eval()
        return self

    @classmethod
    def from_pretrained_spatial(cls, pretrained, tcn_channels, freeze=True, **kwargs):
        """New model with `pretrained`'s spatial module weights, a fresh TCN and a
        fresh head (Experiment B2). `pretrained` is an AsymSETNetGroupedSegments."""
        model = cls(n_segments=pretrained.n_segments, tcn_channels=tcn_channels, **kwargs)
        model.spatial_module.load_state_dict(pretrained.spatial_module.state_dict())
        return model.freeze_spatial() if freeze else model

    def forward(self, x):
        """x: (B, C, T) -> per-segment logits (B, N)"""
        b, c, t_total = x.shape
        n = self.n_segments
        assert t_total % n == 0, "T must be divisible by n_segments"
        t_seg = t_total // n

        x = x.view(b, c, n, t_seg).permute(0, 2, 1, 3).reshape(b * n, 1, c, t_seg)
        feats = self.spatial_module(x).view(b, n, -1).permute(0, 2, 1)   # (B, f2, N)
        feats = self.temporal_module(feats).permute(0, 2, 1)             # (B, N, C_tcn)
        return self.classifier(feats).squeeze(-1)                        # (B, N)

    def predict_proba(self, x):
        """Window-level probability: mean of the segment probabilities, (B,)."""
        return torch.sigmoid(self.forward(x)).mean(dim=-1)


if __name__ == "__main__":
    from experiments.training import FocalLoss

    for window_sec, seg_sec in [(3, 1), (5, 1), (8, 1), (10, 1), (8, 2), (10, 2)]:
        n = window_sec // seg_sec
        x = torch.randn(4, 21, window_sec * 256)
        y = torch.randint(0, 2, (4,)).float()
        for tcn in [(), (32,), (32, 32), (32, 32, 32)]:
            model = AsymSETNetGroupedSegments(n_segments=n, tcn_channels=tcn)
            logits = model(x)
            assert logits.shape == (4, n), logits.shape
            FocalLoss()(logits, y).backward()
            assert model.predict_proba(x).shape == (4,)
        print(f"{window_sec:>2} s window, {seg_sec} s segments (N={n}): ok")

    # Frozen spatial module (B2): only the TCN and head change after a training step
    pre = AsymSETNetGroupedSegments(n_segments=3)
    pre.train()(torch.randn(16, 21, 768))                    # move BN running stats
    model = AsymSETNetGroupedSegments.from_pretrained_spatial(pre, tcn_channels=(32,))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    FocalLoss()(model(torch.randn(8, 21, 768)), torch.ones(8)).backward()
    opt.step()
    after = model.state_dict()
    changed = {k.split(".")[0] for k in before if not torch.equal(before[k], after[k])}
    assert changed == {"temporal_module", "classifier"}, changed
    assert all(torch.equal(v, pre.spatial_module.state_dict()[k])
               for k, v in model.spatial_module.state_dict().items())
    print("frozen spatial module: ok (only temporal_module and classifier changed)")
