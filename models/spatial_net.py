"""
SpatialNet: AsymSETNet without the temporal module and without segmentation.

The whole window goes through one spatial module, then global average pooling
and a linear head. Any window length works (no n_segments to divide T).

    Raw EEG (B, C, T) -> Spatial Module -> GAP -> FC -> ReLU -> FC -> logit (B,)

spatial="height":  SpatialModule from asymsetnet.py, first kernel (kernel_height, 4).
spatial="grouped": GroupedSpatialModule from asymsetnet_grouped.py, one block
                   per electrode chain (`groups`, default the bipolar
                   ELECTRODE_GROUPS; REFERENTIAL_GROUPS for 19-electrode data).
spatial="factorized": FactorizedSpatialModule: the h=1 temporal blocks, then a
                   learned pooling over electrodes (`pool="mix"`, "attn",
                   "grouped", or "height" with `kernel_height`) that starts as
                   the channel mean, i.e. as the h=1 model.

channel_dropout > 0 zeroes whole electrodes at random during training.
"""

import torch
import torch.nn as nn

from models.asymsetnet import SpatialModule
from models.asymsetnet_grouped import (GroupedSpatialModule, ELECTRODE_GROUPS, FLIP_CHANNELS,
                                       REFERENTIAL_GROUPS)


class FactorizedSpatialModule(nn.Module):
    """Temporal features per electrode first (SpatialModule with kernel_height=1),
    then spatial pooling over the electrode axis on those features:

    pool="mix":  depthwise (C, 1) conv, one electrode weighting per feature map.
    pool="attn": softmax attention over electrodes, scored from each electrode's
                 time-averaged features (finds the few channels a focal seizure hits).
    pool="grouped": depthwise (|g|, 1) conv per electrode group (montage chain),
                 then a depthwise (G, 1) fusion across the groups. No polarity
                 flip: the inputs here are nonlinear features, not voltages.
    pool="height": rows reordered to chain order (the concatenated groups), a
                 depthwise (kernel_height, 1) conv sliding over them, then a
                 depthwise fusion over the C - kernel_height + 1 positions.
                 kernel_height=None means C (one kernel over all electrodes).

    All start as the plain electrode mean (height: close to it, the edge rows
    get less weight when kernel_height < C), so at init the module equals the
    h=1 model and the spatial part can only add to it."""

    def __init__(self, n_channels=21, f1=16, f2=32, se_reduction=16, pool="mix",
                 kernel_height=None, groups=ELECTRODE_GROUPS):
        super().__init__()
        self.temporal = SpatialModule(n_channels=n_channels, f1=f1, f2=f2,
                                      se_reduction=se_reduction, kernel_height=1)
        self.pool = pool
        if pool == "mix":
            self.mix = nn.Conv2d(f2, f2, kernel_size=(n_channels, 1), groups=f2, bias=False)
            nn.init.constant_(self.mix.weight, 1.0 / n_channels)
        elif pool == "attn":
            self.score = nn.Linear(f2, 1)
            nn.init.zeros_(self.score.weight)
            nn.init.zeros_(self.score.bias)
        elif pool == "grouped":
            self.group_names = list(groups)
            for name in self.group_names:
                self.register_buffer(f"idx_{name}", torch.tensor(groups[name], dtype=torch.long),
                                     persistent=False)
            self.group_convs = nn.ModuleDict({
                name: nn.Conv2d(f2, f2, kernel_size=(len(groups[name]), 1), groups=f2, bias=False)
                for name in self.group_names})
            for name, conv in self.group_convs.items():
                nn.init.constant_(conv.weight, 1.0 / len(groups[name]))
            self.fusion = nn.Conv2d(f2, f2, kernel_size=(len(groups), 1), groups=f2, bias=False)
            n_total = sum(len(g) for g in groups.values())
            with torch.no_grad():                 # group mean * |g| / total = electrode mean
                self.fusion.weight.copy_(torch.tensor(
                    [len(groups[n]) / n_total for n in self.group_names]).view(1, 1, -1, 1)
                    .expand_as(self.fusion.weight))
        elif pool == "height":
            kh = n_channels if kernel_height is None else kernel_height
            assert 1 <= kh <= n_channels, "kernel_height must be in [1, n_channels]"
            order = [c for g in groups.values() for c in g]
            if sorted(order) != list(range(n_channels)):    # groups overlap or miss channels
                order = list(range(n_channels))
            self.register_buffer("order", torch.tensor(order, dtype=torch.long), persistent=False)
            self.local = nn.Conv2d(f2, f2, kernel_size=(kh, 1), groups=f2, bias=False)
            nn.init.constant_(self.local.weight, 1.0 / kh)
            n_pos = n_channels - kh + 1
            self.fusion = nn.Conv2d(f2, f2, kernel_size=(n_pos, 1), groups=f2, bias=False)
            nn.init.constant_(self.fusion.weight, 1.0 / n_pos)
        else:
            raise ValueError(f"pool must be 'mix', 'attn', 'grouped' or 'height', got {pool!r}")

    def forward(self, x):
        # x: (B, 1, C, T)
        x = self.temporal.block2(self.temporal.block1(x))    # (B, f2, C, T'')
        if self.pool == "mix":
            return self.mix(x).mean(dim=(2, 3))               # (B, f2)
        if self.pool == "grouped":
            x = torch.cat([self.group_convs[n](x.index_select(2, getattr(self, f"idx_{n}")))
                           for n in self.group_names], dim=2)  # (B, f2, G, T'')
            return self.fusion(x).mean(dim=(2, 3))            # (B, f2)
        if self.pool == "height":
            x = self.local(x.index_select(2, self.order))     # (B, f2, C - kh + 1, T'')
            return self.fusion(x).mean(dim=(2, 3))            # (B, f2)
        feats = x.mean(dim=3).transpose(1, 2)                 # (B, C, f2)
        w = torch.softmax(self.score(feats), dim=1)           # (B, C, 1)
        return (w * feats).sum(dim=1)                         # (B, f2)


class SpatialNet(nn.Module):
    def __init__(self, spatial="height", n_channels=21, kernel_height=None,
                 spatial_f1=16, spatial_f2=32, se_reduction=16, hidden=64,
                 groups=ELECTRODE_GROUPS, flip_channels=FLIP_CHANNELS,
                 pool="mix", channel_dropout=0.0):
        super().__init__()
        self.channel_dropout = channel_dropout
        if spatial == "factorized":
            assert kernel_height is None or pool == "height", \
                "factorized takes kernel_height only with pool='height'"
            self.spatial_module = FactorizedSpatialModule(
                n_channels=n_channels, f1=spatial_f1, f2=spatial_f2,
                se_reduction=se_reduction, pool=pool,
                kernel_height=kernel_height, groups=groups,
            )
        elif spatial == "height":
            self.spatial_module = SpatialModule(
                n_channels=n_channels, f1=spatial_f1, f2=spatial_f2,
                se_reduction=se_reduction, kernel_height=kernel_height,
            )
        elif spatial == "grouped":
            assert kernel_height is None, "grouped uses one kernel height per group"
            self.spatial_module = GroupedSpatialModule(
                groups=groups, flip_channels=flip_channels,
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
        if self.training and self.channel_dropout > 0:
            keep = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= self.channel_dropout
            x = x * keep / (1 - self.channel_dropout)
        feats =self.spatial_module(x.unsqueeze(1))    # (B, f2)
        return self.classifier(feats).squeeze(-1)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


if __name__ == "__main__":
    for window_sec in (1, 3, 5, 10):
        x = torch.randn(4, 21, window_sec * 256)
        for kwargs in ({"spatial": "grouped"},
                       {"spatial": "factorized", "pool": "mix"},
                       {"spatial": "factorized", "pool": "attn", "channel_dropout": 0.2},
                       {"spatial": "factorized", "pool": "grouped"},
                       *({"spatial": "factorized", "pool": "height", "kernel_height": h}
                         for h in (4, 16, None)),
                       *({"spatial": "height", "kernel_height": h} for h in (1, 4, 8, 16, 21))):
            out = SpatialNet(**kwargs)(x)
            assert out.shape == (4,), out.shape
        x = torch.randn(4, 19, window_sec * 256)          # referential (Siena)
        for kwargs in ({"spatial": "grouped", "groups": REFERENTIAL_GROUPS, "flip_channels": []},
                       *({"spatial": "height", "kernel_height": h} for h in (1, 4, 8, 16, 19))):
            out = SpatialNet(n_channels=19, **kwargs)(x)
            assert out.shape == (4,), out.shape
        print(f"{window_sec:>2} s: ok")
