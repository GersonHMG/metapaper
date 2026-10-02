"""
AsymSETNet (grouped-spatial variant)
====================================

Same pipeline as the original AsymSETNet, but Spatial Module 1 no longer uses
a single (Cmax x 4) kernel over all 21 electrodes. Instead, every bipolar
electrode group (montage chain) has its OWN asymmetric conv block whose
kernel height equals the size of that group:

    left_temporal        4 ch  -> kernel (4, 4)
    left_parasagittal    4 ch  -> kernel (4, 4)
    right_parasagittal   4 ch  -> kernel (4, 4)
    right_temporal       4 ch  -> kernel (4, 4)
    midline              2 ch  -> kernel (2, 4)
    transverse_temporal  3 ch  -> kernel (3, 4)

Each group block collapses its electrodes to height 1. The G group outputs are
stacked along the height axis -> (B*N, f1, G, T'), and Spatial Module 2 fuses
the groups with a (G x 4) kernel (cross-region / inter-hemispheric features).

Pipeline:
    Raw EEG -> N segments
            -> Group Spatial Module 1 (one Asym Conv + SE per electrode group)
            -> stack groups -> Spatial Module 2 (G x 4 fusion, + SE)
            -> Global Average Pooling -> Temporal Module (stacked TCN)
            -> Classification Head (FC) -> Seizure / Normal

Channel order assumed (index in X[:, idx, :]), from REQUIRED_CHANNELS:
     0 FP1-F7   1 F7-T7    2 P7-O1    3 FP1-F3   4 F3-C3    5 C3-P3
     6 P3-O1    7 FP2-F4   8 F4-C4    9 C4-P4   10 P4-O2   11 FP2-F8
    12 F8-T8   13 T8-P8   14 P8-O2   15 FZ-CZ   16 CZ-PZ   17 P7-T7
    18 T7-FT9  19 FT9-FT10 20 FT10-T8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 0. Electrode groups (indices into the 21-channel window)
# --------------------------------------------------------------------------- #
ELECTRODE_GROUPS = {
    "left_temporal":       [0, 1, 17, 2],    # FP1-F7, F7-T7, P7-T7, P7-O1
    "left_parasagittal":   [3, 4, 5, 6],     # FP1-F3, F3-C3, C3-P3, P3-O1
    "right_parasagittal":  [7, 8, 9, 10],    # FP2-F4, F4-C4, C4-P4, P4-O2
    "right_temporal":      [11, 12, 13, 14], # FP2-F8, F8-T8, T8-P8, P8-O2
    "midline":             [15, 16],         # FZ-CZ, CZ-PZ
    "transverse_temporal": [18, 19, 20],     # T7-FT9, FT9-FT10, FT10-T8
}

# P7-T7 is the reversed-polarity version of T7-P7 (the usual 3rd link of the
# left temporal chain). Channels listed here are multiplied by -1 so the left
# temporal chain is continuous: FP1-F7 -> F7-T7 -> T7-P7 -> P7-O1.
FLIP_CHANNELS = [17]


# --------------------------------------------------------------------------- #
# 1. Squeeze-and-Excitation block (unchanged)
# --------------------------------------------------------------------------- #
class SqueezeExcitation2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.squeeze(x).view(b, c)
        e = self.excite(s).view(b, c, 1, 1)
        return x * e


# --------------------------------------------------------------------------- #
# 2. Asymmetric convolutional block (unchanged)
# --------------------------------------------------------------------------- #
class AsymmetricConvBlock(nn.Module):
    """Conv (kh x kw, dilated) -> AvgPool(time) -> SE -> residual -> LayerNorm.
    kh == input height collapses the height axis to 1."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 pool_size=(1, 2), se_reduction=16):
        super().__init__()
        kh, kw = kernel_size
        dh, dw = dilation
        pad_w = ((kw - 1) * dw) // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(kh, kw),
            dilation=(dh, dw), padding=(0, pad_w),
        )
        self.pool = nn.AvgPool2d(kernel_size=pool_size, ceil_mode=True)
        self.se = SqueezeExcitation2d(out_channels, reduction=se_reduction)

        self.skip_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kh, 1),
                      stride=(kh, 1)),
            nn.AvgPool2d(kernel_size=pool_size, ceil_mode=True),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        skip = self.skip_proj(x)

        y = self.conv(x)
        y = self.pool(y)
        y = self.se(y)

        if y.shape[-1] != skip.shape[-1]:
            w = min(y.shape[-1], skip.shape[-1])
            y, skip = y[..., :w], skip[..., :w]
        if y.shape[-2] != skip.shape[-2]:
            h = min(y.shape[-2], skip.shape[-2])
            y, skip = y[..., :h, :], skip[..., :h, :]

        out = y + skip
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        return F.relu(out)


# --------------------------------------------------------------------------- #
# 3. Grouped Spatial Module: one kernel per electrode group, then fusion
# --------------------------------------------------------------------------- #
class GroupedSpatialModule(nn.Module):
    """
    Block 1: one AsymmetricConvBlock per electrode group, kernel (|group|, 4),
             each with its own weights. Every group -> (B, f1, 1, T').
    Stack:   concatenate groups along height -> (B, f1, G, T').
    Block 2: AsymmetricConvBlock with kernel (G, 4) fusing all groups
             -> (B, f2, 1, T'').
    GAP ->   (B, f2)
    """

    def __init__(self, groups=ELECTRODE_GROUPS, flip_channels=FLIP_CHANNELS,
                 f1=16, f2=32, se_reduction=16):
        super().__init__()
        self.group_names = list(groups.keys())
        n_groups = len(self.group_names)

        # Channel indices per group, stored as buffers (move with .to(device)).
        for name in self.group_names:
            self.register_buffer(
                f"idx_{name}", torch.tensor(groups[name], dtype=torch.long),
                persistent=False,
            )
        self.flip_channels = list(flip_channels)

        # One independent asymmetric conv block per group.
        self.group_blocks = nn.ModuleDict({
            name: AsymmetricConvBlock(
                in_channels=1, out_channels=f1,
                kernel_size=(len(groups[name]), 4), dilation=(1, 2),
                se_reduction=se_reduction,
            )
            for name in self.group_names
        })

        # Fusion across the G groups.
        self.fusion_block = AsymmetricConvBlock(
            in_channels=f1, out_channels=f2,
            kernel_size=(n_groups, 4), dilation=(1, 2),
            se_reduction=se_reduction,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: (B, 1, C, T_segment)
        if self.flip_channels:
            x = x.clone()
            x[:, :, self.flip_channels, :] = -x[:, :, self.flip_channels, :]

        group_feats = []
        for name in self.group_names:
            idx = getattr(self, f"idx_{name}")
            xg = x.index_select(2, idx)                  # (B, 1, |g|, T)
            group_feats.append(self.group_blocks[name](xg))  # (B, f1, 1, T')

        x = torch.cat(group_feats, dim=2)                # (B, f1, G, T')
        x = self.fusion_block(x)                         # (B, f2, 1, T'')
        return self.gap(x).flatten(1)                    # (B, f2)


# --------------------------------------------------------------------------- #
# 4. Temporal Convolutional Network (unchanged)
# --------------------------------------------------------------------------- #
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """causal=True (default) only looks at past segments; causal=False pads
    symmetrically so each output sees both sides."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 dropout=0.2, causal=True):
        super().__init__()
        if causal:
            padding = (kernel_size - 1) * dilation     # left context, trimmed on the right
            chomp = padding
        else:
            assert (kernel_size - 1) * dilation % 2 == 0, "non-causal needs an even receptive step"
            padding = (kernel_size - 1) * dilation // 2  # same amount on both sides
            chomp = 0
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            Chomp1d(chomp), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            Chomp1d(chomp), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2,
                 causal=True):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size,
                                        2 ** i, dropout, causal))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------------------------------- #
# 5. Full model
# --------------------------------------------------------------------------- #
class AsymSETNetGrouped(nn.Module):
    """AsymSETNet with a separate spatial kernel per electrode group.

    Args:
        n_channels:   total electrode channels in the input (21 here).
        n_segments:   temporal segments per window; must divide T_total
                      (768 samples -> 3, 4, 6 or 8; NOT 5).
        groups:       dict name -> list of channel indices.
        flip_channels: channel indices to sign-flip before grouping.
        tcn_channels: TCN layer widths. Empty tuple = no temporal module:
                      the segment features are averaged instead.
        causal:       True = the TCN only sees past segments; False = both
                      sides (symmetric padding).
    """

    def __init__(self, n_channels=21, n_segments=3, groups=ELECTRODE_GROUPS,
                 flip_channels=FLIP_CHANNELS, spatial_f1=16, spatial_f2=32,
                 tcn_channels=(32,32), se_reduction=16, dropout=0.2,
                 causal=True):
        super().__init__()
        used = sorted(i for g in groups.values() for i in g)
        assert max(used) < n_channels, "Group index exceeds n_channels"
        assert len(used) == len(set(used)), "A channel appears in two groups"

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
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, x):
        """x: (B, C, T_total) -> logits (B,)"""
        b, c, t_total = x.shape
        n = self.n_segments
        assert t_total % n == 0, "T_total must be divisible by n_segments"
        t_seg = t_total // n

        x = x.view(b, c, n, t_seg).permute(0, 2, 1, 3).contiguous()
        x = x.view(b * n, 1, c, t_seg)                 # (B*N, 1, C, T_seg)

        feats = self.spatial_module(x)                 # (B*N, f2)
        feats = feats.view(b, n, -1).permute(0, 2, 1)  # (B, f2, N)

        temporal_out = self.temporal_module(feats)     # (B, C_tcn, N)
        if len(self.temporal_module.network) == 0:
            # no TCN: the last segment has seen nothing else, so average all
            pooled = temporal_out.mean(dim=-1)
        else:
            pooled = temporal_out[:, :, -1]
        return self.classifier(pooled).squeeze(-1)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


# --------------------------------------------------------------------------- #
# 6. Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, T_TOTAL = 8, 21, 3 * 256      # matches the processed windows (21, 768)
    N_SEGMENTS = 3                      # 768 / 3 = 256 samples (1 s) per segment

    model = AsymSETNetGrouped(n_channels=C, n_segments=N_SEGMENTS)
    x = torch.randn(B, C, T_TOTAL)
    y = torch.randint(0, 2, (B,)).float()

    logits = model(x)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input shape:           {tuple(x.shape)}")
    print(f"Output (logits) shape: {tuple(logits.shape)}")
    print(f"BCE loss:              {loss.item():.4f}")
    print(f"Trainable parameters:  {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.step()
    print("One optimizer step completed successfully.")