"""
AsymSETNet: Asymmetric Squeeze-Excitation Temporal Network for EEG Seizure Detection
Implementation based on: "Enhancing Seizure Detection via Asymmetric Spatiotemporal
Feature Extraction" (Marihuan & Saavedra).

Pipeline (see Fig. 1 of the paper):
    Raw EEG -> N segments -> Spatial Module (Asymmetric Conv + SE) x2
             -> Global Average Pooling -> Temporal Module (stacked TCN)
             -> Classification Head (FC) -> Seizure / Normal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Squeeze-and-Excitation block (2D, operates on conv feature maps)
# --------------------------------------------------------------------------- #
class SqueezeExcitation2d(nn.Module):
    """Channel-wise recalibration: squeeze (global avg pool) -> excitation
    (bottleneck MLP) -> scale (channel-wise multiplication). See Hu et al.,
    2019, as used between the convolutional layers of the Spatial Module."""

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
# 2. Asymmetric convolutional block: Conv -> AvgPool -> SE -> Add & LayerNorm
# --------------------------------------------------------------------------- #
class AsymmetricConvBlock(nn.Module):
    """One stage of the Spatial Module (Section III-B).

    Applies an asymmetric (M x N) dilated convolution -> average pooling
    along the temporal axis -> Squeeze-and-Excitation recalibration -> a
    residual (skip) connection back to the block's input -> LayerNorm.

    A kernel height M equal to the current electrode/height dimension
    collapses that spatial axis to 1 in a single step (as in Spatial
    Module 1, kernel 22x4); a kernel height of 1 (Spatial Module 2,
    kernel 1x4) then only refines the temporal axis.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 pool_size=(1, 2), se_reduction=16):
        super().__init__()
        kh, kw = kernel_size
        dh, dw = dilation

        # 'same'-style padding along the time axis; no padding along height
        # so that kh == current_height forces the electrode-dimension collapse
        # described in the paper (Eq. 1).
        pad_h = 0
        pad_w = ((kw - 1) * dw) // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(kh, kw),
            dilation=(dh, dw), padding=(pad_h, pad_w),
        )
        self.pool = nn.AvgPool2d(kernel_size=pool_size, ceil_mode=True)
        self.se = SqueezeExcitation2d(out_channels, reduction=se_reduction)

        # Projection for the skip connection: mirrors the main branch's
        # height-collapsing behaviour (kernel/stride = kh along height) so the
        # residual add lines up in channels, height, and (after pooling) width.
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

        # Defensive crop in case of pooling rounding mismatches.
        if y.shape[-1] != skip.shape[-1]:
            w = min(y.shape[-1], skip.shape[-1])
            y, skip = y[..., :w], skip[..., :w]
        if y.shape[-2] != skip.shape[-2]:
            h = min(y.shape[-2], skip.shape[-2])
            y, skip = y[..., :h, :], skip[..., :h, :]

        out = y + skip
        out = out.permute(0, 2, 3, 1)   # channels last for LayerNorm
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        return F.relu(out)


# --------------------------------------------------------------------------- #
# 3. Spatial Module: two stacked Asymmetric Conv (+SE) blocks
# --------------------------------------------------------------------------- #
class SpatialModule(nn.Module):
    """Extracts cross-channel (electrode) spatial features from one EEG
    segment. Kernel sizes follow Table I: (Cmax, 4) then (1, 4), both with
    dilation (1, 2)."""

    def __init__(self, n_channels, f1=16, f2=32, se_reduction=16):
        super().__init__()
        self.block1 = AsymmetricConvBlock(
            in_channels=1, out_channels=f1,
            kernel_size=(n_channels, 4), dilation=(1, 2),
            se_reduction=se_reduction,
        )
        self.block2 = AsymmetricConvBlock(
            in_channels=f1, out_channels=f2,
            kernel_size=(1, 4), dilation=(1, 2),
            se_reduction=se_reduction,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: (B, 1, C, T_segment)
        x = self.block1(x)          # -> (B, f1, 1, T')
        x = self.block2(x)          # -> (B, f2, 1, T'')
        return self.gap(x).flatten(1)   # -> (B, f2)


# --------------------------------------------------------------------------- #
# 4. Temporal Convolutional Network (Bai et al., 2018 style)
# --------------------------------------------------------------------------- #
class Chomp1d(nn.Module):
    """Trims the extra right-side padding used to keep dilated causal
    convolutions the same length as their input."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A single causal, dilated TCN residual block, with skip connection
    (Section III-C)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
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
    """Stack of TemporalBlocks with exponentially increasing dilation,
    expanding the receptive field across the sequence of N segments."""

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size,
                                         dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, N_segments)
        return self.network(x)


# --------------------------------------------------------------------------- #
# 5. Full AsymSETNet model
# --------------------------------------------------------------------------- #
class AsymSETNet(nn.Module):
    """End-to-end Asymmetric Squeeze-Excitation Temporal Network.

    Args:
        n_channels:   number of EEG electrode channels (Cmax), e.g. 22.
        n_segments:   number of temporal segments per window (N), e.g. 5.
        spatial_f1:   feature width of Spatial Module 1.
        spatial_f2:   feature width of Spatial Module 2.
        tcn_channels: output channel sizes for each of the (2) TCN layers.
        se_reduction: SE bottleneck reduction ratio (16 in the paper).
        dropout:      dropout used inside the TCN blocks.
    """

    def __init__(self, n_channels=22, n_segments=5, spatial_f1=16,
                 spatial_f2=32, tcn_channels=(32, 32), se_reduction=16,
                 dropout=0.2):
        super().__init__()
        self.n_segments = n_segments

        self.spatial_module = SpatialModule(
            n_channels=n_channels, f1=spatial_f1, f2=spatial_f2,
            se_reduction=se_reduction,
        )
        self.temporal_module = TemporalConvNet(
            num_inputs=spatial_f2, num_channels=list(tcn_channels),
            kernel_size=3, dropout=dropout,
        )
        self.classifier = nn.Linear(tcn_channels[-1], 1)

    def forward(self, x):
        """
        x: raw EEG window, shape (B, C, T_total)
        returns: raw logits for the seizure class, shape (B,)
        """
        b, c, t_total = x.shape
        n = self.n_segments
        assert t_total % n == 0, "T_total must be divisible by n_segments"
        t_seg = t_total // n

        # Segment the recording: (B, C, T_total) -> (B*N, 1, C, T_seg)
        x = x.view(b, c, n, t_seg).permute(0, 2, 1, 3).contiguous()
        x = x.view(b * n, 1, c, t_seg)

        # Spatial feature extraction per segment -> (B*N, f2)
        feats = self.spatial_module(x)
        feats = feats.view(b, n, -1).permute(0, 2, 1)   # (B, f2, N)

        # Temporal modeling across the N segments
        temporal_out = self.temporal_module(feats)      # (B, C_tcn, N)
        pooled = temporal_out[:, :, -1]                  # last segment embedding

        return self.classifier(pooled).squeeze(-1)       # (B,)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


# --------------------------------------------------------------------------- #
# 6. Quick smoke test / training-loop sketch
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, T_TOTAL = 8, 22, 5 * 256      # 5-second windows @ 256 Hz -> 1280 samples
    N_SEGMENTS = 5

    model = AsymSETNet(n_channels=C, n_segments=N_SEGMENTS)
    x = torch.randn(B, C, T_TOTAL)
    y = torch.randint(0, 2, (B,)).float()

    logits = model(x)
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, y)
    loss.backward()  # verify gradients flow end-to-end

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input shape:            {tuple(x.shape)}")
    print(f"Output (logits) shape:  {tuple(logits.shape)}")
    print(f"BCE loss:               {loss.item():.4f}")
    print(f"Trainable parameters:   {n_params:,}")

    # Example optimizer matching the paper's setup (Section IV-C)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.step()
    print("One optimizer step completed successfully.")