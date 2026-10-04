"""
DCSENet baseline (Aboyeji et al., Computers in Biology and Medicine 185, 2025;
dcsenet_pape.pdf), ported from notebooks/dcsenet.ipynb.

    from models.dcsenet import DCSENet, SpectrogramImage

    spec = SpectrogramImage()                    # raw (N, T) single-channel EEG -> (N, 3, 224, 224)
    model = DCSENet(n_outputs=5)                 # (N, 3, 224, 224) -> (N, 5) logits
    logits = model(spec(x))

Changes from the notebook:
  - `n_outputs` sets the size of the last FC layer (1 = the paper's single
    window logit; W = one logit per 1 s segment for the transition data).
  - forward returns logits (no final sigmoid), like the other models here.
  - SpectrogramImage is Algorithm 1 + the notebook's spectrogram_to_image_tensor,
    vectorised in torch so images are built on the GPU per batch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SFREQ = 256


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block (Hu et al.), Eq. (6)-(8) in the paper."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        # No bias -> matches the 128-parameter count reported in Table 2
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        z = x.mean(dim=(2, 3))                       # squeeze (Eq. 6)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(z))))   # excitation (Eq. 7)
        return x * s.view(b, c, 1, 1)                # scale (Eq. 8)


class DCSENet(nn.Module):
    """Dilated Convolutional Squeeze-and-Excitation Network.
    Input (B, 3, 224, 224) spectrogram images -> (B, n_outputs) logits."""

    def __init__(self, in_channels: int = 3, dilation: int = 2, se_reduction: int = 16,
                 dropout_p: float = 0.5, n_outputs: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, dilation=dilation)  # 224 -> 220
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                   # -> 110
        self.se1 = SqueezeExcitation(32, reduction=se_reduction)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, dilation=dilation)           # -> 106
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                   # -> 53
        self.se2 = SqueezeExcitation(32, reduction=se_reduction)
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(53 * 53 * 32, 64)       # 89,888 flat (Table 2)
        self.fc2 = nn.Linear(64, n_outputs)

    def forward(self, x):
        x = self.se1(self.pool1(F.relu(self.conv1(x))))
        x = self.se2(self.pool2(F.relu(self.conv2(x))))
        x = torch.flatten(self.dropout(x), 1)
        return self.fc2(F.relu(self.fc1(x)))


def log_interp_matrix(tws=SFREQ, fs=SFREQ, f_min=0.5, f_max=40.0, n_freq_bins=224):
    """(n_freq_bins, tws//2+1) matrix M with M @ P = linear interpolation of the
    power spectrum P (rfft bins) at log-spaced frequencies (scipy interp1d,
    fill with the edge values outside the FFT range)."""
    fft_freqs = np.fft.rfftfreq(tws, d=1.0 / fs)
    freqs = np.clip(np.geomspace(f_min, f_max, n_freq_bins), fft_freqs[0], fft_freqs[-1])
    i = np.clip(np.searchsorted(fft_freqs, freqs, side="right") - 1, 0, len(fft_freqs) - 2)
    w = (freqs - fft_freqs[i]) / (fft_freqs[i + 1] - fft_freqs[i])
    M = np.zeros((n_freq_bins, len(fft_freqs)))
    M[np.arange(n_freq_bins), i] = 1 - w
    M[np.arange(n_freq_bins), i + 1] = w
    return M


def get_taper(window_func, N, sigma=0.1):
    """Taper of length N: Hann (Eq. 1), Gaussian (Eq. 2) or none, as in the notebook."""
    n = np.arange(N)
    if window_func == "hann":
        return 0.5 * (1 - np.cos(2 * np.pi * n / N))
    if window_func == "gaussian":
        return np.exp(-0.5 * ((n - N / 2) / (sigma * N / 2)) ** 2)
    if window_func == "none":
        return np.ones(N)
    raise ValueError(f"Unknown window_func: {window_func}")


def bandpass_fir(l_freq=0.5, h_freq=30.0, fs=SFREQ):
    """Linear-phase FIR band-pass designed with a Hamming window (passband ripple
    0.0194, stopband attenuation 53 dB, the values in the paper). Transition
    bands and length follow MNE's defaults: l_trans = min(max(0.25 l, 2), l)
    = 0.5 Hz, h_trans = min(max(0.25 h, 2), fs/2 - h) = 7.5 Hz, length =
    3.3 / min(trans) s, odd. Cutoffs sit in the middle of the transition bands."""
    from scipy.signal import firwin
    l_trans = min(max(0.25 * l_freq, 2.0), l_freq)
    h_trans = min(max(0.25 * h_freq, 2.0), fs / 2 - h_freq)
    numtaps = int(np.ceil(3.3 / min(l_trans, h_trans) * fs)) | 1
    return firwin(numtaps, [l_freq - l_trans / 2, h_freq + h_trans / 2], pass_zero=False,
                  window="hamming", fs=fs)


def fir_filter(x, h):
    """Zero-phase (centred) FIR filtering of (N, T) signals with reflect padding.
    The window must be longer than half the filter."""
    pad = (len(h) - 1) // 2
    xp = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect")
    return F.conv1d(xp, h.flip(0).view(1, 1, -1)).squeeze(1)


class SpectrogramImage(nn.Module):
    """
    Algorithm 1 (enhanced STFT spectrogram, Hann taper) for single-channel
    signals, then the notebook's image conversion: dB relative to the mean
    power, min-max to uint8 levels, bilinear resize to image_size, 3 copies.

    (N, T) raw samples -> (N, 3, image_size, image_size) in [0, 1].
    STFT frames lie fully inside the window (the notebook's T_idx rule) with
    hop `step_s`, so a W s window gives (W - WL) / step + 1 frames.
    """

    def __init__(self, fs=SFREQ, window_length_s=1.0, step_s=0.125, f_min=0.5, f_max=40.0,
                 n_freq_bins=224, image_size=224, taper="hann", sigma=0.1, bandpass=None):
        super().__init__()
        self.tws = int(round(window_length_s * fs))
        self.step = int(round(step_s * fs))
        self.image_size = image_size
        self.register_buffer("taper", torch.from_numpy(get_taper(taper, self.tws, sigma)).float(),
                             persistent=False)
        # optional zero-phase FIR band-pass (paper section 2.1), applied to each signal first
        self.register_buffer("fir", None if bandpass is None else
                             torch.from_numpy(bandpass_fir(*bandpass, fs=fs)).float(),
                             persistent=False)
        self.register_buffer("interp", torch.from_numpy(
            log_interp_matrix(self.tws, fs, f_min, f_max, n_freq_bins)).float(), persistent=False)

    @torch.no_grad()
    def forward(self, x):
        return to_rgb(self.levels(x))

    @torch.no_grad()
    def levels(self, x):
        """(N, T) -> (N, image_size, image_size) uint8 grey levels (what PIL returns);
        cache these and feed to_rgb(levels) to the network."""
        if self.fir is not None:
            x = fir_filter(x, self.fir)
        frames = x.unfold(-1, self.tws, self.step) * self.taper            # (N, n_frames, TWS)
        power = torch.fft.rfft(frames, n=self.tws).abs() ** 2               # (N, n_frames, TWS/2+1)
        stft = (power @ self.interp.T).transpose(1, 2)                      # (N, F, n_frames)
        db = 10 * torch.log10(stft / (stft.mean(dim=(1, 2), keepdim=True) + 1e-12) + 1e-12)
        db = db - db.amin(dim=(1, 2), keepdim=True)
        img = torch.floor(db / (db.amax(dim=(1, 2), keepdim=True) + 1e-12) * 255)   # uint8 levels
        img = F.interpolate(img.unsqueeze(1), size=(self.image_size, self.image_size),
                            mode="bilinear", align_corners=False)
        return img.squeeze(1).round().clamp(0, 255).to(torch.uint8)          # PIL returns uint8


def to_rgb(levels):
    """uint8 grey levels (N, H, W) -> (N, 3, H, W) float in [0, 1], 3 identical channels."""
    return (levels.float() / 255.0).unsqueeze(1).expand(-1, 3, -1, -1)


if __name__ == "__main__":
    from PIL import Image
    from scipy.interpolate import interp1d

    n_params = sum(p.numel() for p in DCSENet().parameters())
    print(f"DCSENet parameters: {n_params:,}")
    assert n_params == 5_763_361, n_params
    assert DCSENet(n_outputs=5)(torch.randn(2, 3, 224, 224)).shape == (2, 5)

    # Reference: notebook cell 3 (channel_to_spectrogram + spectrogram_to_image_tensor)
    def reference(sig, fs=256, wl=1.0, step_s=0.125, taper="hann", sigma=0.1):
        n, tws, step = len(sig), int(wl * fs), int(step_s * fs)
        t_idx = np.arange(tws // 2, max(n - tws // 2, tws // 2) + 1, step)
        freqs_log = np.geomspace(0.5, 40.0, 224)
        nn_ = np.arange(tws)
        wf = (0.5 * (1 - np.cos(2 * np.pi * nn_ / tws)) if taper == "hann"
              else np.exp(-0.5 * ((nn_ - tws / 2) / (sigma * tws / 2)) ** 2))
        fft_freqs = np.fft.rfftfreq(tws, d=1.0 / fs)
        stft = np.zeros((224, len(t_idx)))
        for j, t in enumerate(t_idx):
            seg = sig[max(t - tws // 2, 0):min(t + tws // 2, n)]
            P = np.abs(np.fft.rfft(seg * wf, n=tws)) ** 2
            stft[:, j] = interp1d(fft_freqs, P, kind="linear", bounds_error=False,
                                  fill_value=(P[0], P[-1]))(freqs_log)
        spec = 10 * np.log10(stft / (np.mean(stft) + 1e-12) + 1e-12)
        spec -= spec.min()
        spec /= spec.max() + 1e-12
        img = Image.fromarray((spec * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    rng = np.random.default_rng(0)
    spec = SpectrogramImage()
    for window_sec in (3, 5, 8, 10):
        x = (rng.standard_normal((4, window_sec * SFREQ)) * 30).astype(np.float32)
        out = spec(torch.from_numpy(x)).numpy()
        assert out.shape == (4, 3, 224, 224)
        ref = np.stack([reference(s.astype(np.float64)) for s in x])
        diff = np.abs(out[:, 0] - ref)
        print(f"{window_sec:>2} s: frames={(window_sec * SFREQ - 256) // 32 + 1}, "
              f"max |torch - notebook| = {diff.max() * 255:.1f} levels, "
              f"mean = {diff.mean() * 255:.3f} levels")
        assert diff.mean() * 255 < 1.0

    # E1 setting: Gaussian taper sigma 0.1 with WL = 5 s on a 5 s window (one frame)
    x = (rng.standard_normal((4, 5 * SFREQ)) * 30).astype(np.float32)
    out = SpectrogramImage(window_length_s=5.0, taper="gaussian", sigma=0.1)(torch.from_numpy(x)).numpy()
    ref = np.stack([reference(s.astype(np.float64), wl=5.0, taper="gaussian") for s in x])
    print(f"gaussian WL 5 s on 5 s: mean |torch - notebook| = "
          f"{np.abs(out[:, 0] - ref).mean() * 255:.3f} levels")
    assert np.abs(out[:, 0] - ref).mean() * 255 < 1.0

    # Band-pass: response and zero-phase filtering vs. scipy on a long signal
    from scipy.signal import freqz, convolve
    h = bandpass_fir(0.5, 30.0)
    w, H = freqz(h, worN=8192, fs=SFREQ)
    gain = lambda f: 20 * np.log10(np.abs(H[np.argmin(np.abs(w - f))]) + 1e-12)
    print(f"band-pass: {len(h)} taps; gain 0.1 Hz {gain(0.1):.1f} dB, 2 Hz {gain(2):.2f} dB, "
          f"20 Hz {gain(20):.2f} dB, 40 Hz {gain(40):.1f} dB")
    assert abs(gain(2)) < 0.2 and abs(gain(20)) < 0.2 and gain(40) < -40
    sig = rng.standard_normal((2, 20 * SFREQ))
    ours = fir_filter(torch.from_numpy(sig), torch.from_numpy(h)).numpy()
    mid = slice(len(h), sig.shape[1] - len(h))                  # away from the padded edges
    ref = np.stack([convolve(s, h, mode="same") for s in sig])
    assert np.allclose(ours[:, mid], ref[:, mid], atol=1e-6)
    print("OK")
