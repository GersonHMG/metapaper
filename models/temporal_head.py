"""
TemporalHead: a TCN over a sequence of per-second embeddings (from a frozen
spatial module), with one logit per second (Experiment B3).

    embeddings (B, D, L) -> TemporalConvNet (n layers) -> Linear -> ReLU -> Linear
    -> logits (B, L)

Each TemporalBlock has two dilated convolutions (kernel 3, dilation 2^i), so the
receptive field of n layers is 1 + 4 * (2^n - 1) seconds: 13, 29, 61, 125 s for
n = 2, 3, 4, 5. Causal: all of it is past context; non-causal: half on each side.
Being convolutional, the model runs on sequences of any length (a whole recording).
"""

import torch.nn as nn

from models.asymsetnet_grouped import TemporalConvNet


def receptive_field(n_layers, kernel_size=3):
    return 1 + 2 * (kernel_size - 1) * (2 ** n_layers - 1)


class TemporalHead(nn.Module):
    def __init__(self, in_dim=32, n_layers=3, width=32, causal=False, dropout=0.2, hidden=64):
        super().__init__()
        self.tcn = TemporalConvNet(num_inputs=in_dim, num_channels=[width] * n_layers,
                                   kernel_size=3, dropout=dropout, causal=causal)
        self.classifier = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(inplace=True),
                                        nn.Linear(hidden, 1))

    def forward(self, x):
        """x: (B, D, L) embeddings -> logits (B, L)"""
        return self.classifier(self.tcn(x).transpose(1, 2)).squeeze(-1)


if __name__ == "__main__":
    import torch
    for n in (2, 3, 4, 5):
        for causal in (True, False):
            m = TemporalHead(n_layers=n, causal=causal).eval()
            x = torch.randn(1, 32, 400)
            out = m(x)
            assert out.shape == (1, 400)
            # empirical receptive field: outputs at t=200 that change when input t' changes
            x2 = x.clone(); x2[:, :, 200] += 10
            changed = (m(x2) - out).abs()[0] > 0
            idx = changed.nonzero().flatten()
            span = (idx.min().item() - 200, idx.max().item() - 200)
            print(f"n={n} causal={causal}: RF={receptive_field(n)} s, "
                  f"input t=200 affects outputs t+{span[0]}..t+{span[1]}")
