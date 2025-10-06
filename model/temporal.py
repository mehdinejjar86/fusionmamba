# temporal.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPositionEncoding(nn.Module):
    def __init__(self, channels, max_freq=10):
        super().__init__()
        freq_bands = 2.0 ** torch.linspace(0, max_freq - 1, max_freq)
        self.register_buffer('freq_bands', freq_bands)
        self.time_proj = nn.Sequential(
            nn.Linear(max_freq * 2, channels), nn.SiLU(), nn.Linear(channels, channels)
        )

    def forward(self, t):  # [B,N] in [0,1]
        B, N = t.shape
        t = t.unsqueeze(-1)
        feats = []
        for f in self.freq_bands:
            feats.append(torch.sin(2 * math.pi * f * t))
            feats.append(torch.cos(2 * math.pi * f * t))
        enc = torch.cat(feats, dim=-1)
        return self.time_proj(enc)  # [B,N,C]


class TemporalWeightingMLP(nn.Module):
    """Per-anchor scores from encoded τ, softmax over N."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        ch = hidden_dim // 2
        self.enc = TemporalPositionEncoding(ch)
        self.head = nn.Sequential(
            nn.Linear(ch, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, timesteps):     # [B,N]
        enc = self.enc(timesteps)     # [B,N,C]
        logits = self.head(enc).squeeze(-1)  # [B,N]
        return F.softmax(logits, dim=1)
