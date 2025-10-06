# pyramid.py
# Deformable cross-attention (pyramid)

import torch.nn as nn
import torch
from model.fusion.utils import AdaptiveFrequencyDecoupling
import torch.nn.functional as F
import math

class PyramidCrossAttention(nn.Module):
    def __init__(self, channels, num_heads=4, num_points=4, num_levels=3, init_spatial_range=0.1):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        self.head_dim = channels // num_heads

        self.sampling_offsets = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.GroupNorm(min(8, channels // 4), channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, num_heads * num_levels * num_points * 2, 1)
        )
        self.attention_weights = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.GroupNorm(min(8, channels // 4), channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, num_heads * num_levels * num_points, 1)
        )
        self.value_proj = nn.Conv2d(channels, channels, 1)
        self.output_proj = nn.Conv2d(channels, channels, 1)

        self.freq = AdaptiveFrequencyDecoupling(channels)
        self.hf_scale = nn.Parameter(torch.tensor(0.3))
        self.level_embed = nn.Parameter(torch.zeros(num_levels, channels))
        self._reset_parameters(init_spatial_range)

    def _reset_parameters(self, init_range):
        nn.init.constant_(self.sampling_offsets[-1].weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]) * init_range
        grid = grid.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid[:, :, i, :] *= (i + 1) / self.num_points
        for lvl in range(self.num_levels):
            grid[:, lvl, :, :] *= (2 ** lvl) * 0.1
        with torch.no_grad():
            self.sampling_offsets[-1].bias.copy_(grid.view(-1))
        nn.init.constant_(self.attention_weights[-1].weight.data, 0.)
        nn.init.constant_(self.attention_weights[-1].bias.data, 0.)
        nn.init.normal_(self.level_embed, 0.0, 0.02)

    def forward(self, query, keys, values, hf_res_scale=None):
        # N can vary; if N==1, just return query fast-path
        if keys.shape[1] == 1:
            return query
        B, C, H, W = query.shape
        N = min(keys.shape[1], self.num_levels)
        q_low, q_high = self.freq(query)

        ry, rx = torch.meshgrid(
            torch.linspace(0, 1, H, device=query.device),
            torch.linspace(0, 1, W, device=query.device), indexing='ij')
        ref_points = torch.stack([rx, ry], dim=-1).view(1, H*W, 2)

        offsets = self.sampling_offsets(q_low)
        offsets = offsets.view(B, self.num_heads, self.num_levels, self.num_points, 2, H, W)
        offsets = offsets.permute(0, 5, 6, 1, 2, 3, 4).reshape(B, H*W, self.num_heads, self.num_levels, self.num_points, 2)
        offsets = offsets / torch.tensor([W, H], device=query.device).view(1,1,1,1,1,2)
        locs = (ref_points.unsqueeze(2).unsqueeze(3).unsqueeze(4) + offsets).clamp(0,1)

        attn = self.attention_weights(q_low)
        attn = attn.view(B, self.num_heads, self.num_levels, self.num_points, H, W)
        attn = attn.permute(0, 4, 5, 1, 2, 3).reshape(B, H*W, self.num_heads, self.num_levels * self.num_points)
        attn = F.softmax(attn, dim=-1).view(B, H*W, self.num_heads, self.num_levels, self.num_points)

        out = torch.zeros(B, C, H, W, device=query.device)
        for lvl in range(N):
            v_lvl = self.value_proj(values[:, lvl] + self.level_embed[lvl].view(1, -1, 1, 1))
            for head in range(self.num_heads):
                hs, he = head * self.head_dim, (head + 1) * self.head_dim
                v_head = v_lvl[:, hs:he]
                for pt in range(self.num_points):
                    xy = locs[:, :, head, lvl, pt, :] * 2.0 - 1.0
                    xy = xy.view(B, H, W, 2)
                    sampled = F.grid_sample(v_head, xy, mode='bilinear', align_corners=False)
                    w = attn[:, :, head, lvl, pt].view(B, 1, H, W)
                    out[:, hs:he] += sampled * w

        out = self.output_proj(out)
        scale = hf_res_scale if hf_res_scale is not None else self.hf_scale
        return out + q_high * scale