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
    
"""
Wavelet-based pyramid cross-attention for memory-efficient multi-view fusion.
Only the attention mechanism operates on wavelet-decomposed features.
"""

from pytorch_wavelets import DWTForward, DWTInverse
class WaveletPyramidCrossAttention(nn.Module):
    """
    Memory-efficient cross-attention that PRESERVES multi-view awareness.
    All anchors are processed together in the wavelet domain.
    """
    def __init__(self, dim, num_heads=4, wavelet='haar', wavelet_level=1,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.wavelet_level = wavelet_level
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Wavelet transforms
        self.dwt = DWTForward(J=wavelet_level, wave=wavelet, mode='symmetric')
        self.idwt = DWTInverse(wave=wavelet, mode='symmetric')
        
        # Attention projections
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Normalization
        self.norm = nn.GroupNorm(num_groups=min(32, dim // 4), num_channels=dim)
        
        reduction = 2 ** (2 * wavelet_level)
        print(f"  Wavelet Pyramid Attention: {dim}D, level={wavelet_level}, "
              f"reduction={reduction}x")
    
    def forward(self, query, key_value_multi, value_multi):
        """
        Parallel multi-view wavelet attention.
        
        Args:
            query: [B, C, H, W] - aggregated query
            key_value_multi: [B, N, C, H, W] - ALL anchor features
            value_multi: [B, N, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, N, C, H, W = key_value_multi.shape
        
        # === 1. Decompose query ===
        query_yl, query_yh = self.dwt(query)  # [B, C, H', W']
        _, _, H_low, W_low = query_yl.shape
        
        # === 2. Decompose ALL anchors in parallel ===
        # Flatten N into batch for efficient parallel wavelet transform
        kv_flat = key_value_multi.view(B * N, C, H, W)  # [B*N, C, H, W]
        kv_yl_flat, kv_yh_flat = self.dwt(kv_flat)      # [B*N, C, H', W']
        
        # Reshape back to separate anchors
        kv_yl = kv_yl_flat.view(B, N, C, H_low, W_low)  # [B, N, C, H', W']
        
        # Free high-freq from anchors
        del kv_yh_flat, kv_flat
        
        # === 3. Apply projections ===
        q = self.q_proj(query_yl)  # [B, C, H', W']
        
        # Project all anchor keys/values in parallel
        kv_yl_flat_for_proj = kv_yl.reshape(B * N, C, H_low, W_low)  # Use reshape
        k_flat = self.k_proj(kv_yl_flat_for_proj)  # [B*N, C, H', W']
        v_flat = self.v_proj(kv_yl_flat_for_proj)  # [B*N, C, H', W']
        
        del kv_yl_flat_for_proj
        
        # === 4. Reshape for multi-head attention ===
        # Query: [B, num_heads, H'W', head_dim]
        q = q.reshape(B, self.num_heads, C // self.num_heads, H_low * W_low)
        q = q.transpose(2, 3)  # [B, heads, H'W', dim]
        
        # Keys: [B*N, C, H', W'] -> [B, N, heads, H'W', dim]
        k = k_flat.reshape(B, N, C, H_low, W_low)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads, H_low, W_low)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads, H_low * W_low)
        k = k.permute(0, 2, 1, 4, 3)  # [B, heads, N, H'W', dim]
        k = k.reshape(B, self.num_heads, N * H_low * W_low, C // self.num_heads)  # [B, heads, N*H'W', dim]
        
        # Values: [B*N, C, H', W'] -> [B, N, heads, H'W', dim]
        v = v_flat.reshape(B, N, C, H_low, W_low)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads, H_low, W_low)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads, H_low * W_low)
        v = v.permute(0, 2, 1, 4, 3)  # [B, heads, N, H'W', dim]
        v = v.reshape(B, self.num_heads, N * H_low * W_low, C // self.num_heads)  # [B, heads, N*H'W', dim]
        
        del k_flat, v_flat
        
        # === 5. Cross-attention across ALL anchors ===
        # Attention: query [B, heads, H'W', dim] @ keys [B, heads, N*H'W', dim].T
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, H'W', N*H'W']
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply to values
        out = attn @ v  # [B, heads, H'W', dim]
        out = out.transpose(2, 3).contiguous()  # [B, heads, dim, H'W']
        out = out.reshape(B, C, H_low, W_low)
        
        del attn, k, v, q
        
        # === 6. Project output ===
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # === 7. Reconstruct to full resolution ===
        out_full = self.idwt((out, query_yh))
        del out, query_yh
        
        # Crop to original size
        out_full = out_full[:, :, :H, :W]
        
        # === 8. Normalize and residual ===
        out_full = self.norm(out_full)
        out_full = out_full + query
        
        return out_full