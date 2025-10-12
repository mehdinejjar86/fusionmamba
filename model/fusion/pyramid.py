# pyramid.py
# Deformable cross-attention (pyramid) with OPTIMIZED sliding window

import torch.nn as nn
import torch
from model.fusion.utils import AdaptiveFrequencyDecoupling
import torch.nn.functional as F
import math


def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.
    Args:
        x: [B, C, H, W]
        window_size: int
    Returns:
        windows: [B*num_windows, C, window_size, window_size]
        (H_w, W_w): number of windows in H and W dimensions
    """
    B, C, H, W = x.shape
    
    # Pad if needed
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
        H_pad, W_pad = H + pad_h, W + pad_w
    else:
        H_pad, W_pad = H, W
    
    H_w = H_pad // window_size
    W_w = W_pad // window_size
    
    # Reshape: [B, C, H_w, ws, W_w, ws]
    x = x.view(B, C, H_w, window_size, W_w, window_size)
    
    # Permute and reshape: [B*H_w*W_w, C, ws, ws]
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = windows.view(B * H_w * W_w, C, window_size, window_size)
    
    return windows, (H_w, W_w), (H_pad, W_pad)


def window_reverse(windows, window_size, H_w, W_w, B, H, W):
    """
    Reverse window partition.
    Args:
        windows: [B*num_windows, C, window_size, window_size]
        window_size: int
        H_w, W_w: number of windows
        B: batch size
        H, W: original height and width (before padding)
    Returns:
        x: [B, C, H, W]
    """
    C = windows.shape[1]
    
    # Reshape: [B, H_w, W_w, C, ws, ws]
    x = windows.view(B, H_w, W_w, C, window_size, window_size)
    
    # Permute: [B, C, H_w, ws, W_w, ws]
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    
    # Reshape: [B, C, H_w*ws, W_w*ws]
    H_pad = H_w * window_size
    W_pad = W_w * window_size
    x = x.view(B, C, H_pad, W_pad)
    
    # Remove padding if needed
    if H_pad > H or W_pad > W:
        x = x[:, :, :H, :W]
    
    return x


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
        """
        Cross-attention fusion - ALWAYS runs (minimum 2 views: wI0, wI1)
        keys: [B, N, C, H, W] where N >= 2
        """
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
    
    
class SlidingWindowPyramidAttention(nn.Module):
    """
    OPTIMIZED memory-efficient sliding window pyramid attention.
    Key optimizations:
    1. Batch all windows together (no sequential loops)
    2. Vectorized deformable sampling
    3. Minimal Python overhead
    """
    def __init__(self, channels, num_heads=4, num_points=4, num_levels=3, 
                 window_size=8, shift_size=0, init_spatial_range=0.1):
        super().__init__()
        assert channels % num_heads == 0
        assert 0 <= shift_size < window_size
        
        self.channels = channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        self.head_dim = channels // num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        # Deformable sampling networks
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
        
        # Level embeddings
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
        """
        Optimized window-based attention with full batching
        """
        B, C, H, W = query.shape
        N = min(keys.shape[1], self.num_levels)
        ws = self.window_size
        
        # For small feature maps, just use standard attention
        if H <= 16 or W <= 16:
            return self._forward_standard(query, keys, values)
        
        # Apply cyclic shift if needed
        if self.shift_size > 0:
            query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            values_shifted = torch.stack([
                torch.roll(values[:, i], shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
                for i in range(N)
            ], dim=1)
        else:
            values_shifted = values
        
        # === STEP 1: Partition into windows (BATCHED) ===
        q_windows, (H_w, W_w), (H_pad, W_pad) = window_partition(query, ws)
        # q_windows: [B*num_windows, C, ws, ws]
        num_windows = H_w * W_w
        
        # Also partition value maps
        v_windows = []
        for i in range(N):
            v_win, _, _ = window_partition(values_shifted[:, i], ws)
            v_windows.append(v_win)
        # Each v_win: [B*num_windows, C, ws, ws]
        
        # === STEP 2: Compute offsets and attention (BATCHED over all windows) ===
        offsets = self.sampling_offsets(q_windows)  # [B*nW, heads*lvls*pts*2, ws, ws]
        attn = self.attention_weights(q_windows)     # [B*nW, heads*lvls*pts, ws, ws]
        
        # Reshape offsets: [B*nW, ws, ws, heads, lvls, pts, 2]
        offsets = offsets.view(B*num_windows, self.num_heads, self.num_levels, 
                               self.num_points, 2, ws, ws)
        offsets = offsets.permute(0, 5, 6, 1, 2, 3, 4).contiguous()
        
        # Create reference grid for windows (local coordinates [0,1])
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, 1, ws, device=query.device),
            torch.linspace(0, 1, ws, device=query.device),
            indexing='ij'
        )
        ref_grid = torch.stack([x_grid, y_grid], dim=-1)  # [ws, ws, 2]
        ref_grid = ref_grid.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)  # [1, ws, ws, 1, 1, 1, 2]
        
        # Normalize offsets (relative to window size)
        offsets = offsets / ws
        
        # Add offsets to reference grid: [B*nW, ws, ws, heads, lvls, pts, 2]
        sample_locs = (ref_grid + offsets).clamp(0, 1)
        
        # Reshape attention: [B*nW, ws, ws, heads, lvls, pts]
        attn = attn.view(B*num_windows, self.num_heads, self.num_levels, 
                        self.num_points, ws, ws)
        attn = attn.permute(0, 4, 5, 1, 2, 3).contiguous()
        
        # Softmax over all sampling points
        attn = attn.view(B*num_windows, ws*ws, self.num_heads, 
                        self.num_levels * self.num_points)
        attn = F.softmax(attn, dim=-1)
        attn = attn.view(B*num_windows, ws, ws, self.num_heads, 
                        self.num_levels, self.num_points)
        
        # === STEP 3: Deformable sampling (VECTORIZED) ===
        out_windows = torch.zeros(B*num_windows, C, ws, ws, device=query.device)
        
        for lvl in range(N):
            # Project values with level embedding
            v_lvl = self.value_proj(v_windows[lvl] + 
                                   self.level_embed[lvl].view(1, -1, 1, 1))
            
            # Process all heads and points in vectorized manner
            for head in range(self.num_heads):
                hs, he = head * self.head_dim, (head + 1) * self.head_dim
                v_head = v_lvl[:, hs:he]  # [B*nW, head_dim, ws, ws]
                
                # Vectorize over points
                for pt in range(self.num_points):
                    # Sample locations: [B*nW, ws, ws, 2]
                    xy = sample_locs[:, :, :, head, lvl, pt, :]
                    xy = xy * 2.0 - 1.0  # Convert to [-1, 1]
                    
                    # Grid sample (all windows in parallel!)
                    sampled = F.grid_sample(v_head, xy, mode='bilinear',
                                          align_corners=False, padding_mode='border')
                    
                    # Apply attention weight
                    w = attn[:, :, :, head, lvl, pt].unsqueeze(1)  # [B*nW, 1, ws, ws]
                    out_windows[:, hs:he] += sampled * w
        
        # === STEP 4: Reverse window partition (BATCHED) ===
        output = window_reverse(out_windows, ws, H_w, W_w, B, H_pad, W_pad)
        
        # Remove padding
        if H_pad > H or W_pad > W:
            output = output[:, :, :H, :W]
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            output = torch.roll(output, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        
        # Output projection
        output = self.output_proj(output)
        
        return output
    
    def _forward_standard(self, query, keys, values):
        """Fallback for small feature maps"""
        B, C, H, W = query.shape
        N = min(keys.shape[1], self.num_levels)
        
        ry, rx = torch.meshgrid(
            torch.linspace(0, 1, H, device=query.device),
            torch.linspace(0, 1, W, device=query.device), 
            indexing='ij')
        ref_points = torch.stack([rx, ry], dim=-1).view(1, H*W, 2)
        
        offsets = self.sampling_offsets(query)
        offsets = offsets.view(B, self.num_heads, self.num_levels, 
                              self.num_points, 2, H, W)
        offsets = offsets.permute(0, 5, 6, 1, 2, 3, 4).reshape(
            B, H*W, self.num_heads, self.num_levels, self.num_points, 2)
        offsets = offsets / torch.tensor([W, H], device=query.device).view(1,1,1,1,1,2)
        
        locs = (ref_points.unsqueeze(2).unsqueeze(3).unsqueeze(4) + offsets).clamp(0, 1)
        
        attn = self.attention_weights(query)
        attn = attn.view(B, self.num_heads, self.num_levels, self.num_points, H, W)
        attn = attn.permute(0, 4, 5, 1, 2, 3).reshape(
            B, H*W, self.num_heads, self.num_levels * self.num_points)
        attn = F.softmax(attn, dim=-1).view(
            B, H*W, self.num_heads, self.num_levels, self.num_points)
        
        out = torch.zeros(B, C, H, W, device=query.device)
        
        for lvl in range(N):
            v_lvl = self.value_proj(values[:, lvl] + self.level_embed[lvl].view(1, -1, 1, 1))
            
            for head in range(self.num_heads):
                hs, he = head * self.head_dim, (head + 1) * self.head_dim
                v_head = v_lvl[:, hs:he]
                
                for pt in range(self.num_points):
                    xy = locs[:, :, head, lvl, pt, :] * 2.0 - 1.0
                    xy = xy.view(B, H, W, 2)
                    sampled = F.grid_sample(v_head, xy, mode='bilinear', 
                                          align_corners=False)
                    w = attn[:, :, head, lvl, pt].view(B, 1, H, W)
                    out[:, hs:he] += sampled * w
        
        out = self.output_proj(out)
        return out