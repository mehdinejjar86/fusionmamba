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
    
    
class SlidingWindowPyramidAttention(nn.Module):
    """
    Memory-efficient sliding window pyramid attention
    Key fixes:
    1. Compute offsets/attention PER WINDOW (not globally)
    2. Use checkpointing for gradient computation
    3. Process windows with optimal batch size
    4. Avoid unnecessary tensor allocations
    """
    def __init__(self, channels, num_heads=4, num_points=4, num_levels=3, 
                 window_size=8, shift_size=0, init_spatial_range=0.1, use_checkpointing=True):
        super().__init__()
        assert channels % num_heads == 0
        assert 0 <= shift_size < window_size, "shift_size must be in [0, window_size)"
        
        self.channels = channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        self.head_dim = channels // num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpointing = use_checkpointing
        
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
    
    def _process_window(self, q_window, values_list, h_start, w_start, H_full, W_full):
        """
        Process a single window - this is where the actual memory savings happen
        """
        B, C, Hw, Ww = q_window.shape
        N = len(values_list)
        
        # Compute offsets and attention ONLY for this window (CRITICAL for memory savings)
        offsets = self.sampling_offsets(q_window)  # [B, heads*levels*points*2, Hw, Ww]
        attn = self.attention_weights(q_window)     # [B, heads*levels*points, Hw, Ww]
        
        # Reshape offsets
        offsets = offsets.view(B, self.num_heads, self.num_levels, self.num_points, 2, Hw, Ww)
        offsets = offsets.permute(0, 5, 6, 1, 2, 3, 4).contiguous()  # [B, Hw, Ww, heads, levels, points, 2]
        
        # Normalize offsets to [-1, 1] range for grid_sample
        # Add window position offset to sample from correct location in full image
        y_base = torch.linspace(h_start, h_start + Hw - 1, Hw, device=q_window.device)
        x_base = torch.linspace(w_start, w_start + Ww - 1, Ww, device=q_window.device)
        y_grid, x_grid = torch.meshgrid(y_base, x_base, indexing='ij')
        
        # Reference points in normalized coordinates
        ref_y = y_grid / (H_full - 1) if H_full > 1 else y_grid
        ref_x = x_grid / (W_full - 1) if W_full > 1 else x_grid
        ref_points = torch.stack([ref_x, ref_y], dim=-1)  # [Hw, Ww, 2]
        ref_points = ref_points.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)  # [1, Hw, Ww, 1, 1, 1, 2]
        
        # Add offsets to reference points
        offsets_normalized = offsets / torch.tensor([W_full, H_full], device=offsets.device).view(1, 1, 1, 1, 1, 1, 2)
        sample_locations = (ref_points + offsets_normalized).clamp(0, 1)
        
        # Reshape attention weights and apply softmax
        attn = attn.view(B, self.num_heads, self.num_levels, self.num_points, Hw, Ww)
        attn = attn.permute(0, 4, 5, 1, 2, 3).contiguous()  # [B, Hw, Ww, heads, levels, points]
        attn = attn.view(B, Hw * Ww, self.num_heads, self.num_levels * self.num_points)
        attn = F.softmax(attn, dim=-1)
        attn = attn.view(B, Hw, Ww, self.num_heads, self.num_levels, self.num_points)
        
        # Accumulate output
        out = torch.zeros(B, C, Hw, Ww, device=q_window.device)
        
        for lvl in range(N):
            # Project values with level embedding
            v_lvl = self.value_proj(values_list[lvl] + self.level_embed[lvl].view(1, -1, 1, 1))
            
            for head in range(self.num_heads):
                hs, he = head * self.head_dim, (head + 1) * self.head_dim
                v_head = v_lvl[:, hs:he]  # [B, head_dim, H, W]
                
                # Sample all points for this head/level at once
                locs = sample_locations[:, :, :, head, lvl, :, :]  # [B, Hw, Ww, points, 2]
                attn_hlvl = attn[:, :, :, head, lvl, :]  # [B, Hw, Ww, points]
                
                for pt in range(self.num_points):
                    # Convert to grid_sample format [-1, 1]
                    xy = locs[:, :, :, pt, :] * 2.0 - 1.0  # [B, Hw, Ww, 2]
                    
                    # Sample from value tensor
                    sampled = F.grid_sample(v_head, xy, mode='bilinear', 
                                          align_corners=False, padding_mode='border')
                    
                    # Apply attention weight
                    w = attn_hlvl[:, :, :, pt].unsqueeze(1)  # [B, 1, Hw, Ww]
                    out[:, hs:he] += sampled * w
        
        return out
    
    def forward(self, query, keys, values, hf_res_scale=None):
        """
        Main forward pass with window-based processing and cyclic shifting
        """
        if keys.shape[1] == 1:
            return query
            
        B, C, H, W = query.shape
        N = min(keys.shape[1], self.num_levels)
        
        # For very small feature maps, use standard processing
        if H <= 16 or W <= 16:
            return self._forward_standard(query, keys, values)
        
        # Determine window size
        window_size = min(self.window_size, H, W)
        
        # Apply cyclic shift if shift_size > 0
        if self.shift_size > 0:
            # Shift query
            query_shifted = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            
            # Shift values
            values_list = []
            for i in range(N):
                values_list.append(torch.roll(values[:, i], shifts=(-self.shift_size, -self.shift_size), dims=(2, 3)))
        else:
            query_shifted = query
            values_list = [values[:, i] for i in range(N)]
        
        # Calculate number of windows
        num_h_windows = (H + window_size - 1) // window_size
        num_w_windows = (W + window_size - 1) // window_size
        
        # Initialize output
        output = torch.zeros_like(query_shifted)
        
        # Process each window
        for h_idx in range(num_h_windows):
            for w_idx in range(num_w_windows):
                h_start = h_idx * window_size
                h_end = min(h_start + window_size, H)
                w_start = w_idx * window_size
                w_end = min(w_start + window_size, W)
                
                # Extract window
                q_window = query_shifted[:, :, h_start:h_end, w_start:w_end]
                
                # Process window (with optional gradient checkpointing)
                if self.use_checkpointing and self.training:
                    from torch.utils.checkpoint import checkpoint
                    out_window = checkpoint(
                        self._process_window, 
                        q_window, values_list, h_start, w_start, H, W,
                        use_reentrant=False
                    )
                else:
                    out_window = self._process_window(
                        q_window, values_list, h_start, w_start, H, W
                    )
                
                # Place window output
                output[:, :, h_start:h_end, w_start:w_end] = out_window
        
        # Reverse cyclic shift if needed
        if self.shift_size > 0:
            output = torch.roll(output, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        
        # Output projection
        output = self.output_proj(output)
        
        return output
    
    def _forward_standard(self, query, keys, values):
        """
        Standard attention for small feature maps (fallback)
        """
        B, C, H, W = query.shape
        N = min(keys.shape[1], self.num_levels)
        
        # Reference points
        ry, rx = torch.meshgrid(
            torch.linspace(0, 1, H, device=query.device),
            torch.linspace(0, 1, W, device=query.device), 
            indexing='ij')
        ref_points = torch.stack([rx, ry], dim=-1).view(1, H*W, 2)
        
        # Compute offsets
        offsets = self.sampling_offsets(query)
        offsets = offsets.view(B, self.num_heads, self.num_levels, 
                              self.num_points, 2, H, W)
        offsets = offsets.permute(0, 5, 6, 1, 2, 3, 4).reshape(
            B, H*W, self.num_heads, self.num_levels, self.num_points, 2)
        offsets = offsets / torch.tensor([W, H], device=query.device).view(1,1,1,1,1,2)
        
        # Sampling locations
        locs = (ref_points.unsqueeze(2).unsqueeze(3).unsqueeze(4) + offsets).clamp(0, 1)
        
        # Attention weights
        attn = self.attention_weights(query)
        attn = attn.view(B, self.num_heads, self.num_levels, self.num_points, H, W)
        attn = attn.permute(0, 4, 5, 1, 2, 3).reshape(
            B, H*W, self.num_heads, self.num_levels * self.num_points)
        attn = F.softmax(attn, dim=-1).view(
            B, H*W, self.num_heads, self.num_levels, self.num_points)
        
        # Output
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