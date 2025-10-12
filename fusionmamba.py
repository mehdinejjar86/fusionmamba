import math
import torch
import torch.nn as nn
from torch.nn import init

from model.vfimamba.vfi_adapter import VFIMambaAdapter
from model.vfimamba.warplayer import warp as warp_vfi 
from model.fusion.utils import ConvBlock, LearnedUpsampling, DetailAwareResBlock
from model.fusion.temporal import TemporalWeightingMLP
from model.fusion.pyramid import PyramidCrossAttention, SlidingWindowPyramidAttention


class AnchorFusionNetVFI(nn.Module):
    """
    Motion-first (VFIMamba) → Multi-view fusion with deformable cross-anchor attention.
    - Always uses wI0 and wI1 as base views (minimum 2 views)
    - Optional N temporal anchors can be added
    - Total views = 2 + N (always >= 2)
    - Flows/masks/warps come from VFIMamba
    """
    def __init__(self, base_channels=64, window_mode=False, 
                 vfi_core=None, vfi_down_scale=1.0, vfi_local=False, freeze_vfi=True):
        super().__init__()
        assert vfi_core is not None, "vfi_core (VFIMamba) must be provided"
        self.base_channels = base_channels
        self.window_mode = window_mode

        # Motion prior (real VFIMamba)
        self.vfi_head = VFIMambaAdapter(vfi_core, down_scale=vfi_down_scale, local=vfi_local, freeze_vfi=freeze_vfi)

        # Encoder (shared for all anchors) – Slim+ inputs: P(3)+M(1)+pe0(1)+pe1(1)+|f|(1) = 7
        enc_in = 7
        C = base_channels
        self.encoder = nn.ModuleDict({
            'low': nn.Sequential(
                ConvBlock(enc_in, C, 7, 1, 3, norm='none', activation='leaky'),
                ConvBlock(C, 2*C, 3, 2, 1, norm='gn', activation='leaky'),
                DetailAwareResBlock(2*C, norm='gn', preserve_details=True)
            ),
            'mid': nn.Sequential(
                ConvBlock(2*C, 4*C, 3, 2, 1, norm='gn', activation='leaky'),
                DetailAwareResBlock(4*C, norm='gn', preserve_details=True),
                DetailAwareResBlock(4*C, norm='gn', preserve_details=True)
            ),
            'high': nn.Sequential(
                ConvBlock(4*C, 8*C, 3, 2, 1, norm='gn', activation='leaky'),
                DetailAwareResBlock(8*C, norm='gn', preserve_details=True),
                DetailAwareResBlock(8*C, norm='gn', preserve_details=True),
                DetailAwareResBlock(8*C, norm='gn', preserve_details=False)
            )
        })

        # Cross-anchor deformable attention
        if self.window_mode:
            self.cross_low  = SlidingWindowPyramidAttention(2*C, num_heads=4, window_size=8, shift_size=4)
            self.cross_mid  = SlidingWindowPyramidAttention(4*C, num_heads=4, window_size=8, shift_size=4)
            self.cross_high = SlidingWindowPyramidAttention(8*C, num_heads=4, window_size=8, shift_size=4)
        else:
            self.cross_low  = PyramidCrossAttention(2*C, num_heads=4)
            self.cross_mid  = PyramidCrossAttention(4*C, num_heads=4)
            self.cross_high = PyramidCrossAttention(8*C, num_heads=4)

        # Decoder
        self.up_high_to_mid = nn.Sequential(
            LearnedUpsampling(8*C, 4*C, 2),
            ConvBlock(4*C, 4*C, norm='gn', activation='leaky')
        )
        self.fuse_mid = nn.Sequential(
            ConvBlock(8*C, 4*C, norm='gn', activation='leaky'),
            DetailAwareResBlock(4*C, norm='gn', preserve_details=True),
        )
        self.up_mid_to_low = nn.Sequential(
            LearnedUpsampling(4*C, 2*C, 2),
            ConvBlock(2*C, 2*C, norm='gn', activation='leaky')
        )
        self.fuse_low = nn.Sequential(
            ConvBlock(4*C, 2*C, norm='gn', activation='leaky'),
            DetailAwareResBlock(2*C, norm='gn', preserve_details=True),
        )
        self.up_to_full = nn.Sequential(
            LearnedUpsampling(2*C, C, 2),
            ConvBlock(C, C, norm='gn', activation='leaky')
        )

        # Synthesis
        self.synthesis = nn.Sequential(
            ConvBlock(C + 3, C, norm='gn', activation='leaky'),
            DetailAwareResBlock(C, norm='gn', preserve_details=True),
            DetailAwareResBlock(C, norm='gn', preserve_details=True),
            ConvBlock(C, C // 2, norm='gn', activation='leaky'),
            ConvBlock(C // 2, 3, activation='sigmoid')
        )
        self.residual_head = nn.Sequential(
            ConvBlock(C, C // 2, norm='gn', activation='leaky'),
            ConvBlock(C // 2, 3, activation='tanh')
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # Temporal weighter (fusion-only; handles 2+N views)
        self.temporal_weighter = TemporalWeightingMLP(hidden_dim=128)
        self.temporal_temperature = nn.Parameter(torch.tensor(1.0))

        # Spectral swap params
        self.spectral_alpha = nn.Parameter(torch.tensor(0.3))
        self.spectral_lo = 0.32
        self.spectral_hi = 0.50
        self.spectral_soft = True
        self.detail_weight = nn.Parameter(torch.tensor(0.3))

        self._init_weights()

    # ---------- init & utils ----------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None: init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if getattr(m, 'weight', None) is not None: init.constant_(m.weight, 1)
                if getattr(m, 'bias', None) is not None: init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: init.constant_(m.bias, 0)

    @staticmethod
    def _hf_mask(H, W, lo, hi, soft=True, device='cpu'):
        fy = torch.fft.fftfreq(H, d=1.0).to(device).abs()
        fx = torch.fft.rfftfreq(W, d=1.0).to(device).abs()
        wy, wx = torch.meshgrid(fy, fx, indexing='ij')
        r = torch.sqrt(wx**2 + wy**2)
        if soft:
            t = ((r - lo) / max(hi - lo, 1e-6)).clamp(0, 1)
            mask = 0.5 - 0.5 * torch.cos(math.pi * t)
        else:
            mask = (r >= lo).float()
        return mask.view(1, 1, H, W // 2 + 1)

    def _spectral_swap(self, base, prior, lo=0.32, hi=0.50, alpha=0.3, soft=True):
        B, C, H, W = base.shape
        X = torch.fft.rfft2(base, dim=(-2, -1), norm="ortho")
        P = torch.fft.rfft2(prior, dim=(-2, -1), norm="ortho")
        mag_x, mag_p = torch.abs(X), torch.abs(P)
        phase = torch.angle(X)
        mask = self._hf_mask(H, W, lo, hi, soft, device=base.device)
        new_mag = mag_x + mask * alpha * (mag_p - mag_x)
        Y = new_mag * torch.exp(1j * phase)
        return torch.fft.irfft2(Y, s=(H, W), dim=(-2, -1), norm="ortho")

    def _build_slim_inputs(self, I0b, I1b, f01b, f10b, Mb):
        """Helper to build 7-channel Slim+ input from warps and mask"""
        wI0 = warp_vfi(I0b, f01b)
        wI1 = warp_vfi(I1b, f10b)
        P   = Mb * wI0 + (1 - Mb) * wI1
        pe0 = (wI0 - I0b).abs().mean(1, keepdim=True)
        pe1 = (wI1 - I1b).abs().mean(1, keepdim=True)
        mag = (f01b.pow(2).sum(1, keepdim=True)).sqrt()
        Xb  = torch.cat([P, Mb, pe0, pe1, mag], dim=1)
        return Xb, (wI0, wI1, P)

    def _build_dual_view_anchors(self, I0b, I1b, f01b, f10b):
        """
        Build two base views from wI0 and wI1 (always present).
        These are predictions at time t from different boundary frames.
        """
        wI0 = warp_vfi(I0b, f01b)  # I0 warped to time t
        wI1 = warp_vfi(I1b, f10b)  # I1 warped to time t
        
        # View 1: wI0-based (mask=0 → trust I0 warp)
        M0 = torch.zeros_like(f01b[:, :1])
        P0 = wI0
        pe0_v0 = (wI0 - I0b).abs().mean(1, keepdim=True)
        pe1_v0 = (wI1 - I1b).abs().mean(1, keepdim=True)
        mag0 = f01b.pow(2).sum(1, keepdim=True).sqrt()
        X0 = torch.cat([P0, M0, pe0_v0, pe1_v0, mag0], dim=1)
        
        # View 2: wI1-based (mask=1 → trust I1 warp)
        M1 = torch.ones_like(f10b[:, :1])
        P1 = wI1
        pe0_v1 = (wI0 - I0b).abs().mean(1, keepdim=True)
        pe1_v1 = (wI1 - I1b).abs().mean(1, keepdim=True)
        mag1 = f10b.pow(2).sum(1, keepdim=True).sqrt()
        X1 = torch.cat([P1, M1, pe0_v1, pe1_v1, mag1], dim=1)
        
        return torch.stack([X0, X1], dim=1), torch.stack([wI0, wI1], dim=1)

    # ---------- forward ----------
    def forward(self, I0_all, I1_all, timesteps, flows_all=None, masks_all=None):
        """
        I0_all, I1_all: [B,N,3,H,W] - N temporal anchors (can be 0)
        timesteps:      [B,N] in [0,1] - timesteps for temporal anchors
        flows_all/masks_all: optional overrides
        
        Architecture always uses 2 base views (wI0, wI1) + N temporal anchors.
        Total views = 2 + N (minimum 2).
        """
        B, N, _, H, W = I0_all.shape
        
        # Prepare boundary frames
        I0 = I0_all[:, 0] if N > 0 else I0_all.squeeze(1)  # [B,3,H,W]
        I1 = I1_all[:, 0] if N > 0 else I1_all.squeeze(1)  # [B,3,H,W]
        
        # 1) Get flows/masks for base views (use first temporal anchor's time or t=0.5)
        t_base = timesteps[:, 0:1] if N > 0 else torch.full((B, 1), 0.5, device=I0.device)
        
        if flows_all is None or masks_all is None:
            flows_base, masks_base = self.vfi_head(I0, I1, t_base)
        else:
            flows_base = flows_all[:, 0] if N > 0 else flows_all.squeeze(1)
            masks_base = masks_all[:, 0] if N > 0 else masks_all.squeeze(1)
        
        f01_base, f10_base = flows_base[:, :2], flows_base[:, 2:]
        
        # 2) Build BASE VIEWS: always 2 (wI0, wI1)
        X_base, P_base = self._build_dual_view_anchors(I0, I1, f01_base, f10_base)
        # X_base: [B,2,7,H,W], P_base: [B,2,3,H,W]
        
        # 3) Build TEMPORAL ANCHOR VIEWS if N > 0
        if N > 0:
            BN = B * N
            I0b = I0_all.view(BN, 3, H, W)
            I1b = I1_all.view(BN, 3, H, W)
            tb = timesteps.view(BN, 1)
            
            if flows_all is None or masks_all is None:
                flows_b, masks_b = self.vfi_head(I0b, I1b, tb)
            else:
                flows_b = flows_all.view(BN, 4, H, W)
                masks_b = masks_all.view(BN, 1, H, W)
            
            f01b, f10b = flows_b[:, :2], flows_b[:, 2:]
            Xb, warped_triplet = self._build_slim_inputs(I0b, I1b, f01b, f10b, masks_b)
            Pb = warped_triplet[2]
            
            X_temporal = Xb.view(B, N, 7, H, W)
            P_temporal = Pb.view(B, N, 3, H, W)
            
            # Combine base + temporal
            X = torch.cat([X_base, X_temporal], dim=1)  # [B,2+N,7,H,W]
            P = torch.cat([P_base, P_temporal], dim=1)  # [B,2+N,3,H,W]
        else:
            # Only base views
            X = X_base  # [B,2,7,H,W]
            P = P_base  # [B,2,3,H,W]
        
        N_total = X.size(1)  # Total views = 2 + N
        
        # 4) Temporal weights for ALL views (2+N)
        # Create timesteps for base views + temporal anchors
        if N > 0:
            # Base views get the first temporal anchor's timestep
            t_all = torch.cat([
                t_base.expand(B, 2),  # Same timestep for wI0 and wI1
                timesteps
            ], dim=1)  # [B, 2+N]
        else:
            # Only base views, use t=0.5 for both
            t_all = torch.full((B, 2), 0.5, device=I0.device)  # [B, 2]
        
        t_weights = self.temporal_weighter(t_all * self.temporal_temperature)  # [B,2+N]
        t_weights = t_weights / (t_weights.sum(dim=1, keepdim=True) + 1e-8)
        w = t_weights.view(B, N_total, 1, 1, 1)
        
        # 5) Encode all views (shared encoder)
        BN_total = B * N_total
        Xf = X.view(BN_total, 7, H, W)
        low = self.encoder['low'](Xf)
        mid = self.encoder['mid'](low)
        high = self.encoder['high'](mid)
        C = self.base_channels
        
        low = (low.view(B, N_total, 2*C, H//2, W//2)) * w
        mid = (mid.view(B, N_total, 4*C, H//4, W//4)) * w
        high = (high.view(B, N_total, 8*C, H//8, W//8)) * w
        
        # 6) Cross-anchor fusion (ALWAYS runs, minimum 2 views)
        Qh = high.sum(dim=1)
        Fh = self.cross_high(Qh, high, high)
        Um = self.up_high_to_mid[0](Fh, target_size=(H//4, W//4))
        Um = self.up_high_to_mid[1](Um)
        
        Qm = mid.sum(dim=1)
        Fm = self.cross_mid(Qm, mid, mid)
        Fm = self.fuse_mid(torch.cat([Um, Fm], dim=1))
        
        Ul = self.up_mid_to_low[0](Fm, target_size=(H//2, W//2))
        Ul = self.up_mid_to_low[1](Ul)
        
        Ql = low.sum(dim=1)
        Fl = self.cross_low(Ql, low, low)
        Fl = self.fuse_low(torch.cat([Ul, Fl], dim=1))
        
        D = self.up_to_full[0](Fl, target_size=(H, W))
        D = self.up_to_full[1](D)
        
        # 7) Weighted RGB prior
        P_bar = (P * w).sum(dim=1)
        
        # 8) Synthesis
        synth_in = torch.cat([D, P_bar], dim=1)
        synthesized = self.synthesis(synth_in)
        residual = self.residual_head(D) * self.residual_scale
        out = synthesized + residual
        
        # 9) Spectral HF swap with prior
        alpha = self.spectral_alpha.clamp(0.0, 1.0)
        if alpha.item() > 0:
            out = self._spectral_swap(out, P_bar, lo=self.spectral_lo, hi=self.spectral_hi,
                                      alpha=float(alpha.item()), soft=self.spectral_soft)
        out = torch.clamp(out, 0, 1)
        
        # 10) Prepare auxiliary outputs (flows/masks for all views)
        # Stack base flows + temporal flows
        if N > 0:
            flows_all = torch.cat([
                flows_base.unsqueeze(1).expand(B, 2, 4, H, W),  # Base flows for 2 views
                flows_b.view(B, N, 4, H, W)  # Temporal flows
            ], dim=1)  # [B, 2+N, 4, H, W]
            
            masks_all = torch.cat([
                masks_base.unsqueeze(1).expand(B, 2, 1, H, W),  # Base masks for 2 views
                masks_b.view(B, N, 1, H, W)  # Temporal masks
            ], dim=1)  # [B, 2+N, 1, H, W]
        else:
            flows_all = flows_base.unsqueeze(1).expand(B, 2, 4, H, W)
            masks_all = masks_base.unsqueeze(1).expand(B, 2, 1, H, W)
        
        aux = {
            't_weights': t_weights.detach(),
            'prior_weighted': P_bar.detach(),
            'flows': flows_all.detach(),  # [B, 2+N, 4, H, W]
            'masks': masks_all.detach(),  # [B, 2+N, 1, H, W]
            'n_total_views': N_total,
        }
        return out, aux


# ---------- factory ----------
def build_fusion_net_vfi(base_channels=64, window_mode=True,
                         vfi_core=None, vfi_down_scale=1.0, vfi_local=False, freeze_vfi=True):
    return AnchorFusionNetVFI(
        base_channels=base_channels,
        window_mode=window_mode,
        vfi_core=vfi_core,
        vfi_down_scale=vfi_down_scale,
        vfi_local=vfi_local,
        freeze_vfi=freeze_vfi
    )