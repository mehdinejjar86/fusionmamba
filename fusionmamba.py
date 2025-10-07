# fusionmamba.py
import math
import torch
import torch.nn as nn
from torch.nn import init

from model.vfimamba.vfi_adapter import VFIMambaAdapter
from model.vfimamba.warplayer import warp as warp_vfi 
from model.fusion.utils import ConvBlock, LearnedUpsampling, DetailAwareResBlock
from model.fusion.temporal import TemporalWeightingMLP
from model.fusion.pyramid import PyramidCrossAttention, WaveletPyramidCrossAttention


class AnchorFusionNetVFI(nn.Module):
    """
    Motion-first (VFIMamba) → Slim+ multi-prior fusion with configurable cross-anchor attention.
    - Flows/masks/warps come from real VFIMamba
    - Temporal weights are used ONLY in fusion (not in motion)
    - Works with variable N (including N=1)
    - Supports both standard and wavelet-based attention for memory efficiency
    """
    def __init__(self, base_channels=64, 
                 vfi_core=None, vfi_down_scale=1.0, vfi_local=False, freeze_vfi=True,
                 use_wavelet_attention=False, wavelet_level=2, wavelet_type='haar'):
        """
        Args:
            base_channels: Base number of feature channels
            vfi_core: VFIMamba core model for flow/mask prediction
            vfi_down_scale: Downscale factor for VFI processing
            vfi_local: Use local refinement in VFI
            freeze_vfi: Freeze VFIMamba weights
            use_wavelet_attention: If True, use wavelet-based cross-attention (memory-efficient)
                                   If False, use standard cross-attention
            wavelet_level: Decomposition level for wavelet attention (1 or 2)
                - 1: 4x memory reduction
                - 2: 16x memory reduction (recommended for 4K+)
            wavelet_type: Wavelet type ('haar', 'db2', 'db4', 'sym2')
        """
        super().__init__()
        assert vfi_core is not None, "vfi_core (VFIMamba) must be provided"
        self.base_channels = base_channels
        self.use_wavelet_attention = use_wavelet_attention

        # Motion prior (real VFIMamba)
        self.vfi_head = VFIMambaAdapter(vfi_core, down_scale=vfi_down_scale, 
                                       local=vfi_local, freeze_vfi=freeze_vfi)

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

        # Cross-anchor attention (configurable: standard or wavelet)
        if use_wavelet_attention:
            print(f"\n🌊 Using Wavelet-based Pyramid Attention (memory-efficient):")
            print(f"  Wavelet: {wavelet_type}, Level: {wavelet_level}")
            print(f"  Memory reduction: {2**(2*wavelet_level)}x per attention layer")
            
            self.cross_low  = WaveletPyramidCrossAttention(
                2*C, num_heads=4, wavelet=wavelet_type, wavelet_level=wavelet_level
            )
            self.cross_mid  = WaveletPyramidCrossAttention(
                4*C, num_heads=4, wavelet=wavelet_type, wavelet_level=wavelet_level
            )
            self.cross_high = WaveletPyramidCrossAttention(
                8*C, num_heads=4, wavelet=wavelet_type, wavelet_level=wavelet_level
            )
        else:
            print("\n📐 Using Standard Pyramid Attention:")
            print("  Full-resolution cross-attention (higher memory usage)")
            
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

        # Temporal weighter (fusion-only; N-agnostic)
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
        # Use VFIMamba's warp for exact sampling convention
        wI0 = warp_vfi(I0b, f01b)   # [BN,3,H,W]
        wI1 = warp_vfi(I1b, f10b)   # [BN,3,H,W]
        P   = Mb * wI0 + (1 - Mb) * wI1
        pe0 = (wI0 - I0b).abs().mean(1, keepdim=True)
        pe1 = (wI1 - I1b).abs().mean(1, keepdim=True)
        mag = (f01b.pow(2).sum(1, keepdim=True)).sqrt()
        Xb  = torch.cat([P, Mb, pe0, pe1, mag], dim=1)  # [BN,7,H,W]
        return Xb, (wI0, wI1, P)

    # ---------- forward ----------
    def forward(self, I0_all, I1_all, timesteps, flows_all=None, masks_all=None):
        """
        I0_all, I1_all: [B,N,3,H,W]
        timesteps:      [B,N] in [0,1]
        flows_all/masks_all: optional overrides (else predicted by VFIMamba)
        """
        B, N, _, H, W = I0_all.shape
        BN = B * N

        I0b = I0_all.view(BN, 3, H, W)
        I1b = I1_all.view(BN, 3, H, W)
        tb  = timesteps.view(BN, 1)

        # 1) Motion prior (VFIMamba)
        if flows_all is None or masks_all is None:
            flows_b, masks_b = self.vfi_head(I0b, I1b, tb)     # [BN,4,H,W], [BN,1,H,W]
        else:
            flows_b = flows_all.view(BN, 4, H, W)
            masks_b = masks_all.view(BN, 1, H, W)

        f01b, f10b = flows_b[:, :2], flows_b[:, 2:]
        # sanity
        assert f01b.shape[-2:] == I0b.shape[-2:], "f01/I0 mismatch"
        assert f10b.shape[-2:] == I1b.shape[-2:], "f10/I1 mismatch"

        # Slim+ per-anchor tensors at full-res
        Xb, warped_triplet = self._build_slim_inputs(I0b, I1b, f01b, f10b, masks_b)
        Pb = warped_triplet[2]
        X  = Xb.view(B, N, Xb.size(1), H, W)
        P  = Pb.view(B, N, 3, H, W)
        M  = masks_b.view(B, N, 1, H, W)

        # 2) Fusion weights (fusion-only)
        t_weights = self.temporal_weighter(timesteps * self.temporal_temperature)  # [B,N]
        if N == 1:
            t_weights = torch.ones_like(t_weights)
        t_weights = t_weights / (t_weights.sum(dim=1, keepdim=True) + 1e-8)
        w = t_weights.view(B, N, 1, 1, 1)

        # 3) Encode per anchor (shared)
        Xf  = X.view(BN, X.size(2), H, W)
        low = self.encoder['low'](Xf)              # [BN,2C,H/2,W/2]
        mid = self.encoder['mid'](low)             # [BN,4C,H/4,W/4]
        high= self.encoder['high'](mid)            # [BN,8C,H/8,W/8]
        C   = self.base_channels

        low  = (low .view(B, N, 2*C, H//2, W//2)) * w
        mid  = (mid .view(B, N, 4*C, H//4, W//4)) * w
        high = (high.view(B, N, 8*C, H//8, W//8)) * w

        # 4) Cross-anchor fusion (wavelet or standard based on config)
        Qh = high.sum(dim=1)                                   # [B,8C,H/8,W/8]
        Fh = self.cross_high(Qh, high, high)                   # Pass high as both K and V
        Um = self.up_high_to_mid[0](Fh, target_size=(H//4, W//4))
        Um = self.up_high_to_mid[1](Um)

        Qm = mid.sum(dim=1)                                    # [B,4C,H/4,W/4]
        Fm = self.cross_mid(Qm, mid, mid)                      # Pass mid as both K and V
        Fm = self.fuse_mid(torch.cat([Um, Fm], dim=1))

        Ul = self.up_mid_to_low[0](Fm, target_size=(H//2, W//2))
        Ul = self.up_mid_to_low[1](Ul)

        Ql = low.sum(dim=1)                                    # [B,2C,H/2,W/2]
        Fl = self.cross_low(Ql, low, low)                      # Pass low as both K and V
        Fl = self.fuse_low(torch.cat([Ul, Fl], dim=1))

        D  = self.up_to_full[0](Fl, target_size=(H, W))
        D  = self.up_to_full[1](D)                             # [B,C,H,W]

        # 5) Weighted RGB prior
        P_bar = (P * w).sum(dim=1)                             # [B,3,H,W]

        # 6) Synthesis
        synth_in   = torch.cat([D, P_bar], dim=1)
        synthesized= self.synthesis(synth_in)
        residual   = self.residual_head(D) * self.residual_scale
        out        = synthesized + residual

        # 7) Optional spectral HF swap with prior
        alpha = self.spectral_alpha.clamp(0.0, 1.0)
        if alpha.item() > 0:
            out = self._spectral_swap(out, P_bar, lo=self.spectral_lo, hi=self.spectral_hi,
                                      alpha=float(alpha.item()), soft=self.spectral_soft)
        out = torch.clamp(out, 0, 1)

        aux = {
            't_weights': t_weights.detach(),
            'prior_weighted': P_bar.detach(),
            'flows': flows_b.view(B, N, 4, H, W).detach(),
            'masks': M.detach(),
        }
        return out, aux


# ---------- factory ----------
def build_fusion_net_vfi(base_channels=64,
                         vfi_core=None, vfi_down_scale=1.0, vfi_local=False,
                         freeze_vfi=True,
                         use_wavelet_attention=False, wavelet_level=2, wavelet_type='haar'):
    """
    Build AnchorFusionNet with VFIMamba and configurable attention.
    
    Args:
        base_channels: Base number of feature channels
        vfi_core: VFIMamba core model
        vfi_down_scale: VFI downscale factor
        vfi_local: Use local refinement in VFI
        freeze_vfi: Freeze VFIMamba weights
        
        use_wavelet_attention: Enable wavelet-based cross-attention
            - False: Standard full-resolution attention (default)
            - True: Wavelet-based attention (memory-efficient for 4K+)
        
        wavelet_level: Wavelet decomposition level (only used if use_wavelet_attention=True)
            - 1: 4x memory reduction (good for 2K images)
            - 2: 16x memory reduction (recommended for 4K images)
        
        wavelet_type: Wavelet type (only used if use_wavelet_attention=True)
            - 'haar': Fastest, good quality (recommended)
            - 'db2': Better quality, slightly slower
            - 'db4': Best quality, slower
            - 'sym2': Symmetric wavelets
    
    Returns:
        AnchorFusionNetVFI model
    
    Examples:
        # Standard attention (for HD/2K)
        model = build_fusion_net_vfi(
            base_channels=48,
            vfi_core=vfi_core,
            use_wavelet_attention=False
        )
        
        # Wavelet attention (for 4K)
        model = build_fusion_net_vfi(
            base_channels=48,
            vfi_core=vfi_core,
            use_wavelet_attention=True,
            wavelet_level=2,
            wavelet_type='haar'
        )
        
        # Wavelet attention (for 8K)
        model = build_fusion_net_vfi(
            base_channels=32,  # Reduce channels for memory
            vfi_core=vfi_core,
            use_wavelet_attention=True,
            wavelet_level=2,   # Can also use level=3 for extreme cases
            wavelet_type='haar'
        )
    """
    return AnchorFusionNetVFI(
        base_channels=base_channels,
        vfi_core=vfi_core,
        vfi_down_scale=vfi_down_scale,
        vfi_local=vfi_local,
        freeze_vfi=freeze_vfi,
        use_wavelet_attention=use_wavelet_attention,
        wavelet_level=wavelet_level,
        wavelet_type=wavelet_type
    )