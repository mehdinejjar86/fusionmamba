# vfi_adapter.py
import torch
import torch.nn.functional as F

from model.vfimamba.warplayer import warp as warp_vfi

class VFIMambaAdapter(torch.nn.Module):
    """
    Thin wrapper that exposes the VFIMamba flow/mask as a simple callable.
    It mirrors `Model.hr_inference` / `MultiScaleFlow.calculate_flow` behavior.

    Args:
        vfi_core: an instance exposing:
          - feature_bone (backbone) inside
          - calculate_flow(imgs_down, timestep, local=bool) -> (flow, mask)
        down_scale: e.g., 1.0 (default), or 0.5 for memory
        local: whether to run local IFBlock refinement (True ≈ VFIMamba 'LOCAL' stages)
    """
    def __init__(self, vfi_core, down_scale: float = 1.0, local: bool = False):
        super().__init__()
        self.vfi = vfi_core
        self.down_scale = float(down_scale)
        self.local = bool(local)

    @torch.no_grad()
    def forward(self, I0b, I1b, tb):
        """
        I0b, I1b: [BN, 3, H, W], tb: [BN, 1] or [BN, 1, 1, 1] with values in [0,1]
        Returns flows_b [BN,4,H,W], masks_b [BN,1,H,W] at INPUT resolution.
        """
        BN, _, H, W = I0b.shape
        # Build the VFIMamba input: imgs = concat([I0, I1], dim=1)
        imgs = torch.cat([I0b, I1b], dim=1)  # [BN, 6, H, W]

        # Downscale if requested (exactly like hr_inference)
        if self.down_scale != 1.0:
            imgs_down = F.interpolate(imgs, scale_factor=self.down_scale,
                                      mode="bilinear", align_corners=False)
            # timestep broadcasting: VFIMamba expects scalar or broadcastable tensor
            # we pass [BN,1,1,1] so each anchor can have its own τ
            if tb.dim() == 2:
                tb_ = tb.view(BN, 1, 1, 1)
            else:
                tb_ = tb
            flow, mask = self.vfi.calculate_flow(imgs_down, tb_, local=self.local)
            # Rescale back to input size (pixel units!) just like hr_inference
            flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
            flow[:, 0:1] *= 1.0 / self.down_scale  # x
            flow[:, 1:2] *= 1.0 / self.down_scale  # y
            flow[:, 2:3] *= 1.0 / self.down_scale
            flow[:, 3:4] *= 1.0 / self.down_scale
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
        else:
            if tb.dim() == 2:
                tb_ = tb.view(BN, 1, 1, 1)
            else:
                tb_ = tb
            flow, mask = self.vfi.calculate_flow(imgs, tb_, local=self.local)

        # Return full-res flow/mask
        return flow, torch.sigmoid(mask)

    @staticmethod
    def warp_like_vfi(img, flow):
        return warp_vfi(img, flow)
