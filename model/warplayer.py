import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
backwarp_tenGrid = {}

def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    
    # padding mode on mps is zeros 
    padding_mode = 'zeros' if device.type == 'mps' else 'border'
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode=padding_mode, align_corners=True)
  
  
class FlowWarping(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = {}

    def forward(self, img, flow):
        # img: [B,C,H,W]; flow: [B,2,H,W] in pixels (x,y)
        device = flow.device
        B, _, H, W = flow.shape
        k = (str(device), H, W)
        if k not in self.cache:
            xs = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, -1, H, -1)
            ys = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, -1, -1, W)
            self.cache[k] = torch.cat([xs, ys], 1)
        grid = self.cache[k]
        flow_norm = torch.cat([flow[:, 0:1] / ((W - 1.0) / 2.0),
                               flow[:, 1:2] / ((H - 1.0) / 2.0)], 1)
        g = (grid + flow_norm).permute(0, 2, 3, 1)
        padding_mode = 'zeros' if device.type == 'mps' else 'border'
        return F.grid_sample(img, g, mode='bilinear', padding_mode=padding_mode, align_corners=False)