# build_vfi_and_fusion.py (example usage)
import torch
from model.feature_extractor import feature_extractor as mamba_extractor
from model.flow_estimation import MultiScaleFlow
from fusion_mamba import build_fusion_net_vfi
from config import init_model_config  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# 1) init VFIMamba core and load weights
backbone_cfg, multiscale_cfg = init_model_config(F=16, depth=[2,2,2,3,3], M=False)
vfi_core = MultiScaleFlow(mamba_extractor(**backbone_cfg), **multiscale_cfg).to(device).eval()
# vfi_core.load_state_dict(torch.load(".../model.pkl"), strict=True)  # your weights

# 2) build fusion
B, N, H, W = 1, 2, 1024, 1024
fusion = build_fusion_net_vfi(
    base_channels=48,
    vfi_core=vfi_core,     # <-- real VFIMamba
    vfi_down_scale=1.0,    # or 0.5 for memory (flows are rescaled back)
    vfi_local=False        # True enables local IFBlock refinement in VFIMamba
).to(device).eval()

# 3) run
I0 = torch.rand(B, N, 3, H, W, device=device)
I1 = torch.rand(B, N, 3, H, W, device=device)
t  = torch.full((B, N), 0.5, device=device)
with torch.no_grad():
    Y, aux = fusion(I0, I1, t)
print("Output:", tuple(Y.shape), "t-weights:", aux['t_weights'][0].tolist())
