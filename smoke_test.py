# build_vfi_and_fusion.py (extended smoke test)
import torch
import torch.nn.functional as F
from model.vfimamba.feature_extractor import feature_extractor as mamba_extractor
from model.vfimamba.flow_estimation import MultiScaleFlow
from fusionmamba import build_fusion_net_vfi
from config import init_model_config
from model.loss.fusionloss import AnchorFusionLoss


def test_gradient_flow(model, loss_dict):
    """Check if gradients are flowing through the model"""
    print("\n=== Gradient Flow Check ===")
    
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_stats[name] = grad_norm
            
            # Flag suspicious gradients
            if grad_norm == 0:
                print(f"⚠️  ZERO gradient: {name}")
            elif grad_norm > 100:
                print(f"⚠️  LARGE gradient: {name} = {grad_norm:.2f}")
    
    # Summary statistics
    grad_values = list(grad_stats.values())
    if grad_values:
        print(f"✓ Gradients flowing: {len(grad_values)}/{len(list(model.parameters()))} params")
        print(f"  Mean grad norm: {sum(grad_values)/len(grad_values):.6f}")
        print(f"  Max grad norm:  {max(grad_values):.6f}")
        print(f"  Min grad norm:  {min(grad_values):.6f}")
    else:
        print("❌ NO GRADIENTS FOUND!")
    
    return grad_stats


def test_loss_components(losses):
    """Validate individual loss components"""
    print("\n=== Loss Components ===")
    
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            val = value.item()
            print(f"{key:20s}: {val:.6f}", end="")
            
            # Sanity checks
            if torch.isnan(value):
                print(" ❌ NaN!")
            elif torch.isinf(value):
                print(" ❌ Inf!")
            elif val < 0 and key != 'weight_entropy':  # entropy can be negative
                print(" ⚠️  Negative!")
            else:
                print(" ✓")


def smoke_test_full(B=1, N=3, H=256, W=256):
    """Complete smoke test: forward + loss + backward"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print(f"Device: {device}")
    print("="*60)
    
    # ============ 1. BUILD MODEL ============
    print("\n[1/5] Building VFIMamba and Fusion Model...")
    
    backbone_cfg, multiscale_cfg = init_model_config(F=16, depth=[2,2,2,3,3], M=False)
    vfi_core = MultiScaleFlow(mamba_extractor(**backbone_cfg), **multiscale_cfg).to(device)
    vfi_core.eval()  # Keep VFIMamba frozen
    
    # Build fusion model
    fusion = build_fusion_net_vfi(
        base_channels=48,
        vfi_core=vfi_core,
        vfi_down_scale=1.0,
        vfi_local=False
    ).to(device)
    fusion.train()  # Training mode for fusion
    
    print(f"✓ Model built: {sum(p.numel() for p in fusion.parameters())/1e6:.2f}M params")
    
    # ============ 2. BUILD LOSS ============
    print("\n[2/5] Building Loss Function...")
    
    loss_fn = AnchorFusionLoss(use_gan=False).to(device)  # Start without GAN
    print("✓ Loss function ready (GAN disabled for initial test)")
    
    # ============ 3. FORWARD PASS ============
    print("\n[3/5] Testing Forward Pass...")
    
    I0 = torch.rand(B, N, 3, H, W, device=device)
    I1 = torch.rand(B, N, 3, H, W, device=device)
    t  = torch.rand(B, N, device=device)  # Random timesteps
    target = torch.rand(B, 3, H, W, device=device)  # Ground truth
    
    pred, aux = fusion(I0, I1, t)
    
    print(f"✓ Forward pass successful")
    print(f"  Input:  {tuple(I0.shape)}")
    print(f"  Output: {tuple(pred.shape)}")
    print(f"  Target: {tuple(target.shape)}")
    print(f"  Temporal weights: {aux['t_weights'][0].tolist()}")
    
    # ============ 4. LOSS COMPUTATION ============
    print("\n[4/5] Testing Loss Computation...")
    
    losses = loss_fn(pred, target, aux)
    test_loss_components(losses)
    
    total_loss = losses['total']
    print(f"\n✓ Total Loss: {total_loss.item():.6f}")
    
    # ============ 5. BACKWARD PASS ============
    print("\n[5/5] Testing Backward Pass...")
    
    # Create optimizer
    optimizer = torch.optim.Adam(fusion.parameters(), lr=1e-4)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Backward
    try:
        total_loss.backward()
        print("✓ Backward pass successful")
    except Exception as e:
        print(f"❌ Backward failed: {e}")
        return False
    
    # Check gradient flow
    grad_stats = test_gradient_flow(fusion, losses)
    
    # Optimizer step
    optimizer.step()
    print("✓ Optimizer step successful")
    
    # ============ 6. SECOND ITERATION TEST ============
    print("\n=== Second Iteration (validate no errors) ===")
    
    optimizer.zero_grad()
    pred2, aux2 = fusion(I0, I1, t)
    losses2 = loss_fn(pred2, target, aux2)
    losses2['total'].backward()
    optimizer.step()
    
    print(f"✓ Second iteration successful")
    print(f"  Loss changed: {losses['total'].item():.6f} → {losses2['total'].item():.6f}")
    
    # ============ SUMMARY ============
    print("\n" + "="*60)
    print("SMOKE TEST PASSED ✓")
    print("="*60)
    print(f"Model ready for training with {len(list(fusion.parameters()))} parameters")
    
    return True


def test_with_gan(B=1, N=3, H=256, W=256):
    """Extended test with GAN discriminator"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print("TESTING WITH GAN DISCRIMINATOR")
    print("="*60)
    
    # Build model (simplified)
    backbone_cfg, multiscale_cfg = init_model_config(F=16, depth=[2,2,2,3,3], M=False)
    vfi_core = MultiScaleFlow(mamba_extractor(**backbone_cfg), **multiscale_cfg).to(device).train()
    
    fusion = build_fusion_net_vfi(base_channels=48, vfi_core=vfi_core).to(device).train()
    
    # Loss with GAN
    loss_fn = AnchorFusionLoss(use_gan=True).to(device)
    
    # Optimizers
    optimizer_G = torch.optim.Adam(fusion.parameters(), lr=1e-4)
    optimizer_D = torch.optim.Adam(loss_fn.discriminator.parameters(), lr=1e-4)
    
    # Data
    I0 = torch.rand(B, N, 3, H, W, device=device)
    I1 = torch.rand(B, N, 3, H, W, device=device)
    t = torch.rand(B, N, device=device)
    target = torch.rand(B, 3, H, W, device=device)
    
    print("\n--- Generator Step ---")
    optimizer_G.zero_grad()
    pred, aux = fusion(I0, I1, t)
    pred_detached = pred.detach().clone()  # ← Save for discriminator step
    losses_G = loss_fn(pred, target, aux)
    losses_G['total'].backward()
    optimizer_G.step()
    print(f"✓ Generator loss: {losses_G['total'].item():.6f}")
    print(f"  (includes gan_g: {losses_G.get('gan_g', torch.tensor(0.0)).item():.6f})")
    
    print("\n--- Discriminator Step ---")
    optimizer_D.zero_grad()
    
    # Real
    real_pred = loss_fn.discriminator(target)
    if isinstance(real_pred, (list, tuple)):
        d_real_loss = sum(F.softplus(-rp).mean() for rp in real_pred) / len(real_pred)
    else:
        d_real_loss = F.softplus(-real_pred).mean()
    
    # Fake (use saved detached prediction)
    fake_pred = loss_fn.discriminator(pred_detached)
    if isinstance(fake_pred, (list, tuple)):
        d_fake_loss = sum(F.softplus(fp).mean() for fp in fake_pred) / len(fake_pred)
    else:
        d_fake_loss = F.softplus(fake_pred).mean()
    
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    optimizer_D.step()
    
    print(f"✓ Discriminator loss: {d_loss.item():.6f}")
    print(f"  Real: {d_real_loss.item():.6f}, Fake: {d_fake_loss.item():.6f}")
    
    print("\n✓ GAN TEST PASSED")


if __name__ == "__main__":
    B = 1
    N = 2  
    H = 3072 
    W = 3072
    # Basic test
    success = smoke_test_full(B, N, H, W)
    
    # Optional: Test with GAN
    if success:
        print("\n" + "="*60)
        response = input("Run GAN test? (y/n): ")
        if response.lower() == 'y':
            test_with_gan(B, N, H, W)