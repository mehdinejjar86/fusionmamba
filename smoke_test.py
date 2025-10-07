# build_vfi_and_fusion.py (extended smoke test with wavelet)
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


def get_memory_stats():
    """Get current GPU memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
        return allocated, reserved
    return 0, 0


def smoke_test_full(B=1, N=3, H=256, W=256, use_wavelet=False, wavelet_level=2):
    """Complete smoke test: forward + loss + backward"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print(f"Device: {device}")
    print(f"Attention Mode: {'WAVELET' if use_wavelet else 'STANDARD'}")
    if use_wavelet:
        print(f"Wavelet Level: {wavelet_level} ({2**(2*wavelet_level)}x memory reduction)")
    print("="*60)
    
    # ============ 1. BUILD MODEL ============
    print("\n[1/5] Building VFIMamba and Fusion Model...")
    
    backbone_cfg, multiscale_cfg = init_model_config(F=16, depth=[2,2,2,3,3], M=False)
    vfi_core = MultiScaleFlow(mamba_extractor(**backbone_cfg), **multiscale_cfg).to(device)
    vfi_core.eval()  # Keep VFIMamba frozen
    
    # Build fusion model with wavelet option
    fusion = build_fusion_net_vfi(
        base_channels=48,
        vfi_core=vfi_core,
        vfi_down_scale=1.0,
        vfi_local=False,
        use_wavelet_attention=use_wavelet,
        wavelet_level=wavelet_level,
        wavelet_type='haar'
    ).to(device)
    fusion.train()  # Training mode for fusion
    
    total_params = sum(p.numel() for p in fusion.parameters())
    trainable_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    print(f"✓ Model built: {total_params/1e6:.2f}M params ({trainable_params/1e6:.2f}M trainable)")
    
    # ============ 2. BUILD LOSS ============
    print("\n[2/5] Building Loss Function...")
    
    loss_fn = AnchorFusionLoss(use_gan=False).to(device)  # Start without GAN
    print("✓ Loss function ready (GAN disabled for initial test)")
    
    # ============ 3. FORWARD PASS ============
    print("\n[3/5] Testing Forward Pass...")
    
    # Clear memory before test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    mem_before = get_memory_stats()
    
    I0 = torch.rand(B, N, 3, H, W, device=device)
    I1 = torch.rand(B, N, 3, H, W, device=device)
    t  = torch.rand(B, N, device=device)  # Random timesteps
    target = torch.rand(B, 3, H, W, device=device)  # Ground truth
    
    pred, aux = fusion(I0, I1, t)
    
    mem_after = get_memory_stats()
    
    print(f"✓ Forward pass successful")
    print(f"  Input:  {tuple(I0.shape)}")
    print(f"  Output: {tuple(pred.shape)}")
    print(f"  Target: {tuple(target.shape)}")
    print(f"  Temporal weights: {aux['t_weights'][0].tolist()}")
    
    if torch.cuda.is_available():
        mem_used = mem_after[0] - mem_before[0]
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  Memory used: {mem_used:.2f} GB (peak: {peak_mem:.2f} GB)")
    
    # Validate output
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    assert pred.min() >= 0 and pred.max() <= 1, f"Output range invalid: [{pred.min():.3f}, {pred.max():.3f}]"
    print(f"  Output range: [{pred.min():.3f}, {pred.max():.3f}] ✓")
    
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
        import traceback
        traceback.print_exc()
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
    print(f"Mode: {'WAVELET' if use_wavelet else 'STANDARD'} attention")
    print(f"Model ready for training with {trainable_params} trainable parameters")
    
    # Cleanup
    del fusion, loss_fn, optimizer, I0, I1, t, target, pred, pred2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return True, peak_mem if torch.cuda.is_available() else 0


def compare_attention_modes(B=1, N=2, resolutions=None):
    """Compare standard vs wavelet attention at different resolutions"""
    
    if resolutions is None:
        resolutions = [
            (512, 512, "512p"),
            (1080, 1920, "Full HD"),
            (2160, 3840, "4K"),
        ]
    
    print("\n" + "="*60)
    print("COMPARING ATTENTION MODES AT DIFFERENT RESOLUTIONS")
    print("="*60)
    
    results = []
    
    for H, W, name in resolutions:
        print(f"\n{'='*60}")
        print(f"Resolution: {name} ({H}×{W})")
        print(f"{'='*60}")
        
        # Test standard attention
        print(f"\n--- Testing STANDARD attention ---")
        try:
            success_std, mem_std = smoke_test_full(B, N, H, W, use_wavelet=False)
            if not success_std:
                print(f"⚠️  Standard attention failed at {name}")
                mem_std = float('inf')
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ Standard attention: OUT OF MEMORY at {name}")
                mem_std = float('inf')
                success_std = False
            else:
                raise
        
        # Test wavelet attention
        print(f"\n--- Testing WAVELET attention (level 2) ---")
        try:
            success_wav, mem_wav = smoke_test_full(B, N, H, W, use_wavelet=True, wavelet_level=2)
            if not success_wav:
                print(f"⚠️  Wavelet attention failed at {name}")
                mem_wav = float('inf')
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ Wavelet attention: OUT OF MEMORY at {name}")
                mem_wav = float('inf')
                success_wav = False
            else:
                raise
        
        # Store results
        results.append({
            'resolution': name,
            'size': (H, W),
            'standard_success': success_std,
            'wavelet_success': success_wav,
            'standard_memory': mem_std,
            'wavelet_memory': mem_wav
        })
        
        # Print comparison
        print(f"\n{'='*60}")
        print(f"COMPARISON FOR {name}")
        print(f"{'='*60}")
        
        if torch.cuda.is_available():
            if success_std and success_wav:
                savings = (1 - mem_wav / mem_std) * 100 if mem_std > 0 else 0
                print(f"Standard: {mem_std:.2f} GB")
                print(f"Wavelet:  {mem_wav:.2f} GB")
                print(f"Savings:  {savings:.1f}%")
            elif success_wav and not success_std:
                print(f"Standard: OOM")
                print(f"Wavelet:  {mem_wav:.2f} GB ✓")
            elif success_std and not success_wav:
                print(f"Standard: {mem_std:.2f} GB ✓")
                print(f"Wavelet:  FAILED")
            else:
                print(f"Both modes failed")
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Resolution':<15} {'Standard':<15} {'Wavelet':<15} {'Savings':<10}")
    print("-"*60)
    
    for r in results:
        std_str = f"{r['standard_memory']:.2f} GB" if r['standard_success'] else "OOM"
        wav_str = f"{r['wavelet_memory']:.2f} GB" if r['wavelet_success'] else "FAIL"
        
        if r['standard_success'] and r['wavelet_success']:
            savings = (1 - r['wavelet_memory'] / r['standard_memory']) * 100
            sav_str = f"{savings:.1f}%"
        else:
            sav_str = "N/A"
        
        print(f"{r['resolution']:<15} {std_str:<15} {wav_str:<15} {sav_str:<10}")
    
    return results


def test_with_gan(B=1, N=3, H=256, W=256, use_wavelet=False):
    """Extended test with GAN discriminator"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print(f"TESTING WITH GAN DISCRIMINATOR ({'WAVELET' if use_wavelet else 'STANDARD'})")
    print("="*60)
    
    # Build model
    backbone_cfg, multiscale_cfg = init_model_config(F=16, depth=[2,2,2,3,3], M=False)
    vfi_core = MultiScaleFlow(mamba_extractor(**backbone_cfg), **multiscale_cfg).to(device).eval()
    
    for param in vfi_core.parameters():
        param.requires_grad = False
    
    fusion = build_fusion_net_vfi(
        base_channels=48, 
        vfi_core=vfi_core,
        use_wavelet_attention=use_wavelet,
        wavelet_level=2,
        wavelet_type='haar'
    ).to(device).train()
    
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
    
    # Cleanup
    del fusion, loss_fn, optimizer_G, optimizer_D
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_wavelet_reconstruction():
    """Test that wavelet decomposition/reconstruction is lossless"""
    
    print("\n" + "="*60)
    print("TESTING WAVELET RECONSTRUCTION QUALITY")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        from pytorch_wavelets import DWTForward, DWTInverse
    except ImportError:
        print("⚠️  pytorch_wavelets not installed, skipping reconstruction test")
        return
    
    # Test different wavelet types
    wavelets = ['haar', 'db2', 'db4', 'sym2']
    H, W = 512, 512
    
    for wavelet in wavelets:
        print(f"\n--- Testing {wavelet} wavelet ---")
        
        dwt = DWTForward(J=2, wave=wavelet, mode='symmetric').to(device)
        idwt = DWTInverse(wave=wavelet, mode='symmetric').to(device)
        
        # Create test image
        img = torch.rand(1, 3, H, W, device=device)
        
        # Decompose
        yl, yh = dwt(img)
        print(f"  Low-freq: {yl.shape}")
        print(f"  High-freq levels: {len(yh)}")
        
        # Reconstruct
        reconstructed = idwt((yl, yh))
        
        # Crop to original size (wavelet may add padding)
        reconstructed = reconstructed[:, :, :H, :W]
        
        # Check error
        error = (img - reconstructed).abs().max().item()
        mse = F.mse_loss(img, reconstructed).item()
        
        print(f"  Max error: {error:.8f}")
        print(f"  MSE: {mse:.8f}")
        
        if error < 1e-5:
            print(f"  ✓ Perfect reconstruction")
        elif error < 1e-3:
            print(f"  ✓ Good reconstruction")
        else:
            print(f"  ⚠️  Reconstruction error may be too high")
    
    print("\n✓ Wavelet reconstruction test completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Smoke test for VFI Fusion')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_anchors', type=int, default=2)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--mode', type=str, default='both', 
                       choices=['standard', 'wavelet', 'both', 'compare'],
                       help='Which attention mode to test')
    parser.add_argument('--test_gan', action='store_true',
                       help='Also test GAN discriminator')
    parser.add_argument('--test_reconstruction', action='store_true',
                       help='Test wavelet reconstruction quality')
    
    args = parser.parse_args()
    
    B = args.batch_size
    N = args.num_anchors
    H = args.height
    W = args.width
    
    print("\n" + "="*60)
    print("VFI FUSION MODEL SMOKE TEST")
    print("="*60)
    print(f"Configuration:")
    print(f"  Batch size: {B}")
    print(f"  Num anchors: {N}")
    print(f"  Resolution: {H}×{W}")
    print(f"  Mode: {args.mode}")
    
    # Test wavelet reconstruction if requested
    if args.test_reconstruction:
        test_wavelet_reconstruction()
    
    # Main tests
    if args.mode == 'standard':
        success = smoke_test_full(B, N, H, W, use_wavelet=False)[0]
        
    elif args.mode == 'wavelet':
        success = smoke_test_full(B, N, H, W, use_wavelet=True, wavelet_level=2)[0]
        
    elif args.mode == 'both':
        print("\n" + "="*60)
        print("TESTING STANDARD ATTENTION")
        print("="*60)
        success_std = smoke_test_full(B, N, H, W, use_wavelet=False)[0]
        
        print("\n" + "="*60)
        print("TESTING WAVELET ATTENTION")
        print("="*60)
        success_wav = smoke_test_full(B, N, H, W, use_wavelet=True, wavelet_level=2)[0]
        
        success = success_std and success_wav
        
    elif args.mode == 'compare':
        # Compare at multiple resolutions
        results = compare_attention_modes(B, N)
        success = any(r['wavelet_success'] or r['standard_success'] for r in results)
    
    # GAN test if requested
    if args.test_gan and success:
        print("\n" + "="*60)
        response = input("Run GAN test? (y/n): ")
        if response.lower() == 'y':
            if args.mode == 'wavelet' or args.mode == 'both':
                test_with_gan(B, N, H, W, use_wavelet=True)
            else:
                test_with_gan(B, N, H, W, use_wavelet=False)
    
    # Final summary
    print("\n" + "="*60)
    if success:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ❌")
    print("="*60)