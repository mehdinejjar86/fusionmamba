#!/usr/bin/env python3
"""
Enhanced smoke test for comparing Pyramid vs Sliding Window attention modes
Supports various configurations and performance benchmarking
"""

import argparse
import time
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Dict, Any, Tuple

# Your imports
from model.vfimamba.feature_extractor import feature_extractor as mamba_extractor
from model.vfimamba.flow_estimation import MultiScaleFlow
from fusionmamba import build_fusion_net_vfi
from config import init_model_config
from model.loss.fusionloss import AnchorFusionLoss


# ============ UTILITIES ============

@contextmanager
def timer(name: str):
    """Context manager for timing code blocks"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"⏱️  {name}: {elapsed:.3f}s")


def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0


def print_model_info(model, verbose=False):
    """Print model architecture information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters:     {total_params/1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")
    print(f"  Frozen parameters:    {(total_params-trainable_params)/1e6:.2f}M")
    
    if verbose:
        print("\n  Layer breakdown:")
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            print(f"    {name:20s}: {params/1e6:.2f}M")


def test_gradient_flow(model, loss_dict, verbose=False):
    """Check gradient flow through model"""
    print("\n🔍 Gradient Flow Analysis:")
    
    grad_stats = {}
    zero_grads = []
    large_grads = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats[name] = grad_norm
                
                if grad_norm == 0:
                    zero_grads.append(name)
                elif grad_norm > 100:
                    large_grads.append((name, grad_norm))
    
    # Statistics
    if grad_stats:
        grad_values = list(grad_stats.values())
        print(f"  ✓ Gradients flowing: {len(grad_values)} params")
        print(f"  📈 Mean grad norm: {sum(grad_values)/len(grad_values):.6f}")
        print(f"  📊 Max grad norm:  {max(grad_values):.6f}")
        print(f"  📉 Min grad norm:  {min(grad_values):.6f}")
        
        if zero_grads:
            print(f"  ⚠️  {len(zero_grads)} params with ZERO gradients")
            if verbose:
                for name in zero_grads[:5]:  # Show first 5
                    print(f"      - {name}")
        
        if large_grads:
            print(f"  ⚠️  {len(large_grads)} params with LARGE gradients (>100)")
            if verbose:
                for name, norm in large_grads[:5]:
                    print(f"      - {name}: {norm:.2f}")
    
    return grad_stats


def validate_outputs(pred: torch.Tensor, aux: Dict[str, Any], args):
    """Validate model outputs"""
    print("\n✅ Output Validation:")
    
    # Check prediction
    print(f"  Prediction shape: {tuple(pred.shape)}")
    print(f"  Prediction range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
    
    if torch.isnan(pred).any():
        print("  ❌ NaN values detected in prediction!")
        return False
    if torch.isinf(pred).any():
        print("  ❌ Inf values detected in prediction!")
        return False
    
    # Check auxiliary outputs
    if 't_weights' in aux:
        weights = aux['t_weights']
        print(f"  Temporal weights sum: {weights.sum(dim=1).mean().item():.3f}")
        if args.num_anchors > 1:
            print(f"  Weight distribution: {weights[0].tolist()}")
    
    if 'flows' in aux:
        flows = aux['flows']
        flow_mag = flows[:, :, :2].norm(dim=2).mean()
        print(f"  Mean flow magnitude: {flow_mag.item():.3f}")
    
    print("  ✓ All outputs valid")
    return True


# ============ MAIN TEST FUNCTIONS ============

def build_models(args):
    """Build VFIMamba and Fusion models"""
    print(f"\n🔨 Building Models (window_mode={args.window_mode})...")
    
    # VFIMamba backbone
    backbone_cfg, multiscale_cfg = init_model_config(
        F=args.vfi_channels, 
        depth=[2, 2, 2, 3, 3], 
        M=False
    )
    vfi_core = MultiScaleFlow(
        mamba_extractor(**backbone_cfg), 
        **multiscale_cfg
    ).to(args.device)
    
    if args.freeze_vfi:
        vfi_core.eval()
        for param in vfi_core.parameters():
            param.requires_grad = False
    
    # Fusion model
    fusion = build_fusion_net_vfi(
        base_channels=args.base_channels,
        vfi_core=vfi_core,
        vfi_down_scale=args.vfi_down_scale,
        vfi_local=args.vfi_local,
        window_mode=args.window_mode,
        freeze_vfi=args.freeze_vfi
    ).to(args.device)
    
    return vfi_core, fusion


def test_forward_pass(fusion, args):
    """Test forward pass with timing"""
    print(f"\n🚀 Testing Forward Pass (B={args.batch_size}, N={args.num_anchors}, {args.height}x{args.width})...")
    
    # Create inputs
    I0 = torch.rand(args.batch_size, args.num_anchors, 3, args.height, args.width, device=args.device)
    I1 = torch.rand(args.batch_size, args.num_anchors, 3, args.height, args.width, device=args.device)
    t = torch.rand(args.batch_size, args.num_anchors, device=args.device)
    
    # Warmup
    if args.warmup:
        print("  Warming up...")
        with torch.no_grad():
            for _ in range(3):
                _ = fusion(I0, I1, t)
    
    # Timed forward pass
    with timer("Forward pass"):
        pred, aux = fusion(I0, I1, t)
    
    # Memory usage
    alloc, reserved = get_memory_usage()
    print(f"  💾 GPU Memory: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # FPS estimate
    if args.benchmark:
        print("  Benchmarking...")
        times = []
        with torch.no_grad():
            for _ in range(args.benchmark_iters):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = fusion(I0, I1, t)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        print(f"  ⚡ Average time: {avg_time:.3f}s ({fps:.1f} FPS)")
    
    return pred, aux, I0, I1, t


def test_loss_and_backward(fusion, pred, aux, target, loss_fn, args):
    """Test loss computation and backward pass"""
    print("\n📉 Testing Loss and Backward Pass...")
    
    # Compute losses
    losses = loss_fn(pred, target, aux)
    
    # Print loss components
    print("  Loss components:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            val = value.item()
            status = "✓"
            if torch.isnan(value):
                status = "❌ NaN"
            elif torch.isinf(value):
                status = "❌ Inf"
            print(f"    {key:20s}: {val:10.6f} {status}")
    
    total_loss = losses['total']
    
    # Backward pass with timing
    optimizer = torch.optim.Adam(fusion.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()
    
    with timer("Backward pass"):
        total_loss.backward()
    
    # Gradient analysis
    grad_stats = test_gradient_flow(fusion, losses, verbose=args.verbose)
    
    # Optimizer step
    optimizer.step()
    print("  ✓ Optimizer step completed")
    
    return optimizer, total_loss.item()


def compare_modes(args):
    """Compare pyramid vs sliding window attention"""
    print("\n" + "="*80)
    print("🔬 COMPARING ATTENTION MODES")
    print("="*80)
    
    results = {}
    
    for window_mode in [False, True]:
        mode_name = "Sliding Window" if window_mode else "Original Pyramid"
        print(f"\n{'='*40}")
        print(f"Testing: {mode_name}")
        print(f"{'='*40}")
        
        args.window_mode = window_mode
        
        # Build model
        vfi_core, fusion = build_models(args)
        
        # Print model info
        if args.verbose:
            print_model_info(fusion, verbose=True)
        
        # Test forward
        pred, aux, I0, I1, t = test_forward_pass(fusion, args)
        target = torch.rand(args.batch_size, 3, args.height, args.width, device=args.device)
        
        # Validate outputs
        valid = validate_outputs(pred, aux, args)
        
        # Test loss and backward
        loss_fn = AnchorFusionLoss(use_gan=False).to(args.device)
        optimizer, loss_val = test_loss_and_backward(fusion, pred, aux, target, loss_fn, args)
        
        # Benchmark if requested
        if args.benchmark:
            print(f"\n⏱️  Detailed Timing for {mode_name}:")
            
            times = {'forward': [], 'backward': [], 'total': []}
            
            for i in range(args.benchmark_iters):
                optimizer.zero_grad()
                
                # Forward
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_start = time.perf_counter()
                pred, aux = fusion(I0, I1, t)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_forward = time.perf_counter() - t_start
                
                # Loss + Backward
                losses = loss_fn(pred, target, aux)
                t_start = time.perf_counter()
                losses['total'].backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_backward = time.perf_counter() - t_start
                
                times['forward'].append(t_forward)
                times['backward'].append(t_backward)
                times['total'].append(t_forward + t_backward)
            
            # Statistics
            for key in ['forward', 'backward', 'total']:
                avg = sum(times[key]) / len(times[key])
                fps = 1.0 / avg if key != 'backward' else 0
                print(f"    {key:10s}: {avg:.3f}s" + (f" ({fps:.1f} FPS)" if fps > 0 else ""))
            
            # Memory
            alloc, _ = get_memory_usage()
            results[mode_name] = {
                'forward_time': sum(times['forward']) / len(times['forward']),
                'backward_time': sum(times['backward']) / len(times['backward']),
                'memory_gb': alloc,
                'loss': loss_val
            }
    
    # Comparison summary
    if len(results) == 2:
        print("\n" + "="*80)
        print("📊 COMPARISON SUMMARY")
        print("="*80)
        
        orig = results.get("Original Pyramid", {})
        swin = results.get("Sliding Window", {})
        
        if orig and swin:
            speedup_fwd = orig['forward_time'] / swin['forward_time'] if swin['forward_time'] > 0 else 0
            speedup_bwd = orig['backward_time'] / swin['backward_time'] if swin['backward_time'] > 0 else 0
            mem_reduction = (orig['memory_gb'] - swin['memory_gb']) / orig['memory_gb'] * 100 if orig['memory_gb'] > 0 else 0
            
            print(f"\n🚀 Performance Gains (Sliding Window vs Original):")
            print(f"  Forward speedup:  {speedup_fwd:.2f}x")
            print(f"  Backward speedup: {speedup_bwd:.2f}x")
            print(f"  Memory reduction: {mem_reduction:.1f}%")
            print(f"  Loss difference:  {abs(orig['loss'] - swin['loss']):.6f}")


def test_with_gan(args):
    """Test with GAN discriminator"""
    print("\n" + "="*80)
    print("🎮 TESTING WITH GAN DISCRIMINATOR")
    print("="*80)
    
    # Build models
    vfi_core, fusion = build_models(args)
    
    # Loss with GAN
    loss_fn = AnchorFusionLoss(use_gan=True).to(args.device)
    
    # Optimizers
    optimizer_G = torch.optim.Adam(fusion.parameters(), lr=args.learning_rate)
    optimizer_D = torch.optim.Adam(loss_fn.discriminator.parameters(), lr=args.learning_rate * 2)
    
    # Data
    I0 = torch.rand(args.batch_size, args.num_anchors, 3, args.height, args.width, device=args.device)
    I1 = torch.rand(args.batch_size, args.num_anchors, 3, args.height, args.width, device=args.device)
    t = torch.rand(args.batch_size, args.num_anchors, device=args.device)
    target = torch.rand(args.batch_size, 3, args.height, args.width, device=args.device)
    
    print("\n--- Generator Step ---")
    with timer("Generator"):
        optimizer_G.zero_grad()
        pred, aux = fusion(I0, I1, t)
        pred_detached = pred.detach()
        losses_G = loss_fn(pred, target, aux)
        losses_G['total'].backward()
        optimizer_G.step()
    
    print(f"  ✓ Generator loss: {losses_G['total'].item():.6f}")
    
    print("\n--- Discriminator Step ---")
    with timer("Discriminator"):
        optimizer_D.zero_grad()
        
        # Real
        real_pred = loss_fn.discriminator(target)
        if isinstance(real_pred, (list, tuple)):
            d_real_loss = sum(F.softplus(-rp).mean() for rp in real_pred) / len(real_pred)
        else:
            d_real_loss = F.softplus(-real_pred).mean()
        
        # Fake
        fake_pred = loss_fn.discriminator(pred_detached)
        if isinstance(fake_pred, (list, tuple)):
            d_fake_loss = sum(F.softplus(fp).mean() for fp in fake_pred) / len(fake_pred)
        else:
            d_fake_loss = F.softplus(fake_pred).mean()
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()
    
    print(f"  ✓ Discriminator loss: {d_loss.item():.6f}")
    print(f"    Real: {d_real_loss.item():.6f}, Fake: {d_fake_loss.item():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Smoke test for Fusion model with VFIMamba")
    
    # Model configuration
    parser.add_argument('--window-mode', action='store_true', 
                       help='Use sliding window attention instead of original pyramid')
    parser.add_argument('--base-channels', type=int, default=48,
                       help='Base channels for fusion model (default: 48)')
    parser.add_argument('--vfi-channels', type=int, default=16,
                       help='VFIMamba feature channels (default: 16)')
    parser.add_argument('--vfi-down-scale', type=float, default=1.0,
                       help='VFIMamba downscaling factor (default: 1.0)')
    parser.add_argument('--vfi-local', action='store_true',
                       help='Use local mode for VFIMamba')
    parser.add_argument('--freeze-vfi', action='store_true', default=True,
                       help='Freeze VFIMamba weights (default: True)')
    
    # Data configuration
    parser.add_argument('-B', '--batch-size', type=int, default=1,
                       help='Batch size (default: 1)')
    parser.add_argument('-N', '--num-anchors', type=int, default=3,
                       help='Number of temporal anchors (default: 3)')
    parser.add_argument('-H', '--height', type=int, default=256,
                       help='Image height (default: 256)')
    parser.add_argument('-W', '--width', type=int, default=256,
                       help='Image width (default: 256)')
    
    # Test configuration
    parser.add_argument('--compare', action='store_true',
                       help='Compare pyramid vs sliding window modes')
    parser.add_argument('--gan', action='store_true',
                       help='Test with GAN discriminator')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarking')
    parser.add_argument('--benchmark-iters', type=int, default=10,
                       help='Number of benchmark iterations (default: 10)')
    parser.add_argument('--warmup', action='store_true',
                       help='Perform warmup iterations before timing')
    
    # Training configuration
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    
    # Misc
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Print configuration
    print("="*80)
    print("🔧 CONFIGURATION")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Model: window_mode={args.window_mode}, base_channels={args.base_channels}")
    print(f"Data:  B={args.batch_size}, N={args.num_anchors}, {args.height}x{args.width}")
    print(f"VFI:   channels={args.vfi_channels}, frozen={args.freeze_vfi}")
    
    # Run tests
    if args.compare:
        compare_modes(args)
    else:
        # Single mode test
        print("\n" + "="*80)
        print(f"🧪 SMOKE TEST ({'Sliding Window' if args.window_mode else 'Original Pyramid'})")
        print("="*80)
        
        # Build models
        vfi_core, fusion = build_models(args)
        print_model_info(fusion, verbose=args.verbose)
        
        # Test forward
        pred, aux, I0, I1, t = test_forward_pass(fusion, args)
        target = torch.rand(args.batch_size, 3, args.height, args.width, device=args.device)
        
        # Validate
        valid = validate_outputs(pred, aux, args)
        if not valid:
            print("❌ Output validation failed!")
            return
        
        # Test backward
        loss_fn = AnchorFusionLoss(use_gan=False).to(args.device)
        optimizer, loss = test_loss_and_backward(fusion, pred, aux, target, loss_fn, args)
        
        print("\n" + "="*80)
        print("✅ SMOKE TEST PASSED")
        print("="*80)
    
    # Optional GAN test
    if args.gan:
        test_with_gan(args)
        print("\n✅ GAN TEST PASSED")
    
    print("\n🎉 All tests completed successfully!")


if __name__ == "__main__":
    main()