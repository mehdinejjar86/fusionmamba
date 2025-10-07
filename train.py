# train_fusion.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from datetime import datetime
import warnings
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
from torch.amp import autocast, GradScaler
from torch.amp.autocast_mode import is_autocast_available

# Weights & Biases support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# Your dataset
from data.fusionloader import create_fusion_dataloader

# Your model
from fusionmamba import build_fusion_net_vfi
from model.vfimamba.feature_extractor import feature_extractor as mamba_extractor
from model.vfimamba.flow_estimation import MultiScaleFlow
from config import init_model_config

# Your loss
from model.loss.fusionloss import AnchorFusionLoss


# ----------------------------
# Metrics
# ----------------------------

class MetricsCalculator:
    def __init__(self, eval_size=(256, 256)):
        self.eval_size = eval_size

    def calculate_psnr(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        if pred.ndim == 4:
            pred = pred.transpose(0, 2, 3, 1)
            target = target.transpose(0, 2, 3, 1)
        elif pred.ndim == 3 and pred.shape[0] == 3:
            pred = pred.transpose(1, 2, 0)
            target = target.transpose(1, 2, 0)

        pred = np.clip(pred, 0, 1)
        target = np.clip(target, 0, 1)

        if pred.ndim == 4:
            vals = []
            for i in range(pred.shape[0]):
                vals.append(psnr_func(target[i], pred[i], data_range=1.0))
            return np.mean(vals)
        else:
            return psnr_func(target, pred, data_range=1.0)

    def calculate_ssim(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        if pred.ndim == 4:
            pred = pred.transpose(0, 2, 3, 1)
            target = target.transpose(0, 2, 3, 1)
        elif pred.ndim == 3 and pred.shape[0] == 3:
            pred = pred.transpose(1, 2, 0)
            target = target.transpose(1, 2, 0)

        pred = np.clip(pred, 0, 1)
        target = np.clip(target, 0, 1)

        if pred.ndim == 4:
            vals = []
            for i in range(pred.shape[0]):
                vals.append(ssim_func(target[i], pred[i], data_range=1.0, 
                                     channel_axis=2 if pred.shape[-1] == 3 else None))
            return np.mean(vals)
        else:
            return ssim_func(target, pred, data_range=1.0, 
                           channel_axis=2 if pred.shape[-1] == 3 else None)

    def resize_for_metrics(self, tensor, size=None):
        size = size or self.eval_size
        if tensor.shape[-2:] == size:
            return tensor
        out = torch.nn.functional.interpolate(
            tensor if tensor.dim() == 4 else tensor.unsqueeze(0),
            size=size,
            mode='bilinear',
            align_corners=False
        )
        return out if tensor.dim() == 4 else out.squeeze(0)


# ----------------------------
# Trainer
# ----------------------------

class FusionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self._setup_mixed_precision()

        # Run dirs
        self.run_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.run_name = f"{config.run_name}_{self.run_timestamp}" if getattr(config, 'run_name', None) else self.run_timestamp

        self.config.checkpoint_dir = Path("runs") / self.run_name / Path(self.config.checkpoint_dir)
        self.config.log_dir = Path("runs") / self.run_name / Path(self.config.log_dir)
        self.config.sample_dir = Path("runs") / self.run_name / Path(self.config.sample_dir)
        self.setup_directories()

        # W&B
        self.use_wandb = WANDB_AVAILABLE and getattr(config, 'use_wandb', False)
        if self.use_wandb:
            self._init_wandb()

        # VFIMamba core (frozen initially)
        self.vfi_core = self._build_vfi_core()

        # Data
        self.train_loader, self.train_dataset = self._create_dataloader()
        self.val_loader, self.val_dataset = self._create_validation_dataloader()

        # Model
        self.model = self._create_model()

        # Loss
        self.loss_fn = AnchorFusionLoss(use_gan=False).to(self.device)  # Start without GAN

        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Metrics
        metric_size = tuple(getattr(config, 'metric_size', (256, 256)))
        self.metrics_calc = MetricsCalculator(eval_size=metric_size)

        # Logging
        self.writer = SummaryWriter(self.config.log_dir)
        self.start_epoch = 0
        self.best_psnr = 0
        self.best_ssim = 0
        self.best_val_psnr = 0
        self.best_val_ssim = 0
        self.epoch_metrics = []

        # Resume
        if config.resume:
            self.load_checkpoint(config.resume)

        print("\n" + "="*60)
        print("VFI Fusion Training Configuration")
        print("="*60 + "\n")

    def _init_wandb(self):
        wandb_config = vars(self.config).copy()
        wandb_config['run_timestamp'] = self.run_timestamp
        wandb_config['num_gpus'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        wandb_config['model_type'] = 'AnchorFusionNetVFI'
        wandb.init(
            project=getattr(self.config, 'wandb_project', 'vfi-fusion-model'),
            entity=getattr(self.config, 'wandb_entity', None),
            name=self.run_name,
            config=wandb_config,
            resume=self.config.resume is not None
        )
        if getattr(self.config, 'wandb_watch_model', False):
            wandb.watch(self.model, log='all', log_freq=100)
        print(f"W&B initialized: {wandb.run.url}")

    def _setup_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            ng = torch.cuda.device_count()
            print(f"Using {ng} CUDA GPU(s)")
            for i in range(ng):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        self.precision = getattr(self.config, 'precision', 'fp32')
        self.use_amp = False
        self.scaler = None
        self.amp_dtype = torch.float32
        
        self.autocast_device = self.device.type if self.device.type in ['cuda', 'cpu', 'mps'] else 'cpu'
        autocast_available = is_autocast_available(self.autocast_device)
        
        if self.precision != 'fp32':
            if not autocast_available:
                print(f"Warning: Autocast not available for {self.device.type}. Falling back to FP32.")
                self.precision = 'fp32'
            else:
                if self.device.type == 'cuda':
                    if self.precision == 'bf16':
                        if torch.cuda.is_bf16_supported():
                            self.use_amp = True
                            self.amp_dtype = torch.bfloat16
                        else:
                            print("Warning: BF16 not supported. Falling back to FP32.")
                            self.precision = 'fp32'
                    elif self.precision == 'fp16':
                        self.use_amp = True
                        self.amp_dtype = torch.float16
                        self.scaler = GradScaler('cuda')
                elif self.device.type == 'cpu':
                    if self.precision == 'bf16':
                        self.use_amp = True
                        self.amp_dtype = torch.bfloat16
                    elif self.precision == 'fp16':
                        print("Note: FP16 on CPU may be slower than BF16")
                        self.precision = 'bf16'
                        self.use_amp = True
                        self.amp_dtype = torch.bfloat16
        
        print(f"\nPrecision: {self.precision.upper()}")
        if self.use_amp:
            print(f"  AMP dtype: {self.amp_dtype}")
            print(f"  GradScaler: {'enabled' if self.scaler else 'not needed'}")

    def setup_directories(self):
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.log_dir = Path(self.config.log_dir)
        self.sample_dir = Path(self.config.sample_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRun directory: {self.run_name}")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Logs: {self.log_dir}")
        print(f"  Samples: {self.sample_dir}")

        config_dict = vars(self.config)
        config_dict['run_timestamp'] = self.run_timestamp
        config_dict['run_name'] = self.run_name

        with open(self.checkpoint_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)

    def _build_vfi_core(self):
        """Build VFIMamba core"""
        print("\nBuilding VFIMamba core...")
        backbone_cfg, multiscale_cfg = init_model_config(
            F=getattr(self.config, 'vfi_F', 16),
            depth=getattr(self.config, 'vfi_depth', [2,2,2,3,3]),
            M=False
        )
        vfi_core = MultiScaleFlow(
            mamba_extractor(**backbone_cfg), 
            **multiscale_cfg
        ).to(self.device)
        
        # Load pretrained weights if provided
        if hasattr(self.config, 'vfi_weights') and self.config.vfi_weights:
            print(f"Loading VFIMamba weights from: {self.config.vfi_weights}")
            vfi_core.load_state_dict(torch.load(self.config.vfi_weights), strict=True)
        
        # Freeze VFIMamba initially
        for param in vfi_core.parameters():
            param.requires_grad = False
        vfi_core.eval()
        
        print("VFIMamba core: FROZEN (Phase 1)")
        return vfi_core

    def _create_dataloader(self):
        gt_paths = self.config.gt_paths
        steps = self.config.steps
        if isinstance(steps, int):
            steps = [steps] * len(gt_paths)

        mix_strategy = getattr(self.config, 'mix_strategy', 'uniform')
        path_weights = getattr(self.config, 'path_weights', None)

        num_workers = self.config.num_workers
        if self.device.type == 'mps' and num_workers > 0:
            print("Warning: MPS device detected. Setting num_workers=0")
            num_workers = 0

        train_dataloader, train_dataset = create_fusion_dataloader(
            gt_paths=gt_paths,
            steps=steps,
            num_anchors=self.config.num_anchors,
            scale=self.config.scale,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            mix_strategy=mix_strategy,
            path_weights=path_weights,
            divisor=getattr(self.config, 'divisor', 32)
        )

        print("\nTraining Dataset:")
        print(f"  Total samples: {len(train_dataset)}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Batches per epoch: {len(train_dataloader)}")
        return train_dataloader, train_dataset

    def _create_validation_dataloader(self):
        val_paths = getattr(self.config, 'val_paths', None)
        val_steps = getattr(self.config, 'val_steps', None)
        val_split = getattr(self.config, 'val_split', 0.1)

        if val_paths is not None:
            if isinstance(val_steps, int):
                val_steps = [val_steps] * len(val_paths)
            elif val_steps is None:
                val_steps = self.config.steps

            num_workers = self.config.num_workers
            if self.device.type == 'mps' and num_workers > 0:
                num_workers = 0

            val_dataloader, val_dataset = create_fusion_dataloader(
                gt_paths=val_paths,
                steps=val_steps,
                num_anchors=self.config.num_anchors,
                scale=self.config.scale,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                divisor=getattr(self.config, 'divisor', 32)
            )

            print(f"\nValidation Dataset:")
            print(f"  Samples: {len(val_dataset)}")
            print(f"  Batches: {len(val_dataloader)}")
        elif val_split > 0:
            from torch.utils.data import random_split
            total = len(self.train_dataset)
            val_size = int(total * val_split)
            train_size = total - val_size
            train_subset, val_subset = random_split(
                self.train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            num_workers = self.config.num_workers
            if self.device.type == 'mps' and num_workers > 0:
                num_workers = 0

            self.train_loader = DataLoader(
                train_subset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            val_dataloader = DataLoader(
                val_subset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            val_dataset = val_subset

            print(f"\nValidation (split):")
            print(f"  Train samples: {len(train_subset)}")
            print(f"  Val samples: {len(val_subset)}")
        else:
            print("\nNo validation dataset")
            return None, None

        return val_dataloader, val_dataset

    def _create_model(self):
        """Create fusion model with VFIMamba"""
        model = build_fusion_net_vfi(
            base_channels=self.config.base_channels,
            vfi_core=self.vfi_core,
            vfi_down_scale=getattr(self.config, 'vfi_down_scale', 1.0),
            vfi_local=getattr(self.config, 'vfi_local', False)
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nFusion Model:")
        print(f"  Trainable parameters: {total_params:,}")
        print(f"  Base channels: {self.config.base_channels}")
        print(f"  Num anchors: {self.config.num_anchors}")
        return model

    def _create_optimizer(self):
        params = self.model.parameters()

        if self.config.optimizer == 'adam':
            optimizer = optim.Adam(params, lr=self.config.learning_rate,
                                  betas=(0.9, 0.999),
                                  weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adamw':
            optimizer = optim.AdamW(params, lr=self.config.learning_rate,
                                   betas=(0.9, 0.999),
                                   weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        return optimizer

    def _create_scheduler(self):
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        else:
            return None

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = {'total': 0, 'l1': 0, 'char': 0, 'freq': 0, 
                       'perceptual': 0, 'edge': 0}
        epoch_psnr = 0
        epoch_ssim = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}/{self.config.epochs}")
        for batch_idx, batch in enumerate(pbar):
            I0_all = batch['I0'].to(self.device, non_blocking=True)
            I1_all = batch['I1'].to(self.device, non_blocking=True)
            timesteps = batch['timesteps'].to(self.device, non_blocking=True)
            I_gt = batch['I_gt'].to(self.device, non_blocking=True)

            # Forward pass with optional mixed precision
            with autocast(device_type=self.autocast_device, dtype=self.amp_dtype, 
                         enabled=self.use_amp):
                # Model generates flows internally via VFIMamba
                output, aux = self.model(I0_all, I1_all, timesteps)
                losses = self.loss_fn(output, I_gt, aux)
                total_loss = losses['total']

            self.optimizer.zero_grad()
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 
                                            self.config.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 
                                            self.config.grad_clip)
                
                self.optimizer.step()

            # Metrics (always in fp32)
            with torch.no_grad():
                output_fp32 = output.float() if output.dtype != torch.float32 else output
                I_gt_fp32 = I_gt.float() if I_gt.dtype != torch.float32 else I_gt
                
                output_resized = self.metrics_calc.resize_for_metrics(output_fp32)
                I_gt_resized = self.metrics_calc.resize_for_metrics(I_gt_fp32)
                batch_psnr = self.metrics_calc.calculate_psnr(output_resized, I_gt_resized)
                batch_ssim = self.metrics_calc.calculate_ssim(output_resized, I_gt_resized)
                epoch_psnr += batch_psnr
                epoch_ssim += batch_ssim

            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k] += float(losses[k].item())

            num_batches += 1

            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'l1': f"{losses.get('l1', 0):.4f}",
                'psnr': f"{batch_psnr:.2f}",
                'ssim': f"{batch_ssim:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_loss': total_loss.item(),
                    'train/batch_l1': float(losses.get('l1', 0)),
                    'train/batch_psnr': batch_psnr,
                    'train/batch_ssim': batch_ssim,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'step': epoch * len(self.train_loader) + batch_idx
                })

        for k in epoch_losses:
            epoch_losses[k] /= max(1, num_batches)
        epoch_psnr /= max(1, num_batches)
        epoch_ssim /= max(1, num_batches)
        epoch_losses['psnr'] = epoch_psnr
        epoch_losses['ssim'] = epoch_ssim
        return epoch_losses

    def validate_epoch(self, epoch):
        if self.val_loader is None:
            return None

        self.model.eval()
        val_losses = {'total': 0, 'l1': 0, 'char': 0, 'freq': 0,
                     'perceptual': 0, 'edge': 0}
        val_psnr = 0
        val_ssim = 0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}/{self.config.epochs}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                I0_all = batch['I0'].to(self.device, non_blocking=True)
                I1_all = batch['I1'].to(self.device, non_blocking=True)
                timesteps = batch['timesteps'].to(self.device, non_blocking=True)
                I_gt = batch['I_gt'].to(self.device, non_blocking=True)

                with autocast(device_type=self.autocast_device, dtype=self.amp_dtype,
                             enabled=self.use_amp):
                    output, aux = self.model(I0_all, I1_all, timesteps)
                    losses = self.loss_fn(output, I_gt, aux)

                output_fp32 = output.float() if output.dtype != torch.float32 else output
                I_gt_fp32 = I_gt.float() if I_gt.dtype != torch.float32 else I_gt
                
                output_resized = self.metrics_calc.resize_for_metrics(output_fp32)
                I_gt_resized = self.metrics_calc.resize_for_metrics(I_gt_fp32)
                batch_psnr = self.metrics_calc.calculate_psnr(output_resized, I_gt_resized)
                batch_ssim = self.metrics_calc.calculate_ssim(output_resized, I_gt_resized)
                val_psnr += batch_psnr
                val_ssim += batch_ssim

                for k in val_losses:
                    if k in losses:
                        val_losses[k] += float(losses[k].item())

                num_batches += 1
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'psnr': f"{batch_psnr:.2f}",
                    'ssim': f"{batch_ssim:.4f}"
                })

        for k in val_losses:
            val_losses[k] /= max(1, num_batches)
        val_psnr /= max(1, num_batches)
        val_ssim /= max(1, num_batches)
        val_losses['psnr'] = val_psnr
        val_losses['ssim'] = val_ssim
        return val_losses

    def save_sample_images(self, epoch, num_samples=4):
        self.model.eval()
        with torch.no_grad():
            sample_batch = next(iter(self.val_loader if self.val_loader else self.train_loader))
            num_samples = min(num_samples, sample_batch['I0'].shape[0])

            I0_all = sample_batch['I0'][:num_samples].to(self.device)
            I1_all = sample_batch['I1'][:num_samples].to(self.device)
            timesteps = sample_batch['timesteps'][:num_samples].to(self.device)
            I_gt = sample_batch['I_gt'][:num_samples].to(self.device)

            output, aux = self.model(I0_all, I1_all, timesteps)
            output = output.clamp(0, 1).cpu()
            I_gt = I_gt.clamp(0, 1).cpu()
            I0 = I0_all[:, 0].clamp(0, 1).cpu()
            I1 = I1_all[:, 0].clamp(0, 1).cpu()

            for idx in range(num_samples):
                images = [I0[idx], I1[idx], I_gt[idx], output[idx]]
                error = torch.abs(I_gt[idx] - output[idx]) * 5.0
                error = error.clamp(0, 1)
                images.append(error)

                grid = torch.cat(images, dim=2)
                save_path = self.sample_dir / f'epoch_{epoch:04d}_sample_{idx:02d}.png'
                torchvision.utils.save_image(grid, save_path)

            batch_grid = torchvision.utils.make_grid(
                torch.cat([I0, I1, I_gt, output], dim=0),
                nrow=num_samples, normalize=True, scale_each=True
            )
            batch_path = self.sample_dir / f'epoch_{epoch:04d}_batch.png'
            torchvision.utils.save_image(batch_grid, batch_path)

            if self.use_wandb:
                wandb.log({
                    'samples/batch_grid': wandb.Image(batch_grid.permute(1, 2, 0).numpy()),
                    'epoch': epoch
                })

            self.writer.add_image('samples/inputs_0', 
                                 torchvision.utils.make_grid(I0, nrow=2, normalize=True), epoch)
            self.writer.add_image('samples/ground_truth',
                                 torchvision.utils.make_grid(I_gt, nrow=2, normalize=True), epoch)
            self.writer.add_image('samples/predictions',
                                 torchvision.utils.make_grid(output, nrow=2, normalize=True), epoch)

            print(f"  Samples saved to: {self.sample_dir}")

        self.model.train()

    def save_checkpoint(self, epoch, is_best=False, is_best_val=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'best_val_psnr': self.best_val_psnr,
            'best_val_ssim': self.best_val_ssim,
            'epoch_metrics': self.epoch_metrics,
            'config': vars(self.config),
            'precision': self.precision,
            'wandb_run_id': wandb.run.id if self.use_wandb else None
        }

        if epoch % self.config.save_interval == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print("  Best training model saved!")

        if is_best_val:
            best_val_path = self.checkpoint_dir / 'best_val_model.pth'
            torch.save(checkpoint, best_val_path)
            print("  Best validation model saved!")

        latest_path = self.checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_psnr = checkpoint.get('best_psnr', 0)
        self.best_ssim = checkpoint.get('best_ssim', 0)
        self.best_val_psnr = checkpoint.get('best_val_psnr', 0)
        self.best_val_ssim = checkpoint.get('best_val_ssim', 0)
        self.epoch_metrics = checkpoint.get('epoch_metrics', [])

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.start_epoch}")

    def train(self):
        print(f"\nStarting training for {self.config.epochs} epochs")
        print("-" * 60)
        
        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_start = datetime.now()

            train_losses = self.train_epoch(epoch)
            val_losses = self.validate_epoch(epoch) if self.val_loader else None

            sample_interval = getattr(self.config, 'sample_interval', 5)
            if epoch % sample_interval == 0 or epoch == 0:
                print("\nSaving visual samples...")
                self.save_sample_images(epoch, num_samples=4)

            is_best = train_losses['psnr'] > self.best_psnr
            if is_best:
                self.best_psnr = train_losses['psnr']
                self.best_ssim = train_losses['ssim']
                print(f"  New best train PSNR: {self.best_psnr:.2f} dB")

            is_best_val = False
            if val_losses is not None:
                is_best_val = val_losses['psnr'] > self.best_val_psnr
                if is_best_val:
                    self.best_val_psnr = val_losses['psnr']
                    self.best_val_ssim = val_losses['ssim']
                    print(f"  New best val PSNR: {self.best_val_psnr:.2f} dB")

            if self.scheduler:
                self.scheduler.step()

            # Logging
            for name, value in train_losses.items():
                self.writer.add_scalar(f'train/{name}', value, epoch)
            if val_losses:
                for name, value in val_losses.items():
                    self.writer.add_scalar(f'val/{name}', value, epoch)
            self.writer.add_scalar('learning_rate', 
                                  self.optimizer.param_groups[0]['lr'], epoch)

            if self.use_wandb:
                wb = {f'train/{k}': v for k, v in train_losses.items()}
                if val_losses:
                    wb.update({f'val/{k}': v for k, v in val_losses.items()})
                wb['epoch'] = epoch
                wb['learning_rate'] = self.optimizer.param_groups[0]['lr']
                wandb.log(wb)

            epoch_time = (datetime.now() - epoch_start).total_seconds()
            print(f"\nEpoch {epoch}/{self.config.epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_losses['total']:.4f}, "
                  f"PSNR: {train_losses['psnr']:.2f} dB, SSIM: {train_losses['ssim']:.4f}")
            if val_losses:
                print(f"  Val   - Loss: {val_losses['total']:.4f}, "
                      f"PSNR: {val_losses['psnr']:.2f} dB, SSIM: {val_losses['ssim']:.4f}")

            self.save_checkpoint(epoch, is_best, is_best_val)

            record = {
                'epoch': epoch,
                'train': train_losses,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            if val_losses:
                record['val'] = val_losses
            self.epoch_metrics.append(record)

            with open(self.checkpoint_dir / 'metrics.json', 'w') as f:
                json.dump(self.epoch_metrics, f, indent=2, default=str)

        self.writer.close()
        if self.use_wandb:
            wandb.finish()
        print("\nTraining completed!")
        print(f"Best train PSNR: {self.best_psnr:.2f} dB, SSIM: {self.best_ssim:.4f}")
        if self.val_loader:
            print(f"Best val PSNR: {self.best_val_psnr:.2f} dB, SSIM: {self.best_val_ssim:.4f}")


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Train VFI Fusion Model')

    # Dataset
    parser.add_argument('--gt_paths', type=str, nargs='+', required=True)
    parser.add_argument('--steps', type=int, nargs='+', required=True)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--val_paths', type=str, nargs='+', default=None)
    parser.add_argument('--val_steps', type=int, nargs='+', default=None)
    parser.add_argument('--divisor', type=int, default=32)

    # Model
    parser.add_argument('--num_anchors', type=int, default=3)
    parser.add_argument('--base_channels', type=int, default=48)
    parser.add_argument('--scale', type=float, default=1.0)
    
    # VFIMamba
    parser.add_argument('--vfi_weights', type=str, default=None,
                       help='Path to pretrained VFIMamba weights')
    parser.add_argument('--vfi_down_scale', type=float, default=1.0)
    parser.add_argument('--vfi_local', action='store_true')
    parser.add_argument('--vfi_F', type=int, default=16)
    parser.add_argument('--vfi_depth', type=int, nargs='+', default=[2,2,2,3,3])

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp32', 'fp16', 'bf16'])

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'])
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.1)

    # Logging
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--sample_interval', type=int, default=5)

    # W&B
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='vfi-fusion')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_watch_model', action='store_true')

    # Other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mix_strategy', type=str, default='uniform')
    parser.add_argument('--metric_size', type=int, nargs=2, default=[256, 256])

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = FusionTrainer(args)
    trainer.train()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()