# fusionloss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights  # ← Add VGG19_Weights


class AnchorFusionLoss(nn.Module):
    def __init__(self, use_gan=False):
        super().__init__()
        
        # Perceptual (VGG features) - FIXED
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:36].eval()  
        for p in vgg.parameters(): 
            p.requires_grad = False
        self.vgg = vgg
        self.vgg_layers = [3, 8, 17, 26, 35]
        
        # Optional: Discriminator for adversarial
        self.use_gan = use_gan
        if self.use_gan:
            self.discriminator = MultiScaleDiscriminator()
        else:
            self.discriminator = None
        
    def forward(self, pred, target, aux, pyramid_feats=None):
        """
        pred: [B,3,H,W] - final output
        target: [B,3,H,W] - ground truth
        aux: dict with t_weights, prior_weighted, flows, masks
        pyramid_feats: dict with low/mid/high features for multi-scale loss
        """
        losses = {}
        
        # ============ 1. MULTI-SCALE RECONSTRUCTION ============
        losses['l1'] = F.l1_loss(pred, target)
        losses['char'] = self.charbonnier_loss(pred, target)
        
        # If you expose intermediate features from decoder
        if pyramid_feats:
            for scale, feat in pyramid_feats.items():
                target_down = F.interpolate(target, size=feat.shape[-2:], 
                                            mode='bilinear', align_corners=False)
                losses[f'l1_{scale}'] = F.l1_loss(feat, target_down) * 0.5
        
        # ============ 2. FREQUENCY-AWARE LOSS ============
        losses['freq'] = self.frequency_loss(pred, target, 
                                             lo_weight=1.0, hi_weight=2.0)
        
        # ============ 3. PERCEPTUAL LOSS ============
        losses['perceptual'] = self.perceptual_loss(pred, target)
        
        # ============ 4. EDGE/DETAIL PRESERVATION ============
        losses['edge'] = self.edge_loss(pred, target)
        
        # ============ 5. ADVERSARIAL LOSS (optional) ============
        if self.use_gan and self.discriminator is not None:
            losses['gan_g'] = self.generator_loss(pred)
        
        # ============ 6. TEMPORAL WEIGHT REGULARIZATION ============
        t_weights = aux['t_weights']  # [B,N]
        # Encourage diversity (not all anchors same weight)
        losses['weight_entropy'] = -self.entropy(t_weights) * 0.01
        # Encourage sparsity (some anchors should dominate)
        losses['weight_sparse'] = (t_weights ** 2).sum(dim=1).mean() * 0.01
        
        # ============ 7. PRIOR QUALITY LOSS ============
        prior_weighted = aux['prior_weighted']  # [B,3,H,W]
        losses['prior_l1'] = F.l1_loss(prior_weighted, target) * 0.3
        
        # ============ 8. FLOW SMOOTHNESS (optional) ============
        flows = aux['flows']  # [B,N,4,H,W]
        losses['flow_smooth'] = self.flow_smoothness(flows) * 0.01
        
        # ============ TOTAL LOSS ============
        total = (
            losses['l1'] * 1.0 +
            losses['char'] * 1.0 +
            losses['freq'] * 0.5 +
            losses['perceptual'] * 0.1 +
            losses['edge'] * 0.5 +
            losses.get('gan_g', torch.tensor(0.0, device=pred.device)) * 0.01 +
            losses['weight_entropy'] +
            losses['weight_sparse'] +
            losses['prior_l1'] +
            losses['flow_smooth']
        )
        
        if pyramid_feats:
            for k in losses:
                if k.startswith('l1_'):
                    total += losses[k]
        
        losses['total'] = total
        return losses
    
    # ============ HELPER FUNCTIONS ============
    
    def charbonnier_loss(self, pred, target, eps=1e-6):
        """More robust than L1"""
        return torch.sqrt((pred - target) ** 2 + eps).mean()
    
    def frequency_loss(self, pred, target, lo_weight=1.0, hi_weight=2.0):
        """Separate loss for low/high frequencies"""
        # FFT
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # Magnitude
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # LF/HF masks
        H, W = pred.shape[-2:]
        mask_lf = self._freq_mask(H, W, cutoff=0.3, device=pred.device)
        mask_hf = 1 - mask_lf
        
        loss_lf = F.l1_loss(pred_mag * mask_lf, target_mag * mask_lf)
        loss_hf = F.l1_loss(pred_mag * mask_hf, target_mag * mask_hf)
        
        return lo_weight * loss_lf + hi_weight * loss_hf
    
    def _freq_mask(self, H, W, cutoff=0.3, device='cpu'):
        fy = torch.fft.fftfreq(H, d=1.0).to(device).abs()
        fx = torch.fft.rfftfreq(W, d=1.0).to(device).abs()
        wy, wx = torch.meshgrid(fy, fx, indexing='ij')
        r = torch.sqrt(wx**2 + wy**2)
        return (r < cutoff).float().unsqueeze(0).unsqueeze(0)
    
    def perceptual_loss(self, pred, target):
        """VGG perceptual loss at multiple layers"""
        pred_feats = self.extract_vgg_features(pred)
        target_feats = self.extract_vgg_features(target)
        
        loss = 0
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        for (pf, tf, w) in zip(pred_feats, target_feats, weights):
            loss += F.l1_loss(pf, tf) * w
        return loss / len(pred_feats)
    
    def extract_vgg_features(self, x):
        feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.vgg_layers:
                feats.append(x)
        return feats
    
    def edge_loss(self, pred, target):
        """Sobel edge detection loss"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-2, -1)
        
        def get_edges(img):
            gray = img.mean(dim=1, keepdim=True)
            edge_x = F.conv2d(gray, sobel_x, padding=1)
            edge_y = F.conv2d(gray, sobel_y, padding=1)
            return torch.sqrt(edge_x**2 + edge_y**2)
        
        pred_edges = get_edges(pred)
        target_edges = get_edges(target)
        return F.l1_loss(pred_edges, target_edges)
    
    def entropy(self, weights):
        """Entropy of weight distribution"""
        eps = 1e-8
        return -(weights * torch.log(weights + eps)).sum(dim=1).mean()
    
    def flow_smoothness(self, flows):
        """First-order smoothness on flows"""
        # flows: [B,N,4,H,W]
        diff_x = (flows[..., :, 1:] - flows[..., :, :-1]).abs().mean()
        diff_y = (flows[..., 1:, :] - flows[..., :-1, :]).abs().mean()
        return diff_x + diff_y
    
    def generator_loss(self, pred):
        """Adversarial loss for generator"""
        if self.discriminator is None:
            return torch.tensor(0.0, device=pred.device)
        
        fake_pred = self.discriminator(pred)
        
        # Handle both single output and list of multi-scale outputs
        if isinstance(fake_pred, (list, tuple)):
            # Multi-scale discriminator returns list
            loss = 0
            for fp in fake_pred:
                loss += F.softplus(-fp).mean()
            return loss / len(fake_pred)
        else:
            # Single discriminator
            return F.softplus(-fake_pred).mean()
      
    
class MultiScaleDiscriminator(nn.Module):
    """Multi-scale patch discriminator"""
    def __init__(self, input_channels=3, ndf=64, n_layers=3, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        
        self.discriminators = nn.ModuleList()
        for _ in range(num_scales):
            self.discriminators.append(
                NLayerDiscriminator(input_channels, ndf, n_layers)
            )
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x):
        results = []
        input_x = x
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                input_x = self.downsample(input_x)
            results.append(disc(input_x))
        
        return results  # Return list for multi-scale GAN loss


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator (from pix2pix).
    Classifies whether image patches are real or fake.
    """
    def __init__(self, input_channels=3, ndf=64, n_layers=3):
        super().__init__()
        
        # Building blocks
        sequence = [
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                         kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Final classification layer
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        return self.model(x)