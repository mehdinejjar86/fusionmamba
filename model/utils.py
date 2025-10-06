import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 norm='none', activation='relu', bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm = None
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'gn':
            num_groups = max(1, min(32, out_channels // 4))
            while out_channels % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            self.norm = nn.GroupNorm(num_groups, out_channels)

        self.activation = None
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'silu':
            self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class AdaptiveFrequencyDecoupling(nn.Module):
    def __init__(self, channels, groups=2):
        super().__init__()
        assert channels % groups == 0, "channels must be divisible by groups"
        self.freq_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1), nn.Sigmoid()
        )
        self.low_freq_conv = nn.Conv2d(channels, channels, 3, 1, 1, groups=groups)

    def forward(self, x):
        w = self.freq_weight(x)
        low = self.low_freq_conv(x) * w
        high = x - low
        return low, high


class DetailPreservingAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.low_branch = nn.Sequential(nn.Conv2d(channels, channels // 8, 1), nn.ReLU(inplace=True),
                                        nn.Conv2d(channels // 8, channels, 1))
        self.high_branch = nn.Sequential(nn.Conv2d(channels, channels // 8, 1), nn.ReLU(inplace=True),
                                         nn.Conv2d(channels // 8, channels, 1))
        self.mix_weight = nn.Parameter(torch.tensor([0.7, 0.3]))

    def forward(self, x, low_freq, high_freq):
        low_attn = torch.sigmoid(self.low_branch(low_freq))
        high_attn = torch.sigmoid(self.high_branch(high_freq))
        w = F.softmax(self.mix_weight, dim=0)
        attn = w[0] * low_attn + w[1] * high_attn
        return x * attn


class DetailAwareResBlock(nn.Module):
    def __init__(self, channels, norm='gn', preserve_details=True):
        super().__init__()
        self.preserve_details = preserve_details
        self.conv1 = ConvBlock(channels, channels, norm=norm, activation='leaky')
        self.conv2 = ConvBlock(channels, channels, norm=norm, activation='none')
        if preserve_details:
            self.freq = AdaptiveFrequencyDecoupling(channels)
            self.dpa = DetailPreservingAttention(channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.residual_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        r = x
        y = self.conv1(x); y = self.conv2(y)
        if self.preserve_details:
            low, high = self.freq(y)
            y = self.dpa(y, low, high)
        y = y + r * self.residual_weight
        return self.activation(y)


class LearnedUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.refine = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.fallback_proj = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, target_size=None):
        if target_size is not None:
            sh = target_size[0] / x.shape[2]; sw = target_size[1] / x.shape[3]
            close = abs(sh - self.scale_factor) < 0.1 and abs(sw - self.scale_factor) < 0.1
            if close:
                x = self.pixel_shuffle(self.conv(x))
                x = self.refine(x)
            else:
                x = self.fallback_proj(x)
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
                x = self.refine(x)
        else:
            x = self.pixel_shuffle(self.conv(x))
            x = self.refine(x)
        return x