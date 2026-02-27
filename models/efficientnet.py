import torch
import torch.nn as nn
import math

__all__ = ['EfficientNetB0', 'EfficientNetB1']


class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, in_channels, se_ratio=0.25):
        super(SEBlock, self).__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, se_channels, 1),
            Swish(),
            nn.Conv2d(se_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Expansion phase
        hidden_dim = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            )
        else:
            self.expand_conv = nn.Identity()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        )
        
        # Squeeze-and-Excitation
        self.se = SEBlock(hidden_dim, se_ratio)
        
        # Output projection
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Dropout for stochastic depth
        self.dropout = nn.Dropout(0.2) if self.use_residual else nn.Identity()

    def forward(self, x):
        identity = x
        
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x = self.dropout(x)
            x = x + identity
        
        return x


class EfficientNet(nn.Module):
    """EfficientNet base class"""
    def __init__(self, width_mult=1.0, depth_mult=1.0, num_classes=10, dropout_rate=0.2):
        super(EfficientNet, self).__init__()
        
        # Building blocks configuration: [expand_ratio, channels, num_blocks, stride, kernel_size]
        blocks_config = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3]
        ]
        
        # Stem
        out_channels = self._round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )
        
        # Building blocks
        self.blocks = nn.ModuleList([])
        in_channels = out_channels
        
        for expand_ratio, channels, num_blocks, stride, kernel_size in blocks_config:
            out_channels = self._round_filters(channels, width_mult)
            num_blocks = self._round_repeats(num_blocks, depth_mult)
            
            for i in range(num_blocks):
                self.blocks.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride if i == 0 else 1,
                        expand_ratio=expand_ratio
                    )
                )
                in_channels = out_channels
        
        # Head
        final_channels = self._round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            Swish()
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(final_channels, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _round_filters(self, filters, width_mult):
        """Round number of filters based on width multiplier"""
        if width_mult == 1.0:
            return filters
        filters *= width_mult
        new_filters = max(8, int(filters + 4) // 8 * 8)
        if new_filters < 0.9 * filters:
            new_filters += 8
        return int(new_filters)
    
    def _round_repeats(self, repeats, depth_mult):
        """Round number of repeats based on depth multiplier"""
        if depth_mult == 1.0:
            return repeats
        return int(math.ceil(depth_mult * repeats))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def EfficientNetB0(num_classes=10):
    """EfficientNet-B0"""
    return EfficientNet(width_mult=1.0, depth_mult=1.0, num_classes=num_classes, dropout_rate=0.2)


def EfficientNetB1(num_classes=10):
    """EfficientNet-B1"""
    return EfficientNet(width_mult=1.0, depth_mult=1.1, num_classes=num_classes, dropout_rate=0.2)
