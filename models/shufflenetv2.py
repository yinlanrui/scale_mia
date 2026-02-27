import torch
import torch.nn as nn

__all__ = ['ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']


def channel_shuffle(x, groups):
    """Channel shuffle operation"""
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    
    # transpose
    x = torch.transpose(x, 1, 2).contiguous()
    
    # flatten
    x = x.view(batch_size, -1, height, width)
    
    return x


class InvertedResidual(nn.Module):
    """ShuffleNetV2 basic unit"""
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        
        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride
        
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)
        
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                     branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )
    
    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)
    
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        
        out = channel_shuffle(out, 2)
        
        return out


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2 architecture"""
    def __init__(self, stages_repeats, stages_out_channels, num_classes=10, inverted_residual=InvertedResidual):
        super(ShuffleNetV2, self).__init__()
        
        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(output_channels, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
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
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def shufflenet_v2_x0_5(num_classes=10):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    """
    return ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], num_classes=num_classes)


def shufflenet_v2_x1_0(num_classes=10):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    """
    return ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], num_classes=num_classes)


def shufflenet_v2_x1_5(num_classes=10):
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    """
    return ShuffleNetV2([4, 8, 4], [24, 176, 352, 704, 1024], num_classes=num_classes)


def shufflenet_v2_x2_0(num_classes=10):
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    """
    return ShuffleNetV2([4, 8, 4], [24, 244, 488, 976, 2048], num_classes=num_classes)
