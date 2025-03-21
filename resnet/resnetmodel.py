import torch
import torch.nn as nn
import torchvision.models as models
import dfresearch.loaderconf as config


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ComplexBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResnetClassifier(nn.Module):
    def __init__(self, num_classes, pretrained, use_complex_blocks, in_channels=config.INPUT_CHANNELS):
        super().__init__()
        self.base = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )

        if use_complex_blocks:
            self.modify_resnet_architecture()

        original_conv1 = self.base.conv1
        self.base.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        with torch.no_grad():
            self.base.conv1.weight[:, :3] = original_conv1.weight.clone()
            self.base.conv1.weight[:, 3:] = original_conv1.weight.mean(dim=1, keepdim=True)
        
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        self.init_weights()

    def modify_resnet_architecture(self):
        self.base.layer2 = nn.Sequential(
            ComplexBlock(256, 128, stride=2),
            ComplexBlock(512, 128),
            ComplexBlock(512, 128),
            ComplexBlock(512, 128),
        )

        self.base.layer3 = nn.Sequential(
            ComplexBlock(512, 256, stride=2),
            ComplexBlock(1024, 256),
            ComplexBlock(1024, 256),
            ComplexBlock(1024, 256),
            ComplexBlock(1024, 256),
        )

        self.base.layer4 = nn.Sequential(
            ComplexBlock(1024, 512, stride=2),
            ComplexBlock(2048, 512),
            ComplexBlock(2048, 512),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.base(x)
