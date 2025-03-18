import torch.nn as nn
import torchvision.models as models
from torchvision.ops import StochasticDepth


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
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ca = ChannelAttention(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.stochastic_depth = StochasticDepth(p=0.2, mode="batch")

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out)
        out = self.dropout(out)
        out = self.stochastic_depth(out)
        out += identity
        return nn.ReLU()(out)


class ResnetClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, use_complex_blocks=True):
        super().__init__()
        self.base = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )

        if use_complex_blocks:
            self._modify_resnet_architecture()

        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _modify_resnet_architecture(self):
        self.base.layer2 = nn.Sequential(
            ComplexBlock(512, 512, stride=2),
            ComplexBlock(512, 512),
            ComplexBlock(512, 512),
            ComplexBlock(512, 512),
        )

        self.base.layer3 = nn.Sequential(
            ComplexBlock(1024, 1024, stride=2),
            ComplexBlock(1024, 1024),
            ComplexBlock(1024, 1024),
            ComplexBlock(1024, 1024),
            ComplexBlock(1024, 1024),
        )

        self.base.layer4 = nn.Sequential(
            ComplexBlock(2048, 2048, stride=2),
            ComplexBlock(2048, 2048),
            ComplexBlock(2048, 2048),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.base(x)


def ret_resnet(num_classes, pretrained=True):
    return ResnetClassifier(num_classes, pretrained=pretrained)
