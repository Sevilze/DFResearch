import torch
import torch.nn as nn
import torchvision.models as models


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


class ELA(nn.Module):
    def __init__(self, channels, compression_quality=90):
        super().__init__()
        self.channels = channels
        self.quality = compression_quality
        self.process = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        blur = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        quantized = torch.round(blur * self.quality) / self.quality
        ela = torch.abs(x - quantized)
        return x + self.process(ela)


class FrequencyAnalysis(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_after_fft = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.float()

        fft = torch.fft.rfft2(x)
        fft_real = torch.view_as_real(fft)
        fft_real = fft_real.permute(0, 1, 4, 2, 3).reshape(b, c * 2, h, w // 2 + 1)
        fft_info = torch.nn.functional.interpolate(
            fft_real, size=(h, w), mode="bilinear", align_corners=False
        )
        return self.conv_after_fft(fft_info)


class NoiseAnalysis(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.high_pass = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )

        with torch.no_grad():
            nn.init.constant_(self.high_pass.weight, 0)
            self.high_pass.weight[:, :, 1, 1] = 1
            self.high_pass.weight[:, :, 0, 1] = -0.25
            self.high_pass.weight[:, :, 2, 1] = -0.25
            self.high_pass.weight[:, :, 1, 0] = -0.25
            self.high_pass.weight[:, :, 1, 2] = -0.25
            self.noise_processor = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        noise = self.high_pass(x)
        processed_noise = self.noise_processor(torch.abs(noise))
        return x + processed_noise


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pool = torch.cat([avg_pool, max_pool], dim=1)
        attention = torch.sigmoid(self.conv(pool))
        return x * attention


class CrossResolution(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.down2x = nn.AvgPool2d(2)
        self.down4x = nn.AvgPool2d(4)
        self.up2x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up4x = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

        self.scale_processor = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        down2x = self.down2x(x)
        down4x = self.down4x(x)

        up_from_2x = self.up2x(down2x)
        up_from_4x = self.up4x(down4x)

        diff2x = x - up_from_2x
        diff4x = x - up_from_4x

        combined = torch.cat([x, diff2x, diff4x], dim=1)
        attention = self.scale_processor(combined)

        return x * attention


class ResnetClassifier(nn.Module):
    def __init__(self, num_classes, pretrained, use_complex_blocks, in_channels):
        super().__init__()
        self.base = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.spatial_attention = SpatialAttention()
        self.noise_module = NoiseAnalysis(64)
        self.ela_module = ELA(64)
        self.cross_res_module = CrossResolution(64)

        self.channel_attention1 = ChannelAttention(64)
        self.channel_attention2 = ChannelAttention(256)
        self.channel_attention3 = ChannelAttention(512)
        self.channel_attention4 = ChannelAttention(1024)
        self.channel_attention5 = ChannelAttention(2048)

        if use_complex_blocks:
            self.modify_resnet_architecture()

        original_conv1 = self.base.conv1
        self.base.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )

        with torch.no_grad():
            self.base.conv1.weight[:, :3] = original_conv1.weight.clone()
            self.base.conv1.weight[:, 3:] = original_conv1.weight.mean(
                dim=1, keepdim=True
            )

        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
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
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)

        x = self.channel_attention1(x)
        x = self.spatial_attention(x)
        x = self.cross_res_module(x)

        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.channel_attention2(x)

        x = self.base.layer2(x)
        x = self.channel_attention3(x)

        x = self.base.layer3(x)
        x = self.channel_attention4(x)

        x = self.base.layer4(x)
        x = self.channel_attention5(x)

        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.fc(x)

        return x
