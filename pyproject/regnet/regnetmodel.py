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


class ELA(nn.Module):
    def __init__(self, channels, compression_quality=90):
        super().__init__()
        self.channels = channels
        self.quality = compression_quality
        self.process = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        blur = nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
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
        fft_info = nn.functional.interpolate(
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


class CrossResolution(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        downsampled = self.down(x)
        upsampled = self.up(downsampled)
        diff = x - upsampled
        return self.fusion(torch.cat([x, diff], dim=1))


class RegnetClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, in_channels=3):
        super().__init__()
        self.base = models.regnet_y_8gf(
            weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V2 if pretrained else None
        )
        original_conv = self.base.stem[0]
        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, : min(3, in_channels)] = original_conv.weight[
                :, : min(3, in_channels)
            ]
            if in_channels > 3:
                avg_weight = original_conv.weight[:, :3].mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:] = avg_weight
        self.base.stem[0] = new_conv

        stem_channels = new_conv.out_channels
        self.forensic_stem = nn.Sequential(
            CrossResolution(stem_channels),
            SpatialAttention(),
            ChannelAttention(stem_channels),
        )

        in_features = self.base.fc.in_features
        self.classifier_attention = ChannelAttention(in_features)

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base.stem(x)
        x = self.forensic_stem(x)
        x = self.base.trunk_output(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)

        x = (
            self.classifier_attention(x.unsqueeze(-1).unsqueeze(-1))
            .squeeze(-1)
            .squeeze(-1)
        )
        x = self.classifier(x)
        return x
