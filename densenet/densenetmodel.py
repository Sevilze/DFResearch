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
        fft_info = torch.nn.functional.interpolate(fft_real, size=(h, w), mode="bilinear", align_corners=False)
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
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        downsampled = self.down(x)
        upsampled = self.up(downsampled)
        diff = x - upsampled
        return self.fusion(torch.cat([x, diff], dim=1))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 
                               kernel_size=1, stride=1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.add_module(f'denselayer{i+1}',
                                  DenseLayer(in_channels + i * growth_rate, 
                                            growth_rate, bn_size))
            
    def forward(self, x):
        features = x
        for layer in self.layers:
            features = layer(features)
        return features


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DensenetClassifier(nn.Module):
    def __init__(self, num_classes, pretrained, in_channels):
        super().__init__()
        self.base = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        original_features = self.base.features.conv0
        self.base.features.conv0 = nn.Conv2d(
            in_channels,
            original_features.out_channels,
            kernel_size=original_features.kernel_size,
            stride=original_features.stride,
            padding=original_features.padding,
            bias=False,
        )
        
        with torch.no_grad():
            self.base.features.conv0.weight[:, :3] = original_features.weight.clone()
            if in_channels > 3:
                self.base.features.conv0.weight[:, 3:] = original_features.weight.mean(
                    dim=1, keepdim=True
                )
        
        growth_rate = 32
        block_config = [6, 12, 24, 16]
        
        self.channels = {
            'stem': 64,
            'transition1': 128 + 6 * growth_rate,
            'transition2': 128 + 6 * growth_rate + 12 * growth_rate,
            'transition3': 128 + 6 * growth_rate + 12 * growth_rate + 24 * growth_rate,
            'final': 1024
        }
        
        self.spatial_attention = SpatialAttention()
        self.channel_attention_stem = ChannelAttention(self.channels['stem'])
        self.channel_attention_t1 = ChannelAttention(self.channels['transition1'] // 2)
        self.channel_attention_t2 = ChannelAttention(self.channels['transition2'] // 2)
        self.channel_attention_t3 = ChannelAttention(self.channels['transition3'] // 2)
        self.channel_attention_final = ChannelAttention(self.channels['final'])
        
        self.noise_module = NoiseAnalysis(self.channels['stem'])
        self.ela_module = ELA(self.channels['stem'])
        self.cross_res_module = CrossResolution(self.channels['stem'])
        
        num_ftrs = self.base.classifier.in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.base.features.conv0(x)
        x = self.base.features.norm0(x)
        x = self.base.features.relu0(x)
        
        x = self.channel_attention_stem(x)
        x = self.spatial_attention(x)
        # x = self.noise_module(x)
        # x = self.ela_module(x)
        x = self.cross_res_module(x)
        
        x = self.base.features.pool0(x)
        
        x = self.base.features.denseblock1(x)
        x = self.base.features.transition1.norm(x)
        x = self.base.features.transition1.relu(x)
        x = self.base.features.transition1.conv(x)
        x = self.channel_attention_t1(x)
        x = self.base.features.transition1.pool(x)
        
        x = self.base.features.denseblock2(x)
        x = self.base.features.transition2.norm(x)
        x = self.base.features.transition2.relu(x)
        x = self.base.features.transition2.conv(x)
        x = self.channel_attention_t2(x)
        x = self.base.features.transition2.pool(x)
        
        x = self.base.features.denseblock3(x)
        x = self.base.features.transition3.norm(x)
        x = self.base.features.transition3.relu(x)
        x = self.base.features.transition3.conv(x)
        x = self.channel_attention_t3(x)
        x = self.base.features.transition3.pool(x)
        
        x = self.base.features.denseblock4(x)
        x = self.base.features.norm5(x)
        x = self.channel_attention_final(x)
        
        x = torch.nn.functional.relu(x, inplace=True)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.base.classifier(x)
        
        return x
