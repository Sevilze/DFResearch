import torch
import torch.nn as nn
import torch.jit as jit


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = None
        self.hook = None
        self.hook_registered = False

    def find_last_layer(self):
        last_linear = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        return last_linear

    def hook_fn(self, module, input):
        self.features = input[0].detach().clone()

    def forward(self, x):
        if not self.hook_registered:
            last_layer = self.find_last_layer()
            if last_layer is not None:
                self.hook = last_layer.register_forward_pre_hook(self.hook_fn)
                self.hook_registered = True

        output = self.model(x)

        if self.features is not None:
            return self.features

        return output

class IntermediateFusionEnsemble(nn.Module):
    def __init__(self, num_classes, in_channels, resnet_path, densenet_path, regnet_path, freeze):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.resnet_model = jit.load(resnet_path, map_location=device)
        self.densenet_model = jit.load(densenet_path, map_location=device)
        self.regnet_model = jit.load(regnet_path, map_location=device)

        self.resnet_model.eval()
        self.densenet_model.eval()
        self.regnet_model.eval()

        self.resnet = FeatureExtractor(self.resnet_model)
        self.densenet = FeatureExtractor(self.densenet_model)
        self.regnet = FeatureExtractor(self.regnet_model)

        dummy_input = torch.randn(1, in_channels, 224, 224).to(device)

        with torch.no_grad():
            resnet_features = self.resnet(dummy_input)
            densenet_features = self.densenet(dummy_input)
            regnet_features = self.regnet(dummy_input)

        if resnet_features.dim() > 2:
            resnet_features = torch.flatten(resnet_features, 1)
        if densenet_features.dim() > 2:
            densenet_features = torch.flatten(densenet_features, 1)
        if regnet_features.dim() > 2:
            regnet_features = torch.flatten(regnet_features, 1)

        resnet_dim = resnet_features.shape[1]
        densenet_dim = densenet_features.shape[1]
        regnet_dim = regnet_features.shape[1]

        if freeze:
            for model in [self.resnet_model, self.densenet_model, self.regnet_model]:
                for param in model.parameters():
                    param.requires_grad = False

        total_features = resnet_dim + densenet_dim + regnet_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            feat_resnet = self.resnet(x)
            feat_densenet = self.densenet(x)
            feat_regnet = self.regnet(x)

            if feat_resnet.dim() > 2:
                feat_resnet = torch.flatten(feat_resnet, 1)
            if feat_densenet.dim() > 2:
                feat_densenet = torch.flatten(feat_densenet, 1)
            if feat_regnet.dim() > 2:
                feat_regnet = torch.flatten(feat_regnet, 1)

        combined = torch.cat((feat_resnet, feat_densenet, feat_regnet), dim=1)
        out = self.classifier(combined)
        return out
