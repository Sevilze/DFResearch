import torch
import torch.nn as nn

class EarlyFusionEnsemble(nn.Module):
    def __init__(self, num_classes, in_channels, resnet_model, densenet_model, regnet_model, resnet_path, densenet_path, regnet_path, freeze):
        super().__init__()
        
        self.resnet = resnet_model
        checkpoint = torch.load(resnet_path, weights_only=True)
        self.resnet.load_state_dict(checkpoint["model_state_dict"])
        resnet_dim = self.resnet.base.fc[0].in_features
        self.resnet.base.fc = nn.Identity()
        
        self.densenet = densenet_model
        checkpoint = torch.load(densenet_path, weights_only=True)
        self.densenet.load_state_dict(checkpoint["model_state_dict"])
        densenet_dim = self.densenet.base.classifier[0].in_features
        self.densenet.base.classifier = nn.Identity()
        
        self.regnet = regnet_model
        checkpoint = torch.load(regnet_path, weights_only=True)
        self.regnet.load_state_dict(checkpoint["model_state_dict"])
        regnet_dim = self.regnet.classifier[0].in_features
        self.regnet.classifier = nn.Identity()
        
        if freeze:
            for model in [self.resnet, self.densenet, self.regnet]:
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
        feat_resnet = self.resnet(x)
        feat_densenet = self.densenet(x)
        feat_regnet = self.regnet(x)
        
        combined = torch.cat([feat_resnet, feat_densenet, feat_regnet], dim=1)
        out = self.classifier(combined)
        return out
