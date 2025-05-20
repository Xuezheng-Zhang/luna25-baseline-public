import timm 
import torch
import torch.nn as nn

from experiment_config import config


class TIMMPretrainedModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg 
        
        self.backbone = timm.create_model(
            cfg.MODEL_NAME,
            pretrained=cfg.PRETRAINED,
            in_chans=cfg.IN_CHANS
        )
        if 'efficientnet' in cfg.MODEL_NAME:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.MODEL_NAME:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')
            
        self.classifier = nn.Linear(backbone_out, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        
        logits = self.classifier(features)
        
        return logits
    

if __name__ == "__main__":
    model = timm.create_model('resnet34')
    image = torch.randn(4, 3, 64, 64)
    print(model(image).shape)
    