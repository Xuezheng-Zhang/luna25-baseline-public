import timm
from timm.models.resnet import resnet50
import torch
import torch.nn as nn



class TIMMPretrainedModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg 

        extra_kwargs = {}
        if 'vit' in cfg.MODEL_NAME:
            extra_kwargs['img_size'] = (64, 64)
        
        self.backbone = timm.create_model(
            cfg.MODEL_NAME,
            pretrained=cfg.PRETRAINED,
            in_chans=cfg.IN_CHANS,
            **extra_kwargs
        )

        if 'efficientnet' in cfg.MODEL_NAME:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.MODEL_NAME:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif 'vit' in cfg.MODEL_NAME:
            backbone_out = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')
    
        self.classifier = nn.Linear(backbone_out, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        
        return logits
        

if __name__ == "__main__":
    MODEL_NAME = "resnet50"
    PRETRAINED = True
    IN_CHANS = 3
    timm.create_model(
        MODEL_NAME,
        pretrained=PRETRAINED,
        in_chans=IN_CHANS,
        cache_dir="./resources"
    )
