import timm 
import torch


class TIMMPretrainedModel():
    def __init__(self):
        pass
    

if __name__ == "__main__":
    model = timm.create_model('resnet34')
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)
    