import torch
from torchvision.models import resnet50

def load_model(**kwargs):
    model = resnet50(pretrained=True)
    return model
