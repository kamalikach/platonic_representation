import torch
from torchvision.models import resnet50, ResNet50_Weights

def load_model(**kwargs):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    return model
