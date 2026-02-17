import torch
from torchvision.models import alexnet

def load_model(**kwargs):
    model = alexnet(pretrained=True)
    return model
