import torch
from torchvision.models import alexnet, AlexNet_Weights

def load_model(**kwargs):
    model = alexnet(weights=AlexNet_Weights.DEFAULT)
    return model
