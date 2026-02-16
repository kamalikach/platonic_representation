import torch

class ModelWrapper:
    def __init__(self, model):
        self.model = model.eval()

    def embed(self, x):
        with torch.no_grad():
            return self.model(x).flatten()
