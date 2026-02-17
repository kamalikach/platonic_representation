import torch

class ModelWrapper:
    def __init__(self, model):
        self.model = model.eval()

    def embed(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
        return out.squeeze(0).flatten()
