import torch
import torch.nn as nn
import os

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "mnist_cnn.pt")


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(pretrained=True, **kwargs):
    model = MNISTNet()
    if pretrained:
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(
                f"Pretrained weights not found at {WEIGHTS_PATH}. "
                "Run `python train_mnist.py` first."
            )
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    return model
