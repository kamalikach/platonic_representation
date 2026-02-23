from model.mnist import MNISTNet, WEIGHTS_PATH
import torch
import os


def mnist_cnn(pretrained=True, **kwargs):
    """Small CNN for MNIST (1x28x28 -> 10 logits)."""
    model = MNISTNet()
    if pretrained:
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(
                f"Pretrained weights not found at {WEIGHTS_PATH}. "
                "Run `python train_mnist.py` first."
            )
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    return model
