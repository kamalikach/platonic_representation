from torchvision.datasets import CIFAR10
from .utils import ImagesOnlyDataset
from torchvision import transforms

def load_data(**kwargs):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),  # convert 1→3 channels
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats now valid
            std=[0.229, 0.224, 0.225]
        )
    ])

    return ImagesOnlyDataset(
        MNIST(root="./datasets/", train=False, download=True, transform=transform))










def load_data(**kwargs):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return ImagesOnlyDataset(CIFAR10(root="./datasets/", train=False, download=True, transform=transform))

