import torch 
from torch.utils.data import Dataset

class ImagesOnlyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image

class TensorDataset(Dataset):
    """Load a pre-generated dataset saved as a .pt tensor of shape (N, C, H, W)."""

    def __init__(self, path):
        self.data = torch.load(path)  # (N, C, H, W)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0)  # (1, C, H, W), ready for model(d[i].to(device))
