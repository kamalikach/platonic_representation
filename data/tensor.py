from .utils import TensorDataset

def load_data(path, **kwargs):
    return TensorDataset(path)


