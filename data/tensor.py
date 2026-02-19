from .utils import TensorDataset

def load_data(**kwargs):
    return TensorDataset(kwargs['path'])



