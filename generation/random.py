from .base import DataGenerator
import torch
import os

class RandomGenerator(DataGenerator):
    def __init__(self, cfg):
        self.cfg = cfg

    def generate(self):
        #cfg.N = number of data points
        #cfg.C, cfg.H, cfg.W = each data point is C x H X W
        #cfg.output_dir = path where to save the data

        seed = cfg.get("seed", 0)
        torch.manual_seed(seed)

        ds = torch.rand(cfg.N, cfg.C, cfg.H, cfg.W)
        os.makedirs(os.path.dirname(cfg.output_dir), exist_ok=True)
        torch.save(ds, cfg.output_dir)
        return ds



