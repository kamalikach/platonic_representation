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

        seed = self.cfg.get("seed", 0)
        torch.manual_seed(seed)

        ds = torch.rand(self.cfg.N, self.cfg.C, self.cfg.H, self.cfg.W)
        os.makedirs(os.path.dirname(self.cfg.output_dir), exist_ok=True)
        torch.save(ds, self.cfg.output_dir)
        return ds



