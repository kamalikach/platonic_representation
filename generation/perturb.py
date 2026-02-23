from .base import DataGenerator
import torch
import os


class PerturbGenerator(DataGenerator):
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = dataset
        self.epsilon = cfg.get("epsilon", 8 / 255)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _sample_sphere_surface(self, shape):
        """Sample uniformly from the surface of a sphere of radius epsilon."""
        z = torch.randn(shape)
        z = z / z.flatten(1).norm(dim=1, keepdim=True).view(-1, *([1] * (len(shape) - 1)))
        return z * self.epsilon

    def generate(self):
        N = self.cfg.N
        images, _ = zip(*[self.dataset[i] for i in range(N)])
        images = torch.stack(images)

        perturbation = self._sample_sphere_surface(images.shape)
        perturbed = torch.clamp(images + perturbation, 0, 1)

        output_dir = self.cfg.output_dir
        dirpath = os.path.dirname(output_dir)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save(perturbed, output_dir)

        return perturbed
