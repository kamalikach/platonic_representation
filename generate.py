import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torchvision.transforms as transforms

from generation.random import RandomGenerator
from generation.adversarial import AdversarialImageGenerator
from generation.perturb import PerturbGenerator


def build_generator(cfg: DictConfig):
    name = cfg.generator

    if name == "random":
        return RandomGenerator(cfg)

    if name == "adversarial":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = instantiate(cfg.dataset, transform=transform)
        model = instantiate(cfg.model)
        return AdversarialImageGenerator(cfg, model, dataset)

    if name == "perturb":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = instantiate(cfg.dataset, transform=transform)
        return PerturbGenerator(cfg, dataset)

    raise ValueError(f"Unknown generator '{name}'. Choose from: random, adversarial, perturb")


@hydra.main(config_path="configs/generate", config_name="config", version_base=None)
def main(cfg: DictConfig):
    generator = build_generator(cfg)
    ds = generator.generate()
    print(f"Generated dataset of shape: {ds.shape}")


if __name__ == "__main__":
    main()
