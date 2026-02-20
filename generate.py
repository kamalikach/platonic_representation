import sys
import torch
from omegaconf import OmegaConf
from generation.random import RandomGenerator
from generation.adversarial_cifar import AdversarialCIFARGenerator

GENERATORS = {
    "random": RandomGenerator,
    "adversarial_cifar": AdversarialCIFARGenerator,
}

def build_generator(cfg):
    name = cfg.generator
    if name not in GENERATORS:
        raise ValueError(f"Unknown generator '{name}'. Choose from: {list(GENERATORS.keys())}")

    if name == "adversarial_cifar":
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        return AdversarialCIFARGenerator(cfg, model)

    return GENERATORS[name](cfg)


def main(cfg_name):
    cfg_path = f"configs/generate/{cfg_name}.yaml"
    cfg = OmegaConf.load(cfg_path)

    generator = build_generator(cfg)
    ds = generator.generate()
    print(f"Generated dataset of shape: {ds.shape}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate.py <config_name>")
        print("Available configs: random, adversarial_cifar")
        sys.exit(1)

    main(sys.argv[1])

