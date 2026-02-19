from generation.random import RandomGenerator
from omegaconf import OmegaConf

cfg_dict = {
    'output_dir': 'generated_datasets/random_32_32.pt',
    'N': 100,
    'C': 3,
    'H': 32,
    'W': 32,
    'seed': 0
}


def main(cfg):
    cfg = OmegaConf.create(cfg_dict)
    generator = RandomGenerator(cfg)
    ds = generator.generate()
    print(ds)


if __name__ == '__main__':
    main(cfg_dict)
