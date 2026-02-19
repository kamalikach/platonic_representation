from generation.random import RandomGenerator
from omegaconf import OmegaConf

cfg_dict = { 'output_dir' : 'generated_datasets/random.pt', 
        'N': 100, 
        'C': 3,
        'H': 224,
        'W': 224
}

def main(cfg):
    cfg = OmegaConf.create(cfg_dict)

    generator = RandomGenerator(cfg)
    ds = generator.generate()
    print(ds)



if __name__ == '__main__':
    main(cfg_dict)

