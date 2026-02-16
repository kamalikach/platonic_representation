import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra import compose, initialize

from sampler import sample_patches
from wrapper import ModelWrapper

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    data = instantiate(cfg.dataset)
    model_a = ModelWrapper(instantiate(cfg.model_a))
    model_b = ModelWrapper(instantiate(cfg.model_b))

    print(model_a)
    print(model_b)
    print(data)

    a_list, b_list = sample_patches(model_a, model_b, data, cfg.k)

    print("Done!")
    print(a_list[:5])

if __name__ == "__main__":
    main()
