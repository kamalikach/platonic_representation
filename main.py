import hydra
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra import compose, initialize

from sampler import sample_patches
from wrapper import ModelWrapper

from datetime import datetime

@hydra.main(version_base=None, config_path="configs/eval/", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = instantiate(cfg.data)
    model_a = ModelWrapper(instantiate(cfg.model_a.model).to(device))
    model_b = ModelWrapper(instantiate(cfg.model_b.model).to(device))

    # Instantiate transforms if they exist in config
    transform_a = instantiate(cfg.model_a.transform) if "transform" in cfg.model_a else None
    transform_b = instantiate(cfg.model_b.transform) if "transform" in cfg.model_b else None

    print(model_a)
    print(model_b)
    print(data)
    print(f"Transform A: {transform_a}")
    print(f"Transform B: {transform_b}")

    a_list, b_list = sample_patches(model_a, model_b, data, cfg.k, device,
                                     transform_a=transform_a,
                                     transform_b=transform_b)

    print("Done!")
    print(a_list[:5])

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_fname = "./outputs/" + f"{cfg.data.name}_{cfg.model_a.name}_{cfg.model_b.name}_{cfg.k}_{timestamp}" + ".txt"
    print(output_fname)
    with open(output_fname, "w") as f:
        for x, y in zip(a_list, b_list):
            f.write(f"{x},{y}\n")


if __name__ == "__main__":
    main()
