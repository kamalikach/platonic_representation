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
    model_a = ModelWrapper(instantiate(cfg.model_a).to(device))
    model_b = ModelWrapper(instantiate(cfg.model_b).to(device))

    print(model_a)
    print(model_b)
    print(data)

    a_list, b_list = sample_patches(model_a, model_b, data, cfg.k, device)

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
