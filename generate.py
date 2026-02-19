from generation.adversarial_cifar import AdversarialCIFARGenerator
from omegaconf import OmegaConf
import timm
import torch

cfg_dict = { 'output_dir' : 'generated_datasets/cifar_adv_eps_8.pt', 
        'N': 100,
        'epsilon': 8/255,
        'target_H': 224,
        'target_W': 224,
        'device': 'cuda'
}
        # cfg.model = model used to generate adversarial examples
        # cfg.cifar_loader = dataloader yielding (images, labels)
        # cfg.target_H, cfg.target_W = output spatial size (e.g. 224, 224)
        # cfg.output_dir = path where to save the data
        # cfg.device = 'cuda' or 'cpu'





def main(cfg):
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)

    cfg = OmegaConf.create(cfg_dict)
    cfg.model = model
    generator = AdversarialCIFARGenerator(cfg)
    ds = generator.generate()
    print(ds)



if __name__ == '__main__':
    main(cfg_dict)

