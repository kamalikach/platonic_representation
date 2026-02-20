# generation/adversarial_cifar.py
from .base import DataGenerator
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os


class AdversarialCIFARGenerator(DataGenerator):
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.epsilon = cfg.get("epsilon", 8 / 255)
        self.method = cfg.get("method", "pgd")
        self.pgd_steps = cfg.get("pgd_steps", 5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _fgsm(self, images, labels):
        images = images.clone().detach().to(self.device).requires_grad_(True)
        labels = labels.to(self.device)

        loss = nn.CrossEntropyLoss()(self.model(images), labels)
        self.model.zero_grad()
        loss.backward()

        adversarial = images + self.epsilon * images.grad.sign()
        adversarial = torch.clamp(adversarial, 0, 1).detach()
        return adversarial

    def _pgd(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.to(self.device)
        step_size = self.epsilon / self.pgd_steps * 2

        adversarial = images.clone().detach()
        adversarial += torch.empty_like(adversarial).uniform_(-self.epsilon, self.epsilon)
        adversarial = torch.clamp(adversarial, 0, 1).detach()

        for _ in range(self.pgd_steps):
            adversarial.requires_grad_(True)
            loss = nn.CrossEntropyLoss()(self.model(adversarial), labels)
            self.model.zero_grad()
            loss.backward()

            adversarial = adversarial + step_size * adversarial.grad.sign()
            delta = torch.clamp(adversarial - images, -self.epsilon, self.epsilon)
            adversarial = torch.clamp(images + delta, 0, 1).detach()

        return adversarial

    def generate(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        test_set = torchvision.datasets.CIFAR10(
            root='./datasets', train=False, download=True, transform=transform
        )

        N = self.cfg.N
        images, labels = zip(*[test_set[i] for i in range(N)])
        images = torch.stack(images)   # (N, 3, 32, 32)
        labels = torch.tensor(labels)  # (N,)

        if self.method == "fgsm":
            adversarial_images = self._fgsm(images, labels)
        elif self.method == "pgd":
            adversarial_images = self._pgd(images, labels)
        else:
            raise ValueError(f"Unknown attack method: {self.method}")

        output_dir = self.cfg.output_dir
        dirpath = os.path.dirname(output_dir)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save(adversarial_images, output_dir)

        return adversarial_images
