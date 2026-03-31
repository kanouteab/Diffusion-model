"""Utilitaires dataset et dataloader."""
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def get_dataloader(batch_size=32, image_size=32, train=True, num_workers=2):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CIFAR10(root="./data", train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
