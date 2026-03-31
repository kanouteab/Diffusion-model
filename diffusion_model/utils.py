"""Utilitaires dataset et dataloader."""
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset


def get_dataloader(batch_size=32, image_size=32, train=True, num_workers=2, subset_size=None):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CIFAR10(root="./data", train=train, download=True, transform=transform)
    if subset_size is not None and subset_size > 0:
        subset_size = min(subset_size, len(dataset))
        dataset = Subset(dataset, list(range(subset_size)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
