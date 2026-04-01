import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


def get_dataloaders(
    batch_size=64,
    image_size=32,
    num_workers=2,
    subset_size=None,
    val_split=0.2,
):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    if subset_size is not None:
        subset_size = min(subset_size, len(dataset))
        dataset = Subset(dataset, list(range(subset_size)))

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader