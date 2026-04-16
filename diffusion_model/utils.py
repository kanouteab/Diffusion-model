"""Utilitaires dataset, dataloader et chargement de checkpoints."""
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10

from .model import LegacyUNet, UNet


def _infer_checkpoint_timesteps(state, default_timesteps):
    """Infère le nombre de timesteps à partir d'un state_dict si possible."""
    if isinstance(state, dict):
        betas = state.get("betas")
        if isinstance(betas, torch.Tensor) and betas.ndim == 1 and betas.numel() > 0:
            return int(betas.numel())
    return int(default_timesteps)


def load_model_from_checkpoint(checkpoint, device, timesteps, base_channel=64):
    """Charge un checkpoint en tentant d'abord UNet puis LegacyUNet."""
    state = torch.load(checkpoint, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    checkpoint_timesteps = _infer_checkpoint_timesteps(state, timesteps)

    model = UNet(
        img_channels=3,
        base_channel=base_channel,
        timesteps=checkpoint_timesteps,
    ).to(device)

    try:
        model.load_state_dict(state)
    except RuntimeError:
        print(f"Checkpoint incompatible avec UNet. Chargement du LegacyUNet pour {checkpoint}.")
        model = LegacyUNet(
            img_channels=3,
            base_channel=base_channel,
            timesteps=checkpoint_timesteps,
        ).to(device)
        model.load_state_dict(state, strict=False)

    model.eval()
    return model


def _build_transform(image_size=32):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_dataloader(batch_size=32, image_size=32, train=True, num_workers=2, subset_size=None):
    """Retourne un seul DataLoader CIFAR-10."""
    transform = _build_transform(image_size)

    dataset = CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform,
    )

    if subset_size is not None and subset_size > 0:
        subset_size = min(subset_size, len(dataset))
        dataset = Subset(dataset, list(range(subset_size)))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_dataloaders(
    batch_size=32,
    image_size=32,
    num_workers=2,
    subset_size=None,
    val_split=0.0,
):
    """
    Retourne train_loader et val_loader.
    Si val_split <= 0, val_loader = None.
    """
    transform = _build_transform(image_size)

    dataset = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    if subset_size is not None and subset_size > 0:
        subset_size = min(subset_size, len(dataset))
        dataset = Subset(dataset, list(range(subset_size)))

    if val_split is None or val_split <= 0:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, None

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    if val_size == 0:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, None

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