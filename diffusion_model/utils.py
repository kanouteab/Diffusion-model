"""Utilitaires dataset, dataloader et checkpoints."""
import random
import torch
import torchvision.transforms as T
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Subset

from .model import LegacyUNet, UNet


def _infer_checkpoint_timesteps(state, default_timesteps):
    if isinstance(state, dict):
        betas = state.get("betas")
        if isinstance(betas, torch.Tensor) and betas.ndim == 1 and betas.numel() > 0:
            return int(betas.numel())
    return int(default_timesteps)


def load_model_from_checkpoint(checkpoint, device, timesteps):
    """Charge un checkpoint en tentant d'abord UNet puis LegacyUNet."""
    state = torch.load(checkpoint, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    checkpoint_timesteps = _infer_checkpoint_timesteps(state, timesteps)

    model = UNet(img_channels=3, base_channel=64, timesteps=checkpoint_timesteps).to(device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        print(f"Checkpoint incompatible avec UNet. Chargement du LegacyUNet pour {checkpoint}.")
        model = LegacyUNet(img_channels=3, base_channel=64, timesteps=checkpoint_timesteps).to(device)
        model.load_state_dict(state, strict=False)

    model.eval()
    return model


def get_dataloader(
    batch_size=32,
    image_size=64,
    train=True,
    num_workers=2,
    subset_size=None,
    augment=True,
    seed=42,
):
    """
    DataLoader CelebA avec :
    - resize
    - center crop
    - normalisation
    - augmentation légère sur train
    - sous-échantillonnage aléatoire reproductible
    """
    if train and augment:
        transform = T.Compose([
            T.CenterCrop(178),
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = T.Compose([
            T.CenterCrop(178),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    split = "train" if train else "valid"

    dataset = CelebA(
        root="./data",
        split=split,
        target_type="attr",
        download=True,
        transform=transform,
    )

    if subset_size is not None and subset_size > 0:
        subset_size = min(subset_size, len(dataset))
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), subset_size)
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )
