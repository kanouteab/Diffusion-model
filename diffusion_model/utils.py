"""Utilitaires dataset, dataloader et checkpoints."""
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset

from .model import LegacyUNet, UNet


def _infer_checkpoint_timesteps(state, default_timesteps):
    # Ce helper isole la logique d'inférence pour simplifier la fonction de chargement.
    # Si le checkpoint contient le buffer "betas", sa longueur donne les timesteps réels.
    if isinstance(state, dict):
        # On tente d'extraire le buffer de schedule depuis le state_dict.
        betas = state.get("betas")
        if isinstance(betas, torch.Tensor) and betas.ndim == 1 and betas.numel() > 0:
            # Conversion explicite en int Python pour éviter les surprises de type.
            return int(betas.numel())
    # Sinon, on conserve la valeur demandée en entrée.
    return int(default_timesteps)


def load_model_from_checkpoint(checkpoint, device, timesteps):
    """Charge un checkpoint en tentant d'abord UNet puis LegacyUNet."""
    # Chargement brut des poids depuis disque.
    state = torch.load(checkpoint, map_location=device)
    # Support des checkpoints encapsulés: {"state_dict": ...}.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Détection automatique du nombre de timesteps compatible avec le checkpoint.
    checkpoint_timesteps = _infer_checkpoint_timesteps(state, timesteps)

    # Première tentative: architecture UNet actuelle.
    model = UNet(img_channels=3, base_channel=64, timesteps=checkpoint_timesteps).to(device)
    try:
        # Chargement strict pour détecter les incompatibilités réelles.
        model.load_state_dict(state)
    except RuntimeError:
        # Fallback LegacyUNet pour checkpoints plus anciens.
        print(f"Checkpoint incompatible avec UNet. Chargement du LegacyUNet pour {checkpoint}.")
        model = LegacyUNet(img_channels=3, base_channel=64, timesteps=checkpoint_timesteps).to(device)
        # strict=False pour tolérer les clés absentes/supplémentaires Legacy.
        model.load_state_dict(state, strict=False)
    # Le modèle est renvoyé prêt pour inférence/évaluation.
    model.eval()
    return model


import random
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset


def get_dataloader(
    batch_size=32,
    image_size=32,
    train=True,
    num_workers=2,
    subset_size=None,
    augment=True,
    seed=42,
):
    if train and augment:
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    dataset = CIFAR10(
        root="./data",
        train=train,
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
